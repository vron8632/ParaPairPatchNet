#!/usr/bin/env python3
"""
创新 #5 — Patch-level 跨块一致性 + 定位热图（既回应 R#6，又补上"定位"能力）
=============================================================================
原论文的 cross-instance consistency 有两个弱点（先前分析已指出）：
  (1) 只在 K>=2 生效，而真实数据常是 K<=1（APRICOT 全为 K=0）
  (2) 只给图像级判定，不给"补丁在哪"

本方法把「一致性」从**实例级**下移到**patch 块级**：
  1. 图 resize 到 518，DINOv2 前向得到 37x37 的 patch tokens（14x14 像素/块）
  2. 对每个块算它与「全局中位数块」的余弦相似度
  3. 异常热图 A = 1 - cos_sim；补丁区域因"外来材质"与全图统计不一致而高亮
  4. 用 GT mask 评 IoU / pointing-game，并给出图像级分数（热图 max/mean）

优点: 不依赖实例分割(K 无关)、可定位、与 SPN/DCT/CLIP 分支正交。

用法:
  $PY code/experiments_tvc/patch_consistency_loc.py --n 100
输出: results/tvc/patch_consistency_loc.json
"""
import argparse, json, os, sys, time
import numpy as np

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)
OUT = os.path.join(os.path.dirname(CODE_DIR), "results", "tvc")
DINO = "/media/oyp/工作/Projects/030_mllm/DINOv2/facebook/dinov2-base"
APRICOT_D = "/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard/data/apricot/dev/data_mask_v2"


def heatmap(model, img_t, size=518, device="cuda"):
    """返回 (grid_h, grid_w, D) 的 patch 特征图。"""
    import torch
    import torch.nn.functional as F
    x = F.interpolate(img_t.unsqueeze(0).float(), size=(size, size),
                      mode="bilinear", align_corners=False).to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    x = (x - mean) / std
    with torch.no_grad():
        o = model(x).last_hidden_state[:, 1:]        # (1, P, D)
    P = o.shape[1]
    g = int(P ** 0.5)
    return o[0].reshape(g, g, -1), g


def anomaly_map(feat, g):
    """块级一致性异常图: 1 - cos(block, median_block)，并做局部对比增强。"""
    import torch
    import torch.nn.functional as F
    f = F.normalize(feat.float(), dim=-1)             # (g,g,D)
    med = f.reshape(-1, f.shape[-1]).median(0).values
    med = F.normalize(med, dim=-1)
    sim = (f * med).sum(-1)                            # (g,g)
    anom = 1 - sim
    # 局部对比: 减去 3x3 邻域均值，突出"局部不一致"而非全局偏移
    k = torch.ones(1, 1, 3, 3, device=anom.device) / 9.0
    local = F.conv2d(anom.view(1, 1, g, g), k, padding=1).view(g, g)
    return (anom - local).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--size", type=int, default=518)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    import torch
    from PIL import Image
    from transformers import AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(DINO, local_files_only=True).to(device).eval()

    files = sorted(os.listdir(APRICOT_D), key=lambda f: int(f.split("_")[1].split(".")[0]))
    assert len(files) > 0, APRICOT_D

    g = args.size // 14
    rows = []
    t0 = time.time()
    for k, f in enumerate(files[:args.n]):
        x = torch.load(os.path.join(APRICOT_D, f), map_location="cpu", weights_only=False)
        img = x["Image"][0]                            # (H,W,3) float [0,1]
        mask = np.asarray(x["Mask"])                   # (H,W)
        if mask.max() <= 0:
            continue
        img_t = torch.from_numpy(img).permute(2, 0, 1).float()
        H, W = mask.shape
        feat, gg = heatmap(model, img_t, args.size, device)
        a = anomaly_map(feat, gg)                      # (gg,gg)

        # GT mask 下采样到 patch 网格
        mt = torch.from_numpy(mask.astype(np.float32)).view(1, 1, H, W)
        import torch.nn.functional as F
        m_small = F.interpolate(mt, size=(gg, gg), mode="area").view(gg, gg).numpy()
        gt = (m_small > 0.3)

        # 定位指标: patch 级 IoU（对异常图取最高分、与 GT 面积相当的块）
        n_gt = max(int(gt.sum()), 1)
        thresh = np.sort(a.ravel())[::-1][min(n_gt, a.size) - 1]
        pred = a >= thresh
        inter = (pred & gt).sum()
        union = (pred | gt).sum()
        iou = inter / max(union, 1)
        # pointing game: 最高分块是否落在 GT 内
        hit = bool(gt[np.unravel_index(np.argmax(a), a.shape)])

        rows.append({"file": f, "iou": float(iou), "hit": hit,
                     "anom_max": float(a.max()), "anom_mean": float(a.mean()),
                     "gt_frac": float(gt.mean())})
        if (k + 1) % 20 == 0:
            print(f"  {k+1}/{len(files[:args.n])} ({time.time()-t0:.0f}s) "
                  f"IoU_mean={np.mean([r['iou'] for r in rows]):.3f} "
                  f"PG={np.mean([r['hit'] for r in rows]):.3f}", flush=True)

    ious = [r["iou"] for r in rows]
    hits = [r["hit"] for r in rows]
    rep = {"method": "patch-level cross-block consistency (DINOv2 patch tokens, K-independent)",
           "n": len(rows), "grid": f"{g}x{g}",
           "iou_mean": round(float(np.mean(ious)), 4),
           "iou_median": round(float(np.median(ious)), 4),
           "pointing_game": round(float(np.mean(hits)), 4),
           "iou_gt_0.1": round(float(np.mean([i > 0.1 for i in ious])), 4),
           "iou_gt_0.25": round(float(np.mean([i > 0.25 for i in ious])), 4),
           "seconds": round(time.time() - t0, 1), "rows": rows}
    out = args.out or os.path.join(OUT, "patch_consistency_loc.json")
    json.dump(rep, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"\n[loc] n={rep['n']} grid={rep['grid']}")
    print(f"      IoU mean={rep['iou_mean']} median={rep['iou_median']} "
          f"| IoU>0.1 = {rep['iou_gt_0.1']} | IoU>0.25 = {rep['iou_gt_0.25']}")
    print(f"      Pointing Game = {rep['pointing_game']}")
    print(f"[loc] wrote {out}")


if __name__ == "__main__":
    main()
