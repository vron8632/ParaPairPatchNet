#!/usr/bin/env python3
"""
NutNet 基线复现（CCS 2024）—— L1 直接对比
==========================================
判据（源自上游 patchFilter.py）：把图像 resize 到 416x416，切成 32x32 个 13x13 的块，
每块过预训练自编码器重建，用 **MSE 重建误差**衡量：
  mask1 = loss > 0.2 ; mask2 = |out-in|.sum(dim=1) > 0.25 ; patch_block = mask1 & mask2
本脚本**只取图像级分数**（不跑 YOLO 检测器）：score = max_block_loss（以及补丁块占比）。

评估协议与 fair_comparison.py 一致（同一 seed / 同一批图）。

用法:
  $PY code/experiments_tvc/nutnet_baseline.py --n-pos 100 --n-per-source 60
"""
import argparse, json, os, sys, time

WS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(WS, "results", "tvc")
AE_DIR = os.path.join(WS, "code", "third_party", "NutNet", "ae_weights")
D = "/media/oyp/工作/Projects/042_image_forensic/DuetGuard/data"
MM = "/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard"
APRICOT_D = f"{MM}/data/apricot/dev/data_mask_v2"
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def build_ae32():
    """复刻上游 AutoEncoder32（13x13 块）。"""
    import torch.nn as nn

    class AutoEncoder32(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=2, stride=1, padding=1), nn.ReLU(),
                nn.Conv2d(8, 16, kernel_size=2, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=2, stride=2, padding=1), nn.ReLU())
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=1), nn.ReLU(),
                nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2, padding=1), nn.ReLU(),
                nn.ConvTranspose2d(8, 3, kernel_size=2, stride=1, padding=1), nn.Tanh())

        def forward(self, x):
            return self.decoder(self.encoder(x))
    return AutoEncoder32()


def collect(n_pos, n_per_source, seed=42, max_side=1024):
    import numpy as np, torch
    from PIL import Image
    rng = np.random.RandomState(seed)
    pos, neg = [], {}
    files = sorted(os.listdir(APRICOT_D), key=lambda f: int(f.split("_")[1].split(".")[0]))
    for f in files:
        if len(pos) >= n_pos:
            break
        x = torch.load(os.path.join(APRICOT_D, f), map_location="cpu", weights_only=False)
        if x["Mask"].sum() > 0:
            arr = np.clip(x["Image"][0].astype(np.float32) * 255.0, 0, 255).astype(np.uint8)
            pos.append((f"apricot_patch/{f}", Image.fromarray(arr).convert("RGB")))
    print(f"  pos: {len(pos)}", flush=True)
    au = f"{D}/casia2/CASIA2.0_revised/Au"
    fs = sorted(f for f in os.listdir(au) if f.lower().endswith(IMG_EXT))
    neg["casia_au"] = [Image.open(os.path.join(au, fs[i])).convert("RGB")
                       for i in rng.permutation(len(fs))[:n_per_source]]
    coco = f"{D}/coco/val2017"
    fs = sorted(f for f in os.listdir(coco) if f.lower().endswith(IMG_EXT))
    neg["coco_val"] = [Image.open(os.path.join(coco, fs[i])).convert("RGB")
                       for i in rng.permutation(len(fs))[:n_per_source]]
    col = f"{D}/columbia/ImSpliceDataset"
    if os.path.isdir(col) and n_per_source:
        items = []
        for sub in sorted(os.listdir(col)):
            d2 = os.path.join(col, sub)
            if not os.path.isdir(d2) or sub.startswith("Au-S"):
                continue
            for fn in sorted(os.listdir(d2)):
                if fn.lower().endswith(IMG_EXT):
                    items.append(os.path.join(d2, fn))
        rng.shuffle(items)
        neg["columbia_au"] = [Image.open(p).convert("RGB") for p in items[:n_per_source]]
    inria = f"{D}/inria/data_ped/no_pedestrians"
    if os.path.isdir(inria):
        fs = sorted(f for f in os.listdir(inria) if f.lower().endswith(IMG_EXT))
        neg["inria_noped"] = [Image.open(os.path.join(inria, fs[i])).convert("RGB")
                              for i in rng.permutation(len(fs))[:n_per_source]]
    print(f"  neg: { {k: len(v) for k, v in neg.items()} }", flush=True)
    return pos, neg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pos", type=int, default=100)
    ap.add_argument("--n-per-source", type=int, default=60)
    ap.add_argument("--inp-dim", type=int, default=416)
    ap.add_argument("--box-num", type=int, default=32)
    ap.add_argument("--ae", default=os.path.join(AE_DIR, "n_13.pth"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(OUT, "nutnet_baseline.json"))
    args = ap.parse_args()

    import numpy as np
    import torch
    import torch.nn as nn
    from PIL import Image

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ae = build_ae32().to(dev)
    ae.load_state_dict(torch.load(args.ae, map_location=dev))
    ae.eval()
    bl = args.inp_dim // args.box_num
    print(f"[nutnet] ae={os.path.basename(args.ae)} inp={args.inp_dim} box={args.box_num} bl={bl} dev={dev}")

    loss_fn = nn.MSELoss(reduction="none")

    def score(im):
        im = im.convert("RGB").resize((args.inp_dim, args.inp_dim), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(im).copy()).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        x = (x * 2 - 1).to(dev)
        with torch.no_grad():
            blk = x.unfold(2, bl, bl).unfold(3, bl, bl)                # [1,3,bn,bn,bl,bl]
            blk = blk.permute(0, 2, 3, 4, 5, 1).contiguous().view(-1, bl, bl, 3).permute(0, 3, 1, 2)
            out = ae(blk)
            per_block = torch.mean(loss_fn(out, blk), dim=(1, 2, 3))   # [bn*bn]
            # 维度对齐（上游用 unsqueeze+expand）：per_block [N] 与 delta [N,bl,bl]
            delta = torch.abs(out - blk).sum(dim=1)                  # [N, bl, bl]
            m1 = (per_block > 0.2).float().view(-1, 1, 1)            # [N, 1, 1]
            m2 = (delta > 0.25).float()                              # [N, bl, bl]
            mask = m1 * m2                                           # [N, bl, bl]
            patch_block_frac = float((mask.sum(dim=(1, 2)) > 0).float().mean())
        return float(per_block.max()), patch_block_frac

    pos, neg = collect(args.n_pos, args.n_per_source)
    t0 = time.time()
    pos_s = []
    for i, (nm, im) in enumerate(pos):
        pos_s.append(score(im)[0])
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  pos {i+1}/{len(pos)} ({time.time()-t0:.0f}s)", flush=True)
        if args.limit and (i + 1) >= args.limit:
            print("[limit] 提前停止"); break
    neg_s = {}
    for src, items in neg.items():
        ss = [score(im)[0] for im in items]
        neg_s[src] = ss
        print(f"  {src} {len(ss)} done ({time.time()-t0:.0f}s)", flush=True)

    p = np.array(pos_s); yn = np.concatenate([np.array(v) for v in neg_s.values()])
    res = {"method": "NutNet (CCS 2024)", "n_pos": len(p), "n_neg": len(yn),
           "pos_mean": round(float(p.mean()), 4), "neg_mean_all": round(float(yn.mean()), 4),
           "neg_per_source": {k: {"n": len(v), "mean": round(float(np.mean(v)), 4),
                                  "median": round(float(np.median(v)), 4),
                                  "p90": round(float(np.percentile(v, 90)), 4)}
                              for k, v in neg_s.items()},
           "note": "分数 = 416x416 下 32x32 个 13x13 块的 AE 最大重建 MSE（未用 YOLO 检测器）",
           "ae_weight": os.path.basename(args.ae),
           "pos_scores": [round(float(x), 6) for x in p],
           "neg_scores_by_source": {k: [round(float(x), 6) for x in v] for k, v in neg_s.items()}}
    print("\n=== NutNet 结果 ===")
    print(f"pos mean={p.mean():.4f}  neg mean={yn.mean():.4f}")
    for a in (0.01, 0.05, 0.10, 0.20, 0.33):
        thr = float(np.percentile(yn, 100 * (1 - a)))
        res[f"@FPR{int(a*100)}pct"] = {"threshold": round(thr, 4),
                                       "tpr": round(float((p >= thr).mean()), 4)}
        print(f"  目标 FPR={int(a*100):2d}%: τ={thr:.4f} TPR={res[f'@FPR{int(a*100)}pct']['tpr']:.4f}")
    json.dump(res, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"[nutnet] wrote {args.out}")


if __name__ == "__main__":
    main()
