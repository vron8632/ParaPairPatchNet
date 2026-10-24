#!/usr/bin/env python3
"""
SAC (Segment and Complete, Liu et al. CVPR 2022) 统一协议评测
================================================================
SAC 是"检测 + 抹除"防御：其 patch segmenter（U-Net）输出像素级 patch 掩码。
本脚本把其**检测部分**适配为图像级判据（两种候选分数）：
  (a) seg_max : segmenter 概率图的最大值（连续置信度）
  (b) mask_frac: 二值掩码的像素占比
在统一评测集（138 正 + 4 源各 250）上报告三源/四源双口径 TPR@FPR。
权重：ckpts/coco_at.pth（官方 COCO 自对抗训练 segmenter）。
输出: results/tvc/sac_unified.json

⚠️ 已知障碍（2026-10-16 核验）：
  - 官方仓库与依赖均可得（joellliu/SegmentAndComplete），COCO 权重可加载；
  - 但其面向物理补丁的 **APRICOT-Mask 专用 segmenter 权重托管在 Google Drive**（实测不可达），
    APRICOT-Mask 数据集 FTP/OneDrive 亦不可达；
  - 实测：COCO 权重在 APRICOT 物理补丁域几乎无响应（官方输入约定 [0,1] 下 seg_max≈0，
    正负样本均无区分度），与 SAC 论文"需用 APRICOT-Mask 权重"的说明一致。
  → 因此默认不运行；若后续获得 APRICOT-Mask 权重，用 `--weights <path>` 复跑。
"""
import os, sys, json, math
import numpy as np

WS = "/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard"
TVC = f"{WS}/results/tvc"
EVAL = f"{TVC}/unified_eval_set"
SAC_DIR = f"{WS}/code/third_party/SegmentAndComplete"
S3 = ["casia_au", "coco_val", "inria_noped"]
S4 = S3 + ["columbia_au"]


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k/n; d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (round(max(0.0, c-h), 4), round(min(1.0, c+h), 4))


def main():
    import torch
    from PIL import Image
    from kornia.geometry.transform import Resize
    sys.path.insert(0, SAC_DIR)
    os.chdir(SAC_DIR)                     # 仓库内相对导入
    from patch_detector import PatchDetector
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PatchDetector(3, 1, base_filter=16, square_sizes=[125, 100, 75, 50, 25],
                          n_patch=1, device=dev)
    sd = torch.load("ckpts/coco_at.pth", map_location="cpu", weights_only=False)
    model.unet.load_state_dict(sd)
    model = model.to(dev).eval()
    print("[sac] segmenter 权重加载完成", flush=True)

    @torch.no_grad()
    def scores(im):
        """返回 (seg_max, mask_frac)：在 1024 长边输入上推理"""
        im = im.copy(); im.thumbnail((1024, 1024))
        x = torch.from_numpy(np.asarray(im).copy()).float().permute(2, 0, 1) / 255.0
        xb = x.unsqueeze(0).to(dev)
        prob = torch.sigmoid(model.unet(xb))
        seg_max = float(prob.max())
        mask_frac = float((prob > 0.5).float().mean())
        return seg_max, mask_frac

    def score_dir(d):
        seg, frac = [], []
        fs = sorted(os.listdir(d))
        for i, f in enumerate(fs):
            a, b = scores(Image.open(os.path.join(d, f)).convert("RGB"))
            seg.append(a); frac.append(b)
            if (i+1) % 200 == 0: print(f"    {os.path.basename(d)} {i+1}/{len(fs)}", flush=True)
        return np.array(seg), np.array(frac)

    ps, pf = score_dir(f"{EVAL}/pos")
    ns, nf = {}, {}
    for s in S4:
        a, b = score_dir(f"{EVAL}/{s}"); ns[s], nf[s] = a, b
    res = {"model": "SAC (Segment and Complete, CVPR'22) patch segmenter",
           "weights": "ckpts/coco_at.pth", "protocol": "unified_eval_set",
           "n_pos": len(ps), "variants": {}}
    for tag, p, n in (("seg_max", ps, ns), ("mask_frac", pf, nf)):
        row = {"pos_mean": round(float(p.mean()), 4),
               "neg_mean": {k: round(float(v.mean()), 4) for k, v in n.items()}, "tpr": {}}
        for scope, keys in (("3src", S3), ("4src", S4)):
            n_all = np.concatenate([n[k] for k in keys]); rr = {}
            for a in (0.01, 0.05, 0.10, 0.20):
                thr = float(np.percentile(n_all, 100*(1-a))); k_ = int((p >= thr).sum())
                rr[f"@{int(a*100)}pct"] = {"tpr": round(k_/len(p), 4), "wilson": wilson(k_, len(p)),
                                           "thr": round(thr, 4)}
            row["tpr"][scope] = rr
        res["variants"][tag] = row
        print(f"  [{tag}] 3src: " + "  ".join(f"{a}={row['tpr']['3src'][f'@{a}pct']['tpr']:.3f}" for a in (1,5,10,20)), flush=True)
        print(f"  [{tag}] 4src: " + "  ".join(f"{a}={row['tpr']['4src'][f'@{a}pct']['tpr']:.3f}" for a in (1,5,10,20)), flush=True)
    json.dump(res, open(f"{TVC}/sac_unified.json", "w"), indent=1, ensure_ascii=False)
    print(f"→ {TVC}/sac_unified.json")


if __name__ == "__main__":
    main()
