#!/usr/bin/env python3
"""
PAD 基线复现 v2（绝对尺度版）—— L1 直接对比
==============================================
v1 的问题（已实测确认）：PAD 的 `fuse_heatmap` 内部做**逐图 min-max 归一化**，
使得每张图的融合热图 99 分位都趋近 1 → 图像级分数**跨图不可比**：
实测正样本均值 0.7942 vs 负样本 0.7997，完全重叠、无区分力。

v2 改法：直接使用 `img_heatmap_mi` / `img_heatmap_cd` 的**归一化前原始量**
（互信息有绝对量纲 bits；重压缩残差有绝对量级），并同时输出多个候选图像级分数，
以便判断 PAD 是否在**任何**绝对判据下具备低误报检测能力。

原理提示（PAD, CVPR'24）：
  补丁区域满足 (a) 语义独立——与相邻区域的互信息接近零；
  (b) 空间异质——重压缩残差异常。故候选分数同时覆盖 mi 的高/低分位与均值。

用法:
  $PY code/experiments_tvc/pad_baseline.py --n-pos 100 --n-per-source 60 --max-side 512
"""
import argparse, json, os, sys, time, tempfile

WS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(WS, "results", "tvc")
PAD_DIR = os.path.join(WS, "code", "third_party", "PAD")
D = "/media/oyp/工作/Projects/042_image_forensic/DuetGuard/data"
MM = "/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard"
APRICOT_D = f"{MM}/data/apricot/dev/data_mask_v2"
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

CANDIDATES = ["mi_mean", "mi_p01", "mi_p99", "mi_std", "cd_mean", "cd_p99", "cd_std"]


def collect(n_pos, n_per_source, seed=42, max_side=512):
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
            im = Image.fromarray(arr).convert("RGB"); im.thumbnail((max_side, max_side))
            pos.append((f"apricot_patch/{f}", im))
    print(f"  pos: {len(pos)}", flush=True)
    au = f"{D}/casia2/CASIA2.0_revised/Au"
    fs = sorted(f for f in os.listdir(au) if f.lower().endswith(IMG_EXT))
    neg["casia_au"] = [(f"casia_au/{fs[i]}", Image.open(os.path.join(au, fs[i])).convert("RGB"))
                       for i in rng.permutation(len(fs))[:n_per_source]]
    coco = f"{D}/coco/val2017"
    fs = sorted(f for f in os.listdir(coco) if f.lower().endswith(IMG_EXT))
    neg["coco_val"] = [(f"coco_val/{fs[i]}", Image.open(os.path.join(coco, fs[i])).convert("RGB"))
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
        neg["columbia_au"] = [(f"columbia_au/{os.path.basename(p)}", Image.open(p).convert("RGB"))
                              for p in items[:n_per_source]]
    inria = f"{D}/inria/data_ped/no_pedestrians"
    if os.path.isdir(inria):
        fs = sorted(f for f in os.listdir(inria) if f.lower().endswith(IMG_EXT))
        neg["inria_noped"] = [(f"inria_noped/{fs[i]}", Image.open(os.path.join(inria, fs[i])).convert("RGB"))
                              for i in rng.permutation(len(fs))[:n_per_source]]
    for k, v in neg.items():
        for _, im in v:
            im.thumbnail((max_side, max_side))
    print(f"  neg: { {k: len(v) for k, v in neg.items()} }", flush=True)
    return pos, neg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pos", type=int, default=100)
    ap.add_argument("--n-per-source", type=int, default=60)
    ap.add_argument("--max-side", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(OUT, "pad_baseline.json"))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
    import sys as _sys
    _sys.path.insert(0, PAD_DIR)
    from heatmap_MI import img_heatmap_mi
    from heatmap_CD import img_heatmap_cd

    nan_count = {"n": 0}

    def pad_candidates(impath):
        """返回多个**绝对尺度**候选分数（不做逐图归一化）。"""
        h_mi = np.asarray(img_heatmap_mi(impath), dtype=np.float64).ravel()
        h_cd, _qt = img_heatmap_cd(impath)
        h_cd = np.nanmean(np.asarray(h_cd, dtype=np.float64), axis=0).ravel()
        if np.isnan(h_mi).any() or np.isnan(h_cd).any():
            nan_count["n"] += 1
        h_mi = np.nan_to_num(h_mi, nan=0.0)
        h_cd = np.nan_to_num(h_cd, nan=0.0)
        out = {}
        for tag, a in (("mi", h_mi), ("cd", h_cd)):
            out[f"{tag}_mean"] = float(np.mean(a))
            out[f"{tag}_std"] = float(np.std(a))
            out[f"{tag}_p01"] = float(np.percentile(a, 1))
            out[f"{tag}_p99"] = float(np.percentile(a, 99))
        return out

    pos, neg = collect(args.n_pos, args.n_per_source, max_side=args.max_side)
    tmpf = os.path.join(tempfile.gettempdir(), "pad_in.png")
    t0 = time.time()

    def run(items, tag):
        rows = []
        for i, (nm, im) in enumerate(items):
            im = im.copy(); im.thumbnail((args.max_side, args.max_side)); im.save(tmpf)
            rows.append(pad_candidates(tmpf))
            if (i + 1) % 20 == 0 or i == 0:
                print(f"  {tag} {i+1}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
            if args.limit and (i + 1) >= args.limit:
                break
        return rows

    pos_rows = run(pos, "pos")
    neg_rows = {src: run(items, src) for src, items in neg.items()}

    def arr(rows, c): return np.array([r[c] for r in rows])
    res = {"method": "PAD (CVPR 2024)", "variant": "absolute-scale (no per-image normalization)",
           "n_pos": len(pos_rows), "max_side": args.max_side,
           "n_neg": sum(len(v) for v in neg_rows.values()),
           "nan_safe_count": nan_count["n"], "candidates": CANDIDATES, "results": {}}
    yn_rows = [r for v in neg_rows.values() for r in v]
    print("\n=== 各候选绝对分数的可分性（TPR@FPR）===")
    best = None
    for c in CANDIDATES:
        p = arr(pos_rows, c); n = arr(yn_rows, c)
        row = {"pos_mean": round(float(p.mean()), 4), "neg_mean": round(float(n.mean()), 4)}
        for a in (0.05, 0.10, 0.20, 0.33):
            thr = float(np.percentile(n, 100 * (1 - a)))
            row[f"@FPR{int(a*100)}pct"] = {"threshold": round(thr, 6),
                                           "tpr": round(float((p >= thr).mean()), 4)}
        # 也试反向（低值=补丁的候选，如"语义独立"⇒互信息低）
        for a in (0.05, 0.10, 0.20):
            thr_lo = float(np.percentile(n, 100 * a))
            row[f"@{int(a*100)}pct_rev"] = round(float((p <= thr_lo).mean()), 4)
        res["results"][c] = row
        t5 = row["@FPR5pct"]["tpr"]; tr5 = row.get("@5pct_rev", 0)
        print(f"  {c:9s} pos={row['pos_mean']:.3f} neg={row['neg_mean']:.3f} "
              f"@5%={t5:.3f} @10%={row['@FPR10pct']['tpr']:.3f} @20%={row['@FPR20pct']['tpr']:.3f} "
              f"| 反向@5%={tr5:.3f}")
        for key in ("@FPR5pct", "@FPR10pct"):
            if best is None or row[key]["tpr"] > best[1]:
                best = (c, row[key]["tpr"], key)
    res["best_candidate"] = {"name": best[0], "tpr": best[1], "at": best[2]}
    print(f"\n最优候选：{best[0]}  TPR={best[1]:.4f} ({best[2]})")
    # 保存逐样本候选分数，便于后续分析
    res["pos_rows"] = pos_rows
    res["neg_rows_by_source"] = neg_rows
    json.dump(res, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"[pad] wrote {args.out}")


if __name__ == "__main__":
    main()
