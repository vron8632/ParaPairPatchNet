#!/usr/bin/env python3
"""
M2 重尾归因（CPU 实验，可与 GPU 训练并行）
==========================================
现象：不同来源干净图的 P(YES) 中位数几乎相同（0.032–0.050），但 p90 相差 8–18×
      （columbia 0.677 vs 其它 0.038–0.087）。
问题：为什么是这些图拿到高分？是否由某个**可测量的图像统计量**驱动？

做法：
  1. 精确重现 confound_check.py 的采样（seed=42, n_per=60，同一 RandomState 消耗顺序）
  2. 对每张负样本计算图像统计量：尺寸、亮度均值/σ、Laplacian 方差、灰度熵、饱和度
  3. 与 confound_main_withscores.json 的 P(YES) 做 Spearman 相关 + 分桶看 FPR

用法:
  $PY code/experiments_tvc/mech_tail_attribution.py \
      --scores-json results/tvc/confound_main_withscores.json \
      --out results/tvc/mech_tail_attribution.json
"""
import argparse, json, os, sys
import numpy as np

D = "/media/oyp/工作/Projects/042_image_forensic/DuetGuard/data"
WS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(WS, "results", "tvc")
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def reproduce_paths(n_per=60, seed=42):
    """精确重现 confound_check.collect_neg 的采样顺序（不读图像内容）。

    关键：confound_check 中的 rng 消耗顺序为
      casia_au.permutation -> coco_val.permutation -> columbia_au.shuffle -> inria_noped.permutation
    APRICOT clean 部分不使用 rng，故可安全跳过。
    """
    rng = np.random.RandomState(seed)
    out = {}

    au = f"{D}/casia2/CASIA2.0_revised/Au"
    fs = sorted(f for f in os.listdir(au) if f.lower().endswith(IMG_EXT))
    out["casia_au"] = [os.path.join(au, fs[i]) for i in rng.permutation(len(fs))[:n_per]]

    coco = f"{D}/coco/val2017"
    fs = sorted(f for f in os.listdir(coco) if f.lower().endswith(IMG_EXT))
    out["coco_val"] = [os.path.join(coco, fs[i]) for i in rng.permutation(len(fs))[:n_per]]

    col = f"{D}/columbia/ImSpliceDataset"
    items = []
    for sub in sorted(os.listdir(col)):
        d2 = os.path.join(col, sub)
            # ⚠️ 修复（2026-10-16）：只取 Au-*（authentic）。旧逻辑 startswith("Au-S")
            # 只排除 Au-S，却把 Sp-*（拼接篡改图）当成干净负样本（实测 56% 污染）。
        if not os.path.isdir(d2) or not sub.startswith("Au"):
            continue
        for fn in sorted(os.listdir(d2)):
            if fn.lower().endswith(IMG_EXT):
                items.append(os.path.join(d2, fn))
    rng.shuffle(items)
    out["columbia_au"] = items[:n_per]

    inria = f"{D}/inria/data_ped/no_pedestrians"
    fs = sorted(f for f in os.listdir(inria) if f.lower().endswith(IMG_EXT))
    out["inria_noped"] = [os.path.join(inria, fs[i]) for i in rng.permutation(len(fs))[:n_per]]
    return out


def img_stats(path):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    w, h = im.size
    im2 = im.copy()
    im2.thumbnail((1024, 1024))          # 与评测一致
    a = np.asarray(im2, dtype=np.float32)
    gray = a.mean(2)
    # Laplacian 方差（清晰度/纹理）
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    g = gray
    lap = (k[0, 1] * np.roll(g, 1, 0) + k[1, 0] * np.roll(g, 1, 1) +
           k[1, 2] * np.roll(g, -1, 1) + k[2, 1] * np.roll(g, -1, 0) + k[1, 1] * g)
    lap_var = float(lap.var())
    # 灰度熵
    hist, _ = np.histogram(gray, bins=64, range=(0, 255), density=True)
    hist = hist[hist > 0]
    ent = float(-(hist * np.log2(hist)).sum())
    # 饱和度
    mx, mn = a.max(2), a.min(2)
    sat = float(((mx - mn) / (mx + 1e-6)).mean())
    return {"w": w, "h": h, "mp": round(w * h / 1e6, 3),
            "bright": round(float(gray.mean()), 2), "contrast": round(float(gray.std()), 2),
            "lapvar": round(lap_var, 1), "entropy": round(ent, 3), "sat": round(sat, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-json", default=os.path.join(OUT, "confound_main_withscores.json"))
    ap.add_argument("--out", default=os.path.join(OUT, "mech_tail_attribution.json"))
    ap.add_argument("--n-per-source", type=int, default=60)
    args = ap.parse_args()

    sc = json.load(open(args.scores_json))
    neg_scores = sc["neg_scores_by_source"]
    paths = reproduce_paths(args.n_per_source)

    rows = []
    for src in ("casia_au", "coco_val", "inria_noped", "columbia_au"):
        if src not in neg_scores or src not in paths:
            continue
        ss = neg_scores[src]
        ps = paths[src]
        if len(ss) != len(ps):
            print(f"  ⚠ {src}: 分数数({len(ss)}) != 路径数({len(ps)})，按 min 截断")
        for p, s in list(zip(ps, ss))[:min(len(ss), len(ps))]:
            try:
                st = img_stats(p)
            except Exception as e:
                continue
            st.update({"source": src, "p_yes": float(s), "path": os.path.basename(p)})
            rows.append(st)
        print(f"  {src}: 统计 {sum(1 for r in rows if r['source']==src)} 张", flush=True)

    if not rows:
        raise SystemExit("没有可用样本")

    metrics = ["mp", "bright", "contrast", "lapvar", "entropy", "sat"]
    P = np.array([r["p_yes"] for r in rows])

    def spearman(x, y):
        rx = np.argsort(np.argsort(x)).astype(float)
        ry = np.argsort(np.argsort(y)).astype(float)
        rx -= rx.mean(); ry -= ry.mean()
        den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
        return float((rx * ry).sum() / den) if den > 0 else 0.0

    print("\n=== 全样本 Spearman 相关（图像统计量 vs P(YES)）===")
    corr_all = {}
    for m in metrics:
        v = np.array([r[m] for r in rows], dtype=float)
        c = spearman(v, P)
        corr_all[m] = round(c, 4)
        print(f"  {m:10s} rho={c:+.4f}")

    # 尾部聚焦：只看 P(YES) 最高 20% 的样本，比其统计量与整体是否不同
    thr80 = float(np.percentile(P, 80))
    tail = [r for r in rows if r["p_yes"] >= thr80]
    print(f"\n=== 尾部子集（P(YES) >= p80 = {thr80:.4f}，n={len(tail)}）===")
    tail_stats = {}
    for m in metrics:
        allv = np.array([r[m] for r in rows], dtype=float)
        tailv = np.array([r[m] for r in tail], dtype=float)
        tail_stats[m] = {"all_mean": round(float(allv.mean()), 4),
                         "tail_mean": round(float(tailv.mean()), 4),
                         "ratio": round(float(tailv.mean() / max(allv.mean(), 1e-9)), 3)}
        print(f"  {m:10s} 全体均值={allv.mean():.3f} 尾部均值={tailv.mean():.3f} "
              f"比={tail_stats[m]['ratio']:.2f}")

    # 来源构成
    from collections import Counter
    src_tail = Counter(r["source"] for r in tail)
    src_all = Counter(r["source"] for r in rows)
    print(f"\n  尾部来源构成: {dict(src_tail)}  (全体: {dict(src_all)})")

    res = {"n": len(rows), "spearman_all": corr_all, "tail_threshold_p80": thr80,
           "n_tail": len(tail), "tail_stats": tail_stats,
           "tail_sources": dict(src_tail), "all_sources": dict(src_all), "rows": rows}
    json.dump(res, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"\n[mech] wrote {args.out}")


if __name__ == "__main__":
    main()
