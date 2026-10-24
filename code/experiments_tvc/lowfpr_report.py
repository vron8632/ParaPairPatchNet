#!/usr/bin/env python3
"""
统一低误报报告：读 results/tvc/sota_scores_*.npz（我们自己的模型 + 各 SOTA），
在同一 21k 负样本评测集上输出同一口径的指标。

指标：
  pAUC@1% / @5%   FPR ≤ 1% / 5% 区间的部分 AUC（阈值无关；随机基线 = 0.005 / 0.025）
  TPR@FPR        K 折交叉拟合（其余折定阈值、本折测实测 FPR），报 mean±std
  分源 FPR@1%    检查负样本源异质性
  AUC            全区间 AUC（仅作参考，低误报区间不敏感）

用法：
  python lowfpr_report.py
输出：results/tvc/lowfpr_report.json + 终端 markdown 表
"""
import os, json, glob
import numpy as np

WS = "/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard"
TVC = f"{WS}/results/tvc"
FOLDS = 10
TARGETS = (0.05, 0.01, 0.005, 0.001)


def pauc(fpr, tpr):
    return float(np.trapz(tpr, fpr) / fpr[-1]) if fpr[-1] > 0 else 0.0


def roc(sp, sn, npts=6000):
    thr = np.unique(np.concatenate([sn, sp]))
    thr = np.sort(thr)[::max(1, len(thr) // npts)]
    tpr = np.array([(sp >= t).mean() for t in thr])
    fpr = np.array([(sn >= t).mean() for t in thr])
    o = np.argsort(fpr)
    return fpr[o], tpr[o]


def crossfit(sp, sn, folds=FOLDS, targets=TARGETS):
    rng = np.random.default_rng(0)
    parts = np.array_split(rng.permutation(len(sn)), folds)
    out = {}
    for q in targets:
        tprs, fprs = [], []
        for k in range(folds):
            te = parts[k]; tr = np.concatenate([parts[j] for j in range(folds) if j != k])
            thr = float(np.percentile(sn[tr], 100 * (1 - q)))
            tprs.append(float((sp >= thr).mean()))
            fprs.append(float((sn[te] >= thr).mean()))
        out[f"@{q*100:g}pct"] = {"tpr": round(float(np.mean(tprs)), 4),
                                 "tpr_std": round(float(np.std(tprs, ddof=1)), 4),
                                 "fpr_obs": round(float(np.mean(fprs)), 4)}
    return out


def main():
    files = sorted(glob.glob(f"{TVC}/sota_scores_*.npz"))
    if not files:
        print("未找到 sota_scores_*.npz"); return
    rep = {}
    for f in files:
        name = os.path.basename(f)[len("sota_scores_"):-4]
        z = np.load(f)
        s, y, src = z["scores"].astype(float), z["label"], z["src"]
        sp, sn = s[y == 1], s[y == 0]
        # 阈值口径与 crossfit 一致：用全部负样本 p99 作参考阈值
        thr = {q: float(np.percentile(sn, 100 * (1 - q))) for q in TARGETS}
        fpr, tpr = roc(sp, sn)
        m1 = fpr <= 0.01; m5 = fpr <= 0.05
        e = {"n_pos": int(len(sp)), "n_neg": int(len(sn)),
             "pAUC@1pct": round(pauc(fpr[m1], tpr[m1]), 4) if m1.sum() > 2 else None,
             "pAUC@5pct": round(pauc(fpr[m5], tpr[m5]), 4) if m5.sum() > 2 else None,
             "crossfit": crossfit(sp, sn),
             "pos_mean": round(float(sp.mean()), 4), "neg_mean": round(float(sn.mean()), 4),
             "fpr_by_source@1pct": {k: round(float((sn[src[y == 0] == k] >= thr[0.01]).mean()), 4)
                                    for k in sorted(set(src[y == 0]))}}
        try:
            from sklearn.metrics import roc_auc_score
            e["auc"] = round(float(roc_auc_score(y, s)), 4)
        except Exception:
            e["auc"] = None
        rep[name] = e

    # 打印表（按 pAUC@1% 排序）
    order = sorted(rep, key=lambda k: -(rep[k]["pAUC@1pct"] or -1))
    print(f"\n{'方法 / ckpt':34s} {'pAUC@1%':>8s} {'pAUC@5%':>8s} {'TPR@1%':>7s} {'TPR@5%':>7s} {'TPR@0.1%':>8s} {'AUC':>6s}")
    print("-" * 92)
    for k in order:
        e = rep[k]; cf = e["crossfit"]
        print(f"{k:34s} {e['pAUC@1pct']!s:>8s} {e['pAUC@5pct']!s:>8s} "
              f"{cf['@1pct']['tpr']!s:>7s} {cf['@5pct']['tpr']!s:>7s} {cf['@0.1pct']['tpr']!s:>8s} {e['auc']!s:>6s}")
    print("\n分源 FPR@1%（阈值=全负样本 p99）：")
    srcs = sorted({s for e in rep.values() for s in e["fpr_by_source@1pct"]})
    print(f"{'方法 / ckpt':34s} " + " ".join(f"{s:>12s}" for s in srcs))
    for k in order:
        print(f"{k:34s} " + " ".join(f"{rep[k]['fpr_by_source@1pct'].get(s, float('nan'))!s:>12s}" for s in srcs))
    json.dump(rep, open(f"{TVC}/lowfpr_report.json", "w"), indent=1, ensure_ascii=False)
    print(f"\n→ {TVC}/lowfpr_report.json")


if __name__ == "__main__":
    main()
