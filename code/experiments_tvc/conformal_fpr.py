#!/usr/bin/env python3
"""
FPR 保证升级：经验分位数 → 有限样本保证（conformal/-order statistic 风格）
==================================================================
回答："把 FPR 保证从经验分位数升级为分布无关保证"是否可行并有效。

做法（无分布假设）：
  1. 按分位数修正阈值：经验 (1-α) 分位数对应的 order statistic 位置修正为
     k = ceil((n+1)(1-α))，这是标准的 conformal upper quantile（保证 coverage）。
  2. 实测覆盖率（coverage）：多次重采样——每次从负样本池抽 n_cal 做校准定阈值，
     在 held-out 负样本上测实际 FPR，看 P(FPR ≤ α) 是否 ≥ 1-δ。
数据：confound_main_withscores.json 的逐样本分数（多源真实负样本）。
输出: results/tvc/conformal_fpr.json（含覆盖率为实测值）
"""
import json, math, os
import numpy as np

TVC = "/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard/results/tvc"


def conformal_threshold(neg_cal, alpha):
    """order-statistic conformal: 取第 ceil((n+1)(1-α)) 小的分数（cap 到 n）"""
    n = len(neg_cal)
    k = min(n, math.ceil((n + 1) * (1 - alpha)))
    return float(np.sort(neg_cal)[k - 1])


def main():
    d = json.load(open(f"{TVC}/confound_main_withscores.json"))
    neg = np.concatenate([np.array(v) for v in d["neg_scores_by_source"].values()])
    pos = np.array(d["pos_scores"])
    print(f"pool: n_neg={len(neg)} n_pos={len(pos)}")

    res = {"n_pool": int(len(neg)), "alphas": {}, "coverage_test": {}}
    for alpha in (0.05, 0.10, 0.20):
        # 修正阈值 vs 普通经验分位数（全池）
        thr_c = conformal_threshold(neg, alpha)
        thr_e = float(np.percentile(neg, 100 * (1 - alpha)))
        fpr_c = float((neg >= thr_c).mean())
        tpr_c = float((pos >= thr_c).mean())
        res["alphas"][f"{alpha}"] = {
            "thr_conformal": round(thr_c, 6), "thr_empirical": round(thr_e, 6),
            "FPR_pooled": round(fpr_c, 4), "TPR_pooled": round(tpr_c, 4)}
        print(f"α={alpha}: conformal τ={thr_c:.4f} vs empirical τ={thr_e:.4f} | "
              f"实测FPR={fpr_c:.4f} TPR={tpr_c:.4f}")

        # coverage 实测：重复抽样校准/测试分割
        rng = np.random.RandomState(0)
        trials, ok = 60, 0
        n_cal = 120
        for t in range(trials):
            perm = rng.permutation(len(neg))
            cal, test = neg[perm[:n_cal]], neg[perm[n_cal:]]
            thr = conformal_threshold(cal, alpha)
            fpr_test = float((test >= thr).mean())
            if fpr_test <= alpha + 1e-12:
                ok += 1
        cov = ok / trials
        res["coverage_test"][f"{alpha}"] = {"coverage": round(cov, 4),
                                             "n_cal": n_cal, "trials": trials,
                                             "target_1_minus_delta": 0.95}
        print(f"  coverage实测: {cov:.3f}  (目标 ≥0.95, 校准集 n={n_cal})")

    json.dump(res, open(f"{TVC}/conformal_fpr.json", "w"), indent=1, ensure_ascii=False)
    print(f"→ {TVC}/conformal_fpr.json")


if __name__ == "__main__":
    main()
