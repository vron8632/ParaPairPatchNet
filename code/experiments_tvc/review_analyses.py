#!/usr/bin/env python3
"""
扩展分析（基于已有分数文件）：
  A. 主表阈值三策略（empirical / order-statistic / conservative-binomial）
  B. Bootstrap 95% CI（TPR@FPR 点估计）
  C. APDE 按攻击族拆表 + Wilson CI
输出: results/tvc/review_analyses.json
"""
import json, math, os, re
import numpy as np

TVC = "/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard/results/tvc"
rng = np.random.RandomState(0)


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (max(0.0, c-h), min(1.0, c+h))


def boot_tpr(pos, neg, alpha, n_boot=2000):
    """对阈值与正样本同时重采样，得 TPR@FPR 的 Bootstrap 95% CI"""
    vals = []
    nb, np_ = len(neg), len(pos)
    for _ in range(n_boot):
        thr = np.percentile(neg[rng.randint(0, nb, nb)], 100*(1-alpha))
        vals.append((pos[rng.randint(0, np_, np_)] >= thr).mean())
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


out = {}

# ── B+C 主模型（confound 分数，APRICOT 100 正 + 240 负）──
d = json.load(open(f"{TVC}/confound_main_withscores.json"))
pos = np.array(d["pos_scores"]); neg = np.concatenate([np.array(v) for v in d["neg_scores_by_source"].values()])
print("=== B. 主表 Bootstrap 95% CI（100 正 + 240 负）===")
out["bootstrap_ci"] = {}
for a in (0.05, 0.10):
    thr = np.percentile(neg, 100*(1-a)); k = int((pos >= thr).sum())
    lo_h, hi_h = wilson(k, len(pos)); lo_b, hi_b = boot_tpr(pos, neg, a)
    out["bootstrap_ci"][f"@{int(a*100)}pct"] = {
        "tpr": round(k/len(pos), 4), "wilson": [round(lo_h,4), round(hi_h,4)],
        "bootstrap": [round(lo_b,4), round(hi_b,4)]}
    print(f"  @{int(a*100)}%: TPR={k/len(pos):.3f}  Wilson[{lo_h:.3f},{hi_h:.3f}]  Bootstrap[{lo_b:.3f},{hi_b:.3f}]")

# ── A. 阈值三策略（主表协议：100 正 240 负）──
print("\n=== A. 阈值三策略 ===")
def conformal_thr(cal, alpha):
    n = len(cal); k = min(n, math.ceil((n+1)*(1-alpha)))
    return float(np.sort(cal)[k-1])
def binom_thr(cal, alpha, delta=0.05):
    from scipy import stats
    s = np.sort(cal); n = len(cal)
    for k in range(n, 0, -1):
        m = n - k + 1
        upper = stats.beta.ppf(1-delta, m+1, n-m) if m < n else 1.0
        if upper <= alpha: return float(s[k-1])
    return float(s[-1])
out["threshold_strategies"] = {}
for a in (0.05, 0.10):
    row = {}
    for name, fn in (("empirical", lambda c, al: float(np.percentile(c, 100*(1-al)))),
                     ("order_statistic", conformal_thr), ("conservative_binomial", binom_thr)):
        thr = fn(neg, a); row[name] = {"threshold": round(thr,4), "tpr": round(float((pos >= thr).mean()),4)}
    out["threshold_strategies"][f"@{int(a*100)}pct"] = row
    print(f"  @{int(a*100)}%: " + "  ".join(f"{k}={v['tpr']:.3f}" for k, v in row.items()))

# ── C. APDE 按攻击族拆表 ──
print("\n=== C. APDE 攻击族拆分 ===")
ap = json.load(open(f"{TVC}/apde_eval.json"))
pos_all = ap["pos_scores"]; neg_a = np.array(ap["neg_scores"])
img_dir = f"{TVC}/apde_imgs/pos"
files = sorted(os.listdir(img_dir))
fams = {}
for i, f in enumerate(files):
    fam = f.split("_")[0]
    fams.setdefault(fam, []).append(pos_all[i])
out["apde_by_attack"] = {}
print(f"  {'攻击族':16s} {'n':>4} {'@5% FPR':>9} {'Wilson':>18}")
for fam, sc in sorted(fams.items()):
    s = np.array(sc); thr = np.percentile(neg_a, 95); k = int((s >= thr).sum())
    lo, hi = wilson(k, len(s))
    out["apde_by_attack"][fam] = {"n": len(s), "tpr5": round(k/len(s),4), "wilson": [round(lo,4), round(hi,4)]}
    print(f"  {fam:16s} {len(s):>4} {k/len(s):>9.3f}  [{lo:.3f}, {hi:.3f}]")
# APDE 全表 Wilson CI
k = int((np.array(pos_all) >= np.percentile(neg_a, 95)).sum()); lo, hi = wilson(k, len(pos_all))
out["apde_overall_wilson"] = {"tpr5": round(k/len(pos_all),4), "ci": [round(lo,4), round(hi,4)]}
print(f"  APDE 总体 @5%: {k/len(pos_all):.3f} Wilson[{lo:.3f},{hi:.3f}]")

json.dump(out, open(f"{TVC}/review_analyses.json", "w"), indent=1, ensure_ascii=False)
print(f"\n→ {TVC}/review_analyses.json")
