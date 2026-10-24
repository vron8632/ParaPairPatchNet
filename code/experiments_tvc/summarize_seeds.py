#!/usr/bin/env python3
"""
统一汇总：融合方式 × 多 seed 的统计（mean / std / bootstrap CI / 同 seed 配对比较）

数据源：
  sem / tex            results/tvc/expA_{sem,tex}_s{0,1,2}.json   （同协议新增的单分支基线）
  attention / concat / gate  results/tvc/fusion_ablation_evt.json（s0）+ expA_*_s{1,2}.json
  moe                  results/tvc/mech_moe_s0.json + fab_moe_s{1,2}.json
  moe_att              results/tvc/fab_moeatt_s{0,1,2}.json
  lbreg                results/tvc/lbreg_*.json + expB_lb*_s0.json

输出：results/tvc/exp_summary_seeds.json + 终端 markdown 表
"""
import os, json, glob, itertools
import numpy as np

WS = "/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard"
TVC = f"{WS}/results/tvc"
MODES = ["sem", "tex", "concat", "gate", "attention", "moe", "moe_att"]
EXTRA = ["sem+mix", "moe_att+mix"]
MODES = MODES + EXTRA


def load_run(path):
    d = json.load(open(path))
    mode = next(iter(d))
    r = d[mode]
    ap = r["apricot"]
    return {"unseen": r["apde"]["unseen_mean"], "upc": r["apde"]["UPC"],
            "seen": r["apde"]["seen_mean"], "apde5": r["apde"]["@5pct"],
            "apr5": ap["@5pct"]["empirical"], "apr1": ap["@1pct"]["empirical"],
            "lb": r.get("lb_lambda", 0)}


def collect():
    runs = {m: {} for m in MODES}
    for m in MODES:
        for p in sorted(glob.glob(f"{TVC}/mix_{m}_s*.json")):
            s_ = int(p.split("_s")[-1].split(".")[0])
            runs[f"{m}+mix"][s_] = load_run(p)
        for p in sorted(glob.glob(f"{TVC}/expA_{m}_s*.json")):
            s = int(p.split("_s")[-1].split(".")[0])
            runs[m][s] = load_run(p)
    # 旧结果（s0 来自 fusion_ablation_evt.json）
    fab = json.load(open(f"{TVC}/fusion_ablation_evt.json"))
    for m in ("concat", "gate", "attention"):
        if 0 not in runs[m] and m in fab:
            r = fab[m]; ap = r["apricot"]
            runs[m][0] = {"unseen": r["apde"]["unseen_mean"], "upc": r["apde"]["UPC"],
                          "seen": r["apde"]["seen_mean"], "apde5": r["apde"]["@5pct"],
                          "apr5": ap["@5pct"]["empirical"], "apr1": ap["@1pct"]["empirical"], "lb": 0}
    # moe / moe_att
    for f, s in [("mech_moe_s0", 0), ("fab_moe_s1", 1), ("fab_moe_s2", 2)]:
        p = f"{TVC}/{f}.json"
        if os.path.exists(p) and s not in runs["moe"]:
            runs["moe"][s] = load_run(p)
    for s in (0, 1, 2):
        p = f"{TVC}/fab_moeatt_s{s}.json"
        if os.path.exists(p):
            runs["moe_att"][s] = load_run(p)
    return runs


def boot_ci(x, n=10000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (float(x.mean()), float(x.mean()))
    rng = np.random.default_rng(seed)
    b = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    runs = collect()
    keys = [("unseen", "跨族"), ("upc", "UPC"), ("apr5", "跨集@5%"), ("apr1", "跨集@1%")]
    out = {"runs": runs, "summary": {}}
    print(f"\n{'模式':10s} {'seeds':>7s} " + " ".join(f"{lab:>22s}" for _, lab in keys))
    for m in MODES:
        d = runs[m]
        if not d:
            continue
        seeds = sorted(d)
        cells = []
        for k, _ in keys:
            v = np.array([d[s][k] for s in seeds], float)
            lo, hi = boot_ci(v)
            cells.append(f"{v.mean():.3f}±{v.std(ddof=1) if len(v)>1 else 0:.3f}")
        print(f"{m:10s} {str(seeds):>7s} " + " ".join(f"{c:>22s}" for c in cells))
        out["summary"][m] = {k: {"mean": round(float(np.mean([d[s][k] for s in seeds])), 4),
                                 "std": round(float(np.std([d[s][k] for s in seeds], ddof=1)) if len(seeds) > 1 else 0.0, 4),
                                 "n": len(seeds),
                                 "ci95": [round(lo, 4), round(hi, 4)] if (lo := boot_ci([d[s][k] for s in seeds])[0]) is not None else None}
                             for k, _ in keys}

    # 同 seed 配对比较（相对 moe_att 与相对 sem）
    print("\n—— 同 seed 配对差值（只用两种模式都有的 seed）——")
    pairs = {}
    for base in ("moe_att", "sem", "moe_att+mix"):
        for m in MODES:
            if m == base or not runs.get(m) or not runs.get(base):
                continue
            common = sorted(set(runs[m]) & set(runs[base]))
            if len(common) < 2:
                continue
            row = {}
            for k, lab in keys:
                diff = np.array([runs[m][s][k] - runs[base][s][k] for s in common], float)
                row[k] = {"diff_mean": round(float(diff.mean()), 4),
                          "seeds": common, "diffs": [round(float(x), 4) for x in diff],
                          "all_same_sign": bool(np.all(diff > 0) or np.all(diff < 0))}
                print(f"  {m:9s} - {base:8s} {lab:8s} Δ={diff.mean():+.3f}  "
                      f"逐 seed {[round(float(x),3) for x in diff]} {'(符号一致)' if row[k]['all_same_sign'] else ''}")
            pairs[f"{m}-{base}"] = row
    out["paired"] = pairs
    json.dump(out, open(f"{TVC}/exp_summary_seeds.json", "w"), indent=1, ensure_ascii=False)
    print(f"\n→ {TVC}/exp_summary_seeds.json")


if __name__ == "__main__":
    main()
