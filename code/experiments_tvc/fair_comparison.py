#!/usr/bin/env python3
"""
三方公平对比 — SOTA 基线 vs 本方法（同一评估集）
==================================================
评审必需：所有方法必须在**完全相同**的评估集上比较。
评估集（统一）:
  正: APRICOT 100 张真实打印补丁
  负: casia_au 60 + coco_val 60 + inria_noped 60（均由 RandomState(42) 选取，三批同一批图）

方法:
  - PSCC-Net (ICCV 2021)  -> pscc_baseline.json
  - TruFor   (CVPR 2023)  -> trufor_baseline.json
  - RL-LoRA  (ours)       -> confound_check.json（含 inria_noped）

用法:
  $PY code/experiments_tvc/fair_comparison.py
输出: results/tvc/fair_comparison.json + 终端表
"""
import json, os, sys
import numpy as np

WS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(WS, "results", "tvc")
COMMON_NEG = ("casia_au", "coco_val", "inria_noped")


def load(name):
    p = os.path.join(OUT, name)
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    ps = load("pscc_baseline.json")
    tf = load("trufor_baseline.json")
    cf = load("confound_after_rl.json")  # 旧 ckpt（lora_rl_ckpt），保留作对照
    main_scores = load("confound_main_withscores.json")  # 主模型 lora_patch_detector（逐样本分数）

    rows, detail = [], {}

    # ---- ours: 主模型，与基线同一 3 源负样本池按目标 FPR 定阈值 ----
    if main_scores and "pos_scores" in main_scores:
        import numpy as _np
        pos = _np.asarray(main_scores["pos_scores"], dtype=float)
        neg3 = {k: _np.asarray(v, dtype=float) for k, v in
                main_scores["neg_scores_by_source"].items() if k in COMMON_NEG and len(v)}
        if len(pos) and len(neg3) == len(COMMON_NEG):
            allneg = _np.concatenate([neg3[k] for k in COMMON_NEG])
            ours = {"name": "NP-PatchLM (ours)",
                    "ckpt": "lora_patch_detector",
                    "pos_mean": round(float(pos.mean()), 4),
                    "neg_mean": {k: round(float(neg3[k].mean()), 4) for k in COMMON_NEG},
                    "tpr": {}}
            for a, tag in ((0.01, "@FPR1pct"), (0.05, "@FPR5pct"), (0.10, "@FPR10pct"),
                           (0.20, "@FPR20pct"), (0.33, "@FPR33pct")):
                thr = _np.percentile(allneg, 100 * (1 - a))
                ours["tpr"][tag] = round(float((pos >= thr).mean()), 4)
            rows.append(ours)

    # ---- 旧 ckpt 对照行（若有）----
    if cf and all(k in cf.get("neg_per_source", {}) for k in COMMON_NEG):
        legacy = {"name": "RL-LoRA (legacy ckpt)",
                  "ckpt": "lora_rl_ckpt",
                  "pos_mean": cf["pos_stats"]["mean"],
                  "neg_mean": {k: cf["neg_per_source"][k]["mean"] for k in COMMON_NEG},
                  "tpr": {}}
        # 严格按 key 取值，不做模糊匹配（否则会把 @33% 误标为 @20%）
        for tag, exact in (("@FPR1pct", "@target_FPR1pct"),
                           ("@FPR5pct", "@target_FPR5pct"),
                           ("@FPR10pct", "@target_FPR10pct"),
                           ("@FPR20pct", "@target_FPR20pct"),
                           ("@FPR33pct", "@target_FPR33pct")):
            if exact in cf:
                legacy["tpr"][tag] = cf[exact]["tpr"]
        rows.append(legacy)

    for d, name in ((ps, "PSCC-Net (ICCV 2021)"), (tf, "TruFor (CVPR 2023)")):
        if not d:
            continue
        rows.append({"name": name, "pos_mean": d["pos_mean"],
                     "neg_mean": {k: d["neg_per_source"][k]["mean"]
                                  for k in COMMON_NEG if k in d["neg_per_source"]},
                     "tpr": {tag: d[tag]["tpr"] for tag in ("@FPR5pct", "@FPR10pct", "@FPR20pct")
                             if tag in d}})

    # 打印
    print("=" * 84)
    print("三方公平对比（同一评估集：APRICOT 100 正 + casia/coco/inria 各 60 负）")
    print("=" * 84)
    print(f"{'方法':26s} {'pos_mean':>9} {'neg(casia)':>11} {'neg(coco)':>10} "
          f"{'neg(inria)':>11} {'pos/neg比':>9}")
    print("-" * 84)
    for r in rows:
        nm = r["neg_mean"]
        nmean = np.mean(list(nm.values())) if nm else float("nan")
        ratio = r["pos_mean"] / max(nmean, 1e-6)
        print(f"{r['name']:26s} {r['pos_mean']:>9.4f} {nm.get('casia_au', float('nan')):>11.4f} "
              f"{nm.get('coco_val', float('nan')):>10.4f} {nm.get('inria_noped', float('nan')):>11.4f} "
              f"{ratio:>9.2f}")
    print("-" * 84)
    print(f"{'方法':26s} {'TPR@5%':>9} {'TPR@10%':>9} {'TPR@20%':>9} {'TPR@33%':>9}")
    print("-" * 84)
    for r in rows:
        t = r["tpr"]
        print(f"{r['name']:26s} {t.get('@FPR5pct', '—'):>9} {t.get('@FPR10pct', '—'):>9} "
              f"{t.get('@FPR20pct', '—'):>9} {t.get('@FPR33pct', '—'):>9}")
    print("=" * 84)
    print("注: 'pos/neg比' 越大说明正负分数分离度越好（TruFor≈1 即正负不分）。")

    out = os.path.join(OUT, "fair_comparison.json")
    json.dump({"eval_set": {"pos": "APRICOT 100", "neg": list(COMMON_NEG) + ["(columbia for ours)"]},
               "methods": rows}, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"[fair] wrote {out}")


if __name__ == "__main__":
    main()
