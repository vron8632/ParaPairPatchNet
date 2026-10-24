#!/usr/bin/env python3
"""
创新 #6 — 级联架构评估：轻量取证分支粗筛 + MLLM 复核
======================================================
回应"MLLM 推理慢（~2s/图）、无法部署"这一必然质疑。

设计（两级级联）:
  Stage-1  轻量取证分支（SPN+DCT+CLIP MLP，27FPS，AUC 0.783）
           输出分数 s_v；高置信区间（s_v ≤ τ_lo 或 ≥ τ_hi）直接决策；
  Stage-2  仅对**中置信区间**的样本调用 MLLM（LoRA-Qwen2-VL，AUC 0.978）
           输出 s_m，用 s_m 决策。

两个分支的分数尺度不同，先用**负类经验 CDF** 把各自分数映射到 [0,1] 的
"FPR 尺度"（等价于 ROC 空间中的统一校准），再级联。

指标: 级联后 TPR@FPR、**MLLM 调用率**（成本）、估计吞吐。

用法:
  $PY code/experiments_tvc/cascade_eval.py
输出: results/tvc/cascade_eval.json
"""
import json, os
import numpy as np

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS = os.path.dirname(CODE_DIR)
OUT = os.path.join(WS, "results", "tvc")
SCORES = os.path.join(OUT, "adapter_fusion_scores.json")

# 吞吐假设（RTX 4060 Ti 实测/论文口径）
THR_V = 1 / 27.0        # 轻量分支: >27 FPS
THR_M = 2.0             # MLLM: ~2 s/图


def to_fpr_scale(s, y_neg):
    """把分数映射为该分支 ROC 空间的统一尺度：s' = 负类经验 CDF(s)。
    s' 即"若以此值为阈值，该分支的 FPR"——两分支因此可比、可级联。"""
    order = np.argsort(s[y_neg])
    ranks = np.empty(len(s))
    ranks[y_neg] = np.arange(1, y_neg.sum() + 1)      # 1..n_neg
    # 其余（正类）用插值放到同一 CDF 上
    sorted_neg = np.sort(s[y_neg])
    pos = ~y_neg
    ranks[pos] = np.searchsorted(sorted_neg, s[pos], side="right") + 0.5
    return ranks / max(y_neg.sum(), 1)


def main():
    d = json.load(open(SCORES))
    m = {(r["domain"], r["idx"]): r for r in d["mllm"]}
    v = {(r["domain"], r["idx"]): r for r in d["visual"]}
    keys = sorted(set(m) & set(v))
    y = np.array([m[k]["label"] for k in keys])
    s_m_raw = np.array([m[k]["score"] for k in keys])
    s_v_raw = np.array([v[k]["score"] for k in keys])
    neg = y == 0

    # 统一校准到 FPR 尺度
    s_m = to_fpr_scale(s_m_raw, neg)
    s_v = to_fpr_scale(s_v_raw, neg)

    # 纯分支基线
    def rep(p, tag, mlm_rate):
        pos = y == 1
        order = np.argsort(np.concatenate([p[pos], p[neg]]))
        rk = np.empty(len(order)); rk[order] = np.arange(1, len(order) + 1)
        auc = (rk[:pos.sum()].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum())
        th = np.unique(np.concatenate([
            np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5]),
            np.percentile(p[neg], np.linspace(0, 100, 201))]))
        sweep = [{"tau": round(float(t), 4),
                  "tpr": round(float(((p >= t) & pos).sum() / pos.sum()), 4),
                  "fpr": round(float(((p >= t) & neg).sum() / neg.sum()), 4)} for t in th]
        out = {"tag": tag, "mlm_call_rate": round(float(mlm_rate), 4),
               "auc": round(float(auc), 4), "sweep": sweep}
        for b in (0.05, 0.10, 0.20):
            c = [s for s in sweep if s["fpr"] <= b]
            out[f"tpr@{int(b*100)}"] = max(c, key=lambda s: s["tpr"])["tpr"] if c else None
        print(f"[{tag}] AUC={out['auc']} 调用率={out['mlm_call_rate']} "
              f"TPR@5%={out['tpr@5']} @10%={out['tpr@10']} @20%={out['tpr@20']}", flush=True)
        return out

    rows = {"mllm_only": rep(s_m, "MLLM only", 1.0),
            "visual_only": rep(s_v, "轻量分支 only", 0.0)}

    # 级联：margin 扫描（0.5=全用轻量，0=全用 MLLM）
    margins = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    for mg in margins:
        use_v = (s_v <= 0.5 - mg) | (s_v >= 0.5 + mg)     # 高置信 => 轻量直接决策
        p = np.where(use_v, s_v, s_m)
        rate = float((~use_v).mean())
        rows[f"cascade_mg{int(mg*100)}"] = rep(p, f"级联 margin={mg:.2f}", rate)

    out = os.path.join(OUT, "cascade_eval.json")
    json.dump(rows, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"[cascade] wrote {out}")


if __name__ == "__main__":
    main()
