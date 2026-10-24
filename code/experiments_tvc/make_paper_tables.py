#!/usr/bin/env python3
"""
论文 LaTeX 表格片段生成器
=======================================
从 results/tvc/*.json 生成可直接粘贴进 main.tex 的 tabularx 片段，
输出 docs/paper_tables_扩展分析.tex（不修改论文本体）。
用法: $PY make_paper_tables.py
"""
import os, io, json
import numpy as np

WS = "/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard"
TVC = f"{WS}/results/tvc"
OUT = f"{WS}/docs/paper_tables_扩展分析.tex"


def L(name):
    p = f"{TVC}/{name}"
    return json.load(open(p)) if os.path.exists(p) else None


blocks = []
w = blocks.append

# ══ T1 主表（双口径 × 多 seed）══
d = L("main_table_v2.json")
w("% ==== T1: main results with both negative-pool scopes ====")
if d and any(isinstance(v, dict) and "tpr" in v for v in d.values()):
    w(r"""\begin{table*}[t]
\centering\footnotesize
\caption{Main results under the unified protocol, reported under two negative-pool
scopes: three sources (excluding Columbia) and the full four sources.
Values are TPR@FPR; the last column reports the cross-seed mean$\pm$std (three training seeds).}
\label{tab:main_v2}
\begin{tabularx}{\columnwidth}{lXcccc}
\toprule
Method & Scope & @1\% & @5\% & @10\% & @20\% \\
\midrule""")
    order = ["lora_patch_detector", "full_lora_s1", "full_lora_s2"]
    for k in order:
        if k not in d or not isinstance(d[k], dict) or "tpr" not in d[k]: continue
        v = d[k]
        for tag, name in (("3src", "3 src"), ("4src", "4 src")):
            row = v["tpr"][tag]
            cells = " & ".join(f"{row[f'@{a}pct']['tpr']:.2f}" for a in (1, 5, 10, 20))
            w(f"{k} & {name} & {cells} \\\\")
        w("\\midrule")
    s = d.get("summary_mean_std", {})
    if s:
        row = s.get("4src@5pct") or s.get("3src@5pct")
        if row:
            w(f"\\multicolumn{{6}}{{l}}{{\\emph{{cross-seed @5\\%: {row['mean']:.3f}$\\pm${row['std']:.3f} (all: {row['all']})}}}} \\\\")
    w("\\bottomrule\n\\end{tabularx}\n\\end{table*}")
else:
    w("% (main_table_v2.json 尚未生成)")

# ══ T2 阈值三策略 ══
d = L("review_analyses.json")
w("\n% ==== T2: threshold strategies ====")
if d and "threshold_strategies" in d:
    w(r"""\begin{table}[!htbp]
\centering\footnotesize
\caption{Threshold strategies and the resulting detection rate at the same nominal budgets.
The empirical quantile is the common practice; conservative corrections trade detection for coverage.}
\label{tab:threshold_v2}
\begin{tabularx}{\columnwidth}{lXcc}
\toprule
Strategy & & @5\% FPR & @10\% FPR \\
\midrule""")
    ts = d["threshold_strategies"]
    for strat in ("empirical", "order_statistic", "conservative_binomial"):
        t5 = ts.get("@5pct", {}).get(strat, {}).get("tpr")
        t10 = ts.get("@10pct", {}).get(strat, {}).get("tpr")
        if t5 is None: continue
        w(f"{strat.replace('_', ' ')} & & {t5:.3f} & {t10:.3f} \\\\")
    w("\\bottomrule\n\\end{tabularx}\n\\end{table}")
    # APDE 攻击族
    ap = d.get("apde_by_attack")
    if ap:
        w(r"""
% ==== T3: APDE by attack family ====
\begin{table}[!htbp]
\centering\footnotesize
\caption{APDE second benchmark broken down by attack family (TPR@5\% FPR, $n$ per family).
The weakest family (UPC) is reported faithfully.}
\label{tab:apde_family}
\begin{tabularx}{\columnwidth}{lXcc}
\toprule
Attack family & $n$ & TPR@5\% & Wilson 95\% CI \\
\midrule""")
        for fam, v in sorted(ap.items(), key=lambda x: -x[1]["tpr5"]):
            w(f"{fam} & {v['n']} & {v['tpr5']:.3f} & [{v['wilson'][0]:.3f}, {v['wilson'][1]:.3f}] \\\\")
        ov = d.get("apde_overall_wilson", {})
        if ov:
            w(f"\\midrule\n\\textbf{{overall}} & {sum(v['n'] for v in ap.values())} & "
              f"\\textbf{{{ov['tpr5']:.3f}}} & [{ov['ci'][0]:.3f}, {ov['ci'][1]:.3f}] \\\\")
        w("\\bottomrule\n\\end{tabularx}\n\\end{table}")
else:
    w("% (review_analyses.json 尚未生成)")

# ══ T4 超参敏感性 ══
w("\n% ==== T4: hyperparameter sensitivity ====")
hp = [("baseline ($\\beta$0.15/$\\lambda$1.0/$\\tau_0$0.5)", "abl_both_soft")]
for tag, lab in (("hp_beta0.05", "$\\beta$=0.05"), ("hp_beta0.30", "$\\beta$=0.30"),
                 ("hp_lam0.5", "$\\lambda$=0.5"), ("hp_lam2.0", "$\\lambda$=2.0"),
                 ("hp_tau0.3", "$\\tau_0$=0.3"), ("hp_tau0.7", "$\\tau_0$=0.7")):
    if os.path.exists(f"{TVC}/lora_mllm_eval_{tag}.json"): hp.append((lab, tag))
if len(hp) > 1:
    w(r"""\begin{table}[!htbp]
\centering\footnotesize
\caption{Hyperparameter sensitivity of the constraint ($\beta$, $\lambda$, $\tau_0$);
all other settings fixed.}
\label{tab:hyperparam}
\begin{tabularx}{\columnwidth}{lXcc}
\toprule
Setting & & AUC & @5\% FPR \\
\midrule""")
    for lab, tag in hp:
        r = L(f"lora_mllm_eval_{tag}.json")
        if not r: continue
        w(f"{lab} & & {r['auc']:.4f} & "
          f"{(r.get('best_tpr_at_fpr<=5pct') or {}).get('tpr', float('nan')):.3f} \\\\")
    w("\\bottomrule\n\\end{tabularx}\n\\end{table}")
else:
    w("% (超参结果尚未生成)")

# ══ T5 数据规模 + 负样本增强 ══
w("\n% ==== T5: training data scale & negative augmentation ====")
sc = []
for tag, lab in (("scale_025", "25\\%"), ("scale_05", "50\\%"), ("abl_both_soft", "100\\% (1.5k)"),
                 ("scale_3k", "3.3k (A+B+C)"), ("negaug", "low-sat augment")):
    r = L(f"lora_mllm_eval_{tag}.json")
    if r: sc.append((lab, r))
if len(sc) > 1:
    w(r"""\begin{table}[!htbp]
\centering\footnotesize
\caption{Training-set scale and negative augmentation ablations.}
\label{tab:datascale}
\begin{tabularx}{\columnwidth}{lXcc}
\toprule
Training data & & AUC & @5\% FPR \\
\midrule""")
    for lab, r in sc:
        w(f"{lab} & & {r['auc']:.4f} & "
          f"{(r.get('best_tpr_at_fpr<=5pct') or {}).get('tpr', float('nan')):.3f} \\\\")
    w("\\bottomrule\n\\end{tabularx}\n\\end{table}")
else:
    w("% (规模/增强结果尚未生成)")

# ══ T6 鲁棒性 ══
w("\n% ==== T6: robustness (fixed vs adaptive threshold) ====")
d = L("adv_attack_v2.json")
if d:
    w(r"""\begin{table}[!htbp]
\centering\footnotesize
\caption{White-box PGD under fixed- and adaptive-threshold attacker models
(three seeds; ASR = drop in TPR at the fixed deployment threshold).}
\label{tab:adv_v2}
\begin{tabularx}{\columnwidth}{lXcc}
\toprule
Attacker / budget & & TPR clean$\to$attacked & ASR \\
\midrule""")
    ck = [k for k in d if k != "summary"][:1]
    if ck:
        for cfg, row in d[ck[0]].items():
            w(f"{cfg.replace('_', ' ')} & & {row['tpr_clean']:.3f}$\\to${row['tpr_attacked']:.3f} & {row['asr']:.3f} \\\\")
    if "summary" in d:
        w("\\midrule")
        for k, v in d["summary"].items():
            w(f"\\emph{{{k.replace('_', ' ')}}} & & & {v['asr_mean']:.3f}$\\pm${v['asr_std']:.3f} \\\\")
    w("\\bottomrule\n\\end{tabularx}\n\\end{table}")
else:
    w("% (adv_attack_v2.json 尚未生成)")

io.open(OUT, "w", encoding="utf-8").write("\n".join(blocks) + "\n")
print(f"→ {OUT}（{len(blocks)} 个片段）")
