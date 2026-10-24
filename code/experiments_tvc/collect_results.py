#!/usr/bin/env python3
"""
实验结果汇总器
=========================
扫描 results/tvc/*.json，把相关结果汇总为一张 markdown 表，
写入 docs/实验结果汇总_扩展分析.md。缺文件的项跳过（支持部分完成时运行）。
用法: $PY collect_results.py
"""
import os, io, json, glob
import numpy as np

WS = "/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard"
TVC = f"{WS}/results/tvc"
OUT = f"{WS}/docs/实验结果汇总_扩展分析.md"


def load(name):
    p = f"{TVC}/{name}"
    return json.load(open(p)) if os.path.exists(p) else None


def tpr_row(d, keys=("TPR@1pct", "TPR@5pct", "TPR@10pct", "TPR@20pct"), label=""):
    return " | ".join(f"{d.get(k, d.get(k.replace('TPR@',''))) }" for k in keys)


lines = ["# 扩展分析实验结果汇总（自动生成）", "",
         f"> 生成时间：{os.popen('date +%F_%T').read().strip()}　"
         f"数据源：`results/tvc/`　脚本：`code/experiments_tvc/collect_results.py`", ""]

# ── 1. 主表 v2（双口径 × 多 seed）──
d = load("main_table_v2.json")
lines += ["## 1. 主表 v2（138/100 正样本 × 三源/四源 × 多 seed）", ""]
if d:
    lines += ["| ckpt | 正样本 | 三源 @1/5/10/20% | 四源 @1/5/10/20% |", "|---|---|---|---|"]
    for k, v in d.items():
        if k == "summary_mean_std" or not isinstance(v, dict) or "tpr" not in v: continue
        t3 = v["tpr"]["3src"]; t4 = v["tpr"]["4src"]
        f = lambda t: "/".join(str(t[f"@{a}pct"]["tpr"]) for a in (1, 5, 10, 20))
        lines.append(f"| {k} | {v['n_pos']} | {f(t3)} | {f(t4)} |")
    s = d.get("summary_mean_std", {})
    if s:
        lines += ["", "**跨 seed 汇总**："]
        for k, v in s.items():
            lines.append(f"- `{k}`: {v['mean']:.4f}±{v['std']:.4f}  (各 seed: {v['all']})")
else:
    lines.append("_（尚未生成）_")

# ── 2. 零成本分析 ──
d = load("review_analyses.json")
lines += ["", "## 2. Bootstrap/Wilson CI、阈值三策略、APDE 攻击族拆表", ""]
if d:
    b = d.get("bootstrap_ci", {})
    for a, v in b.items():
        lines.append(f"- **{a}** TPR={v['tpr']}　Wilson={v['wilson']}　Bootstrap={v['bootstrap']}")
    ts = d.get("threshold_strategies", {})
    for a, row in ts.items():
        lines.append(f"- 阈值策略 {a}: " + "　".join(f"{k}={v['tpr']}" for k, v in row.items()))
    ap = d.get("apde_by_attack", {})
    if ap:
        lines += ["", "**APDE 按攻击族（@5% FPR）**：", "", "| 攻击族 | n | TPR@5% | Wilson |", "|---|---|---|---|"]
        for fam, v in sorted(ap.items(), key=lambda x: -x[1]["tpr5"]):
            lines.append(f"| {fam} | {v['n']} | {v['tpr5']} | {v['wilson']} |")
        ov = d.get("apde_overall_wilson", {})
        lines.append(f"| **总体** | {sum(v['n'] for v in ap.values())} | {ov.get('tpr5')} | {ov.get('ci')} |")
else:
    lines.append("_（尚未生成）_")

# ── 3. CLIP@512 ──
d = load("clip_linear_512.json")
lines += ["", "## 3. CLIP@512 对照（分辨率混淆控制）", ""]
if d:
    lines.append(f"- 三源: " + "　".join(f"{k}={v}" for k, v in d.get("3src", {}).items()))
    lines.append(f"- 四源: " + "　".join(f"{k}={v}" for k, v in d.get("4src", {}).items()))
    lines.append(f"- pos_mean={d.get('pos_mean')}")
else:
    lines.append("_（尚未生成）_")

# ── 4. Prompt 鲁棒性 ──
d = load("prompt_robustness.json")
lines += ["", "## 4. Prompt 措辞鲁棒性", ""]
if d:
    lines += ["| prompt | pos_mean | neg_mean | @5% | @10% |", "|---|---|---|---|---|"]
    for k, v in d.items():
        lines.append(f"| {k} | {v['pos_mean']} | {v['neg_mean']} | {v['TPR@5pct']} | {v['TPR@10pct']} |")
else:
    lines.append("_（尚未生成）_")

# ── 5. 超参敏感性 ──
lines += ["", "## 5. 超参敏感性（β / λ / τ0）", ""]
rows = []
for tag in ("hp_beta0.05", "hp_beta0.30", "hp_lam0.5", "hp_lam2.0", "hp_tau0.3", "hp_tau0.7"):
    p = f"{TVC}/lora_mllm_eval_{tag}.json"
    if os.path.exists(p):
        r = json.load(open(p))
        rows.append((tag, r["auc"], (r.get("best_tpr_at_fpr<=5pct") or {}).get("tpr"),
                     (r.get("best_tpr_at_fpr<=10pct") or {}).get("tpr")))
if rows or os.path.exists(f"{TVC}/lora_mllm_eval_abl_both_soft.json"):
    base = json.load(open(f"{TVC}/lora_mllm_eval_abl_both_soft.json"))
    lines += ["| 配置 | AUC | @5% | @10% |", "|---|---|---|---|",
              f"| baseline (β0.15/λ1.0/τ0.5) | {base['auc']:.4f} | "
              f"{(base.get('best_tpr_at_fpr<=5pct') or {}).get('tpr')} | "
              f"{(base.get('best_tpr_at_fpr<=10pct') or {}).get('tpr')} |"]
    for t, a, t5, t10 in rows:
        lines.append(f"| {t} | {a:.4f} | {t5} | {t10} |")
else:
    lines.append("_（尚未生成）_")

# ── 6. 数据规模 / 负样本增强 ──
lines += ["", "## 6. 数据规模消融 与 低饱和负样本增强", ""]
for tag, desc in (("scale_025", "25% 数据"), ("scale_05", "50% 数据"),
                  ("abl_both_soft", "100%（1500）"), ("scale_3k", "3300（A+B+C）"),
                  ("negaug", "低饱和增强（1684）")):
    p = f"{TVC}/lora_mllm_eval_{tag}.json"
    if os.path.exists(p):
        r = json.load(open(p))
        lines.append(f"- {desc} (`{tag}`): AUC={r['auc']:.4f}　@5%="
                     f"{(r.get('best_tpr_at_fpr<=5pct') or {}).get('tpr')}　@10%="
                     f"{(r.get('best_tpr_at_fpr<=10pct') or {}).get('tpr')}")
if not any(os.path.exists(f"{TVC}/lora_mllm_eval_{t}.json") for t in ("scale_025", "scale_3k", "negaug")):
    lines.append("_（尚未生成）_")

# ── 7. 分解消融（多 seed）──
lines += ["", "## 7. 约束分解消融（多 seed，软化后）", ""]
for base, name in (("abl_mean_v2", "mean-only (soft)"), ("abl_wp_soft", "wp-only (soft)"),
                   ("abl_both_soft", "both (soft)"), ("sft_lam0", "none")):
    vals = []
    for suf in ("", "_s1", "_s2"):
        p = f"{TVC}/lora_mllm_eval_{base}{suf}.json"
        if os.path.exists(p):
            r = json.load(open(p))
            vals.append((r["auc"], (r.get("best_tpr_at_fpr<=5pct") or {}).get("tpr")))
    if vals:
        a = np.mean([v[0] for v in vals]); t = np.mean([v[1] for v in vals])
        lines.append(f"- {name}: AUC {a:.4f} (n={len(vals)})　@5% {t:.4f}　原始: {vals}")

# ── 8. 鲁棒性 ──
d = load("adv_attack_v2.json")
lines += ["", "## 8. 鲁棒性强化（3 seed × fixed/adaptive τ）", ""]
if d:
    for k, v in d.items():
        if k == "summary": continue
        lines.append(f"**{k}**:")
        for cfg, row in v.items():
            lines.append(f"  - {cfg}: TPR {row['tpr_clean']}→{row['tpr_attacked']} (ASR {row['asr']})")
    if "summary" in d:
        lines.append("跨 seed 汇总: " + "　".join(f"{k}={v['asr_mean']}±{v['asr_std']}"
                                                  for k, v in d["summary"].items()))
else:
    lines.append("_（尚未生成）_")

# ── 8.5 统一协议（1024 + 1000 负样本）──
lines += ["", "## 8.5 统一协议主表与基线（长边 1024 + 1000 负样本）", ""]
for f, name in (("unified_main_1024.json", "NP-PatchLM 3 seed + CLIP"),
                ("unified_baselines.json", "NutNet + 轻量 CNN"),
                ("trufor_unified.json", "TruFor"), ("pscc_unified.json", "PSCC-Net")):
    d = load(f)
    if not d:
        lines.append(f"- {name}: _（尚未生成）_"); continue
    if "tpr" in d:
        for key in ("3src", "4src"):
            if key in d["tpr"]:
                lines.append(f"- {name} {key}: " +
                             "　".join(f"{a}={d['tpr'][key][a]['tpr']}" for a in d["tpr"][key]))
    elif "summary_vlm_mean_std" in d:
        for k, v in d["summary_vlm_mean_std"].items():
            lines.append(f"- {name} {k}: {v['mean']:.4f}±{v['std']:.4f} {v['all']}")
    else:
        lines.append(f"- {name}: keys={list(d.keys())[:6]}")

# ── 9. 黑盒迁移 + 粗定位 ──
lines += ["", "## 9. 黑盒迁移攻击与粗定位", ""]
d = load("blackbox_transfer.json")
if d:
    lines.append(f"- 黑盒迁移: TPR {d['tpr_clean']}→{d['tpr_transferred']} (ASR {d['asr_transfer']}), "
                 f"代理={d['proxy']}")
else:
    lines.append("- 黑盒迁移: _（尚未生成）_")
d = load("patch_localization.json")
if d:
    lines.append(f"- 粗定位（遮挡法 {d['grid']}×{d['grid']}）: 命中率 {d['hit_rate']} "
                 f"(n={d['n_evaluated']})，可视化见 `results/tvc/loc_viz/`")
else:
    lines.append("- 粗定位: _（尚未生成）_")

io.open(OUT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print(f"→ {OUT}（{len(lines)} 行）")
