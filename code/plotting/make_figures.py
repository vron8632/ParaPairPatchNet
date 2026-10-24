#!/usr/bin/env python3
"""
按 nature-figure 风格契约绘制论文图（matplotlib，PDF 300dpi）
==============================================================
风格契约（docs/skills/nature-figure/stance.md）要点：
  * 白底；每图一个克制配色：1 neutral 灰 + 1 signal 蓝 + 1 accent 橙；
    红/绿仅用于方向性（失败/成功）暗示。
  * hero panel + 辅助面板；类别标签直接标注（避免远距离图例）。
  * 统计口径与 n 写进图注/图内；同类面板不确定度定义一致。
  * 保持模型名规范大小写（NP-PatchLM），禁止盲目 .title()。

产出（paper/figures/）：
  fig_tail.pdf        分来源 P(YES) 箱线（hero）+ 尾部来源构成（辅助）
  fig_cascade.pdf     级联精度--成本曲线（hero）
  fig_mechanism.pdf   (a) 分支 AUC  (b) 线性探针
  fig_robustness.pdf  (a) clean/对抗 AUC  (b) ASR
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 仓库根
FIG = os.path.join(WS, "paper")
os.makedirs(FIG, exist_ok=True)

# ── 配色（1 neutral + 1 signal + 1 accent；红绿仅作方向暗示）──
C_NEUTRAL = "#9aa5b1"
C_SIGNAL  = "#3b6fb6"
C_ACCENT  = "#e08214"
C_FAIL    = "#c0392b"
C_OK      = "#2e7d32"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "xtick.direction": "out", "ytick.direction": "out",
    "pdf.fonttype": 42, "ps.fonttype": 42,       # 嵌入 TrueType（出版安全）
    "figure.dpi": 300, "savefig.dpi": 300,
})

TVC = "/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard/results/tvc"


def fig_tail():
    """Hero：分来源 P(YES) 箱线 + p90 直接标注（箱体上方）；辅助：尾部来源构成。"""
    d = json.load(open(f"{TVC}/confound_main_withscores.json"))
    neg = d["neg_scores_by_source"]; pos = np.array(d["pos_scores"])
    order = ["casia_au", "coco_val", "inria_noped", "columbia_au", "APRICOT\n(patch, pos)"]
    data = [np.array(neg[s]) for s in order[:-1]] + [pos]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9),
                                  gridspec_kw={"width_ratios": [2.1, 1.0]})
    bp = ax.boxplot(data, tick_labels=order, showmeans=False, widths=0.5,
                    patch_artist=True, medianprops=dict(color="black", lw=1.2))
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(C_FAIL if i == 3 else (C_SIGNAL if i == 4 else C_NEUTRAL))
        box.set_alpha(0.75)
    # p90 标注：统一放在箱体正上方，避开箱体与 whisker
    for i, arr in enumerate(data):
        p90 = float(np.percentile(arr, 90))
        color = C_FAIL if i == 3 else "dimgray"
        ax.annotate(f"p90={p90:.2f}", xy=(i + 1, min(p90, 0.98)),
                    xytext=(i + 1, 1.06), ha="center", fontsize=7.5, color=color,
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="gray",
                                    shrinkA=0, shrinkB=1))
    ax.set_ylabel("P(YES), patch probability")
    ax.set_ylim(-0.03, 1.16)
    ax.tick_params(axis="x", labelsize=7)

    thr = float(np.percentile(np.concatenate(data[:-1]), 80))
    tail = [s for s, arr in zip(order[:-1], data[:-1]) for _ in arr if _ >= thr]
    names = ["casia_au", "coco_val", "inria_noped", "columbia_au"]
    cnt = [tail.count(s) for s in names]
    ax2.bar(range(len(names)), cnt,
            color=[C_NEUTRAL, C_NEUTRAL, C_NEUTRAL, C_FAIL], alpha=0.85)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels([n.split("_")[0] for n in names], rotation=30,
                        ha="right", fontsize=7.5)
    for i, c in enumerate(cnt):
        ax2.text(i, c + 0.5, f"{c}", ha="center", fontsize=8)
    ax2.set_ylabel("clean images in top-20% tail")
    ax2.set_title("tail composition (P(YES) ≥ p80)", fontsize=8.5)
    ax2.grid(axis="y", ls=":", alpha=0.4)
    fig.text(0.01, 0.005, "n = 60 per source, 100 positives",
             fontsize=7, color="dimgray")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(f"{FIG}/fig_tail.pdf"); fig.savefig(f"{FIG}/fig_tail.png")
    plt.close(fig)
    print("fig_tail done")


def fig_cascade():
    # 与其它单栏图保持一致的视觉比例：figsize 6.4in 插入 \columnwidth(≈3.3in)，
    # 缩放 ~0.52；字号同步放大，使有效字号与 fig_mechanism/fig_robustness 相当。
    d = json.load(open(f"{TVC}/cascade_eval.json"))
    pts = sorted((v["mlm_call_rate"], v["auc"]) for k, v in d.items()
                 if k.startswith("cascade_") or k in ("mllm_only", "visual_only"))
    x = [p[0] for p in pts]; y = [p[1] for p in pts]

    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.plot(x, y, "-o", color=C_SIGNAL, lw=1.4, ms=4)
    ax.plot([1.0], [d["mllm_only"]["auc"]], "s", color=C_OK, ms=6)
    ax.plot([0.0], [d["visual_only"]["auc"]], "^", color=C_NEUTRAL, ms=6)
    xi = x.index(0.69)
    ax.plot([x[xi]], [y[xi]], "o", color=C_ACCENT, ms=7, zorder=5)

    ax.annotate("MLLM only (AUC 0.978)", xy=(1.0, d["mllm_only"]["auc"]),
                xytext=(0.52, 0.995), fontsize=7.5, color=C_OK,
                arrowprops=dict(arrowstyle="-", lw=0.6, color="gray"))
    ax.annotate("visual triage only (AUC 0.787)", xy=(0.0, d["visual_only"]["auc"]),
                xytext=(0.10, 0.772), fontsize=7.5, color="dimgray",
                arrowprops=dict(arrowstyle="-", lw=0.6, color="gray"))
    ax.annotate("recommended:\n69% calls, AUC 0.972", xy=(x[xi], y[xi]),
                xytext=(0.46, 0.845), fontsize=7.5, color=C_ACCENT,
                arrowprops=dict(arrowstyle="->", lw=0.7, color=C_ACCENT))

    ax.set_xlabel("fraction of images routed to the VLM", fontsize=9)
    ax.set_ylabel("AUC (same-source val, n=1000)", fontsize=9)
    ax.set_ylim(0.74, 1.03); ax.set_xlim(-0.04, 1.08)
    ax.tick_params(labelsize=8)
    ax.grid(ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig_cascade.pdf"); fig.savefig(f"{FIG}/fig_cascade.png")
    plt.close(fig)
    print("fig_cascade done")


def fig_mechanism():
    d1 = json.load(open(f"{TVC}/diag_branches.json"))["tasks"]
    d2 = json.load(open(f"{TVC}/lora_probe.json"))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.5))

    # (a) 分支 AUC（分组条形）
    branches = ["SPN", "DCT", "CLIP"]
    dig = [d1["clean_vs_digital"][b.lower()]["auc"] for b in branches]
    phy = [d1["clean_vs_physical"][b.lower()]["auc"] for b in branches]
    x = np.arange(3); w = 0.36
    ax1.bar(x - w / 2, dig, w, color=C_NEUTRAL, alpha=0.85, label="digital patch (pos)")
    ax1.bar(x + w / 2, phy, w, color=C_ACCENT, alpha=0.85, label="physical patch (pos)")
    for xi, (a, b) in enumerate(zip(dig, phy)):
        ax1.text(xi - w / 2, a + 0.015, f"{a:.2f}", ha="center", fontsize=7)
        ax1.text(xi + w / 2, b + 0.015, f"{b:.2f}", ha="center", fontsize=7)
    ax1.axhline(0.5, ls=":", lw=0.8, color=C_FAIL)
    ax1.text(2.42, 0.51, "chance", fontsize=7, color=C_FAIL)
    ax1.set_xticks(x); ax1.set_xticklabels(branches)
    ax1.set_ylabel("AUC vs. clean")
    ax1.set_ylim(0, 1.0)
    ax1.set_title("(a) forensic cues fail on patches\n(both digital and physical)", fontsize=8.5)
    ax1.legend(fontsize=7, frameon=False, loc="upper left")
    ax1.grid(axis="y", ls=":", alpha=0.4)

    # (b) 线性探针
    names = ["base\n(no LoRA)", "LoRA\nfine-tuned"]
    aucs = [d2["base"]["auc_mean"], d2["finetuned"]["auc_mean"]]
    errs = [d2["base"]["auc_std"], d2["finetuned"]["auc_std"]]
    ax2.bar(range(2), aucs, 0.5, color=[C_NEUTRAL, C_SIGNAL], alpha=0.9,
            yerr=errs, capsize=3)
    for i, (a, e) in enumerate(zip(aucs, errs)):
        ax2.text(i, a + e + 0.015, f"{a:.3f}", ha="center", fontsize=8)
    ax2.set_xticks(range(2)); ax2.set_xticklabels(names, fontsize=8)
    ax2.set_ylim(0.5, 1.02)
    ax2.set_ylabel("linear-probe AUC (5-fold)")
    ax2.set_title("(b) frozen representations are already\nseparable; fine-tuning aligns readout", fontsize=8.5)
    ax2.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig_mechanism.pdf"); fig.savefig(f"{FIG}/fig_mechanism.png")
    plt.close(fig)
    print("fig_mechanism done")


def fig_robustness():
    r1 = json.load(open(f"{TVC}/adv_attack_pgd.json"))
    r2 = json.load(open(f"{TVC}/adv_attack_pgd_robust.json"))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.7))

    names = ["main", "+ adv.\nfine-tuning"]
    clean = [r1["workpoints"]["tau=0.5"]["tpr_clean"], r2["workpoints"]["tau=0.5"]["tpr_clean"]]
    adv = [r1["workpoints"]["tau=0.5"]["tpr_adv"], r2["workpoints"]["tau=0.5"]["tpr_adv"]]
    x = np.arange(2); w = 0.32
    ax1.bar(x - w / 2, clean, w, color=C_NEUTRAL, alpha=0.9, label="clean")
    ax1.bar(x + w / 2, adv, w, color=C_FAIL, alpha=0.85, label="under PGD")
    for xi, (a, b) in enumerate(zip(clean, adv)):
        ax1.text(xi - w / 2, a + 0.02, f"{a:.2f}", ha="center", fontsize=6.5)
        ax1.text(xi + w / 2, b + 0.02, f"{b:.2f}", ha="center", fontsize=6.5)
    ax1.set_xticks(x); ax1.set_xticklabels(names, fontsize=8)
    ax1.set_ylabel("TPR @ tau=0.5", fontsize=8)
    ax1.set_ylim(0, 1.05)
    ax1.set_title("(a) white-box PGD (eps=0.03, 10 steps)", fontsize=8)
    ax1.legend(fontsize=6.5, frameon=False, ncol=2,
               loc="upper center", bbox_to_anchor=(0.5, -0.16))
    ax1.grid(axis="y", ls=":", alpha=0.4)

    asr = [r1["workpoints"]["tau=0.5"]["tpr_drop"], r2["workpoints"]["tau=0.5"]["tpr_drop"]]
    ax2.bar(range(2), asr, 0.45, color=[C_FAIL, C_OK], alpha=0.85)
    for i, a in enumerate(asr):
        ax2.text(i, a + 0.015, f"{a*100:.0f}%", ha="center", fontsize=8.5)
    ax2.set_xticks(range(2)); ax2.set_xticklabels(names, fontsize=8)
    ax2.set_ylabel("attack success rate (TPR drop)", fontsize=8)
    ax2.set_ylim(0, 0.5)
    ax2.set_title("(b) attack success rate", fontsize=8)
    ax2.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig_robustness.pdf"); fig.savefig(f"{FIG}/fig_robustness.png")
    plt.close(fig)
    print("fig_robustness done")


if __name__ == "__main__":
    fig_tail(); fig_cascade(); fig_mechanism(); fig_robustness()
    print("all figures ->", FIG)
