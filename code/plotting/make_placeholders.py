#!/usr/bin/env python3
"""
⚠️ 本脚本会覆盖 paper/fig_motivation.png 与 fig_framework.png；正式图就位后勿再运行。

16:9 占位图（motivation + framework）—— 供 Overleaf 排版预览
正式图由 gpt-image-2 按 paper/figures/gpt-image-2_*.md 生成后原位替换。
生成 fig_motivation.pdf / fig_framework.pdf（16:9, 3840x2160 @300dpi 等比）。
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '')
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

W, H = 8.0, 4.5          # 16:9 版面尺寸（英寸）；300dpi → 2400x1350 等比占位
GRAY = "#9aa5b1"
BLUE = "#3b6fb6"
ORANGE = "#e08214"
RED = "#c0392b"
GREEN = "#2e7d32"


def _panel(ax, x, y, w, h, title, body, edge="#333333", fc="white",
           fts=11, fbs=8.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008",
                                linewidth=1.2, edgecolor=edge, facecolor=fc))
    ax.text(x + w / 2, y + h - 0.06, title, ha="center", va="top",
            fontsize=fts, fontweight="bold")
    ax.text(x + w / 2, y + h / 2 - 0.06, body, ha="center", va="center",
            fontsize=fbs, color="#444444")


def placeholder_motivation():
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.text(0.5, 0.955, "[PLACEHOLDER] Fig. 1 — Motivation  (16:9, to be replaced by gpt-image-2 output)",
            ha="center", fontsize=12, style="italic", color="#666666")
    ax.plot([0.06, 0.94], [0.90, 0.90], ls="--", lw=0.8, color=GRAY)
    _panel(ax, 0.06, 0.22, 0.26, 0.62,
           "(a) Threat",
           "Printed adversarial\npatch attached to\na real object")
    _panel(ax, 0.37, 0.22, 0.26, 0.62,
           "(b) Forensic: blind",
           "no heatmap response\non the patch\n\nforensic detectors:\nfail\npatch-specific:\nunreliable at low FPR",
           edge=RED)
    ax.text(0.63 - 0.012, 0.245, "✗", fontsize=26, color=RED, ha="center")
    _panel(ax, 0.68, 0.22, 0.26, 0.62,
           "(c) NP-PatchLM",
           "highlight box on\npatch + 'YES' bubble\n\nreliable detection\nat controlled FPR",
           edge=GREEN)
    ax.text(0.94 - 0.012, 0.245, "✓", fontsize=26, color=GREEN, ha="center")
    for x0, x1 in ((0.32, 0.37), (0.63, 0.68)):
        ax.add_patch(FancyArrowPatch((x0, 0.53), (x1, 0.53), arrowstyle="-|>",
                                     mutation_scale=14, lw=1.4, color="#555555"))
    ax.text(0.5, 0.12, "Forensic cues are blind to physical patches; semantic vision-language cues are not.",
            ha="center", fontsize=12, fontweight="bold")
    ax.text(0.5, 0.065, "branch diagnosis: low-level (noise/frequency) cues uninformative; semantic cues informative",
            ha="center", fontsize=8.5, color="#777777")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_motivation.pdf"))
    fig.savefig(os.path.join(FIG, "fig_motivation.png"), dpi=200)
    plt.close(fig)
    print("fig_motivation done (16:9)")


def placeholder_framework():
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.text(0.5, 0.955, "[PLACEHOLDER] Fig. 2 — NP-PatchLM framework  (16:9, to be replaced by gpt-image-2 output)",
            ha="center", fontsize=12, style="italic", color="#666666")
    # 区块 1: 判别器（面板 0.04-0.32）
    _panel(ax, 0.04, 0.30, 0.28, 0.55,
           "VLM Detector",
           "Image → Vision Encoder\n(frozen)\n+ LoRA adapter\n(trainable)\n→ LLM Decoder\n→ YES / NO tokens\n\nP(YES)=σ(z_YES−z_NO)",
           fc="#eef3fa", edge=BLUE, fts=10.5, fbs=7.8)
    # 区块 2: 训练目标（面板 0.37-0.63）
    _panel(ax, 0.37, 0.30, 0.26, 0.55,
           "FPR Loss",
           "L = L_CE + λ·Φ\nΦ = ReLU(mean p_neg\n − β)²      (i)\n + mean 1[p_neg ≥ τ0]\n     (ii)\n\n(i) mean term:\nweak alone\n(ii) operating-point\nterm: effective\n\nadversarial branch:\nmore robust",
           fc="#fdf3e6", edge=ORANGE, fts=10.5, fbs=7.6)
    # 区块 3: 级联（面板 0.68-0.96）
    _panel(ax, 0.68, 0.30, 0.28, 0.55,
           "Lightweight Cascade",
           "Image → CLIP linear\nscore s\n→ s ∈ [τ_lo, τ_hi]?\n  no  → direct decision\n      (far fewer VLM calls)\n  yes → VLM review\n\nthresholds: negative\nempirical CDF",
           fc="#eef7ee", edge=GREEN, fts=10.5, fbs=7.8)
    for x0, x1 in ((0.325, 0.365), (0.645, 0.675)):
        ax.add_patch(FancyArrowPatch((x0, 0.575), (x1, 0.575), arrowstyle="-|>",
                                     mutation_scale=13, lw=1.3, color="#555555"))
    ax.text(0.345, 0.605, "train", fontsize=8, ha="center", color="#555555")
    ax.text(0.66, 0.605, "deploy", fontsize=8, ha="center", color="#555555")
    ax.add_patch(FancyArrowPatch((0.18, 0.30), (0.42, 0.30), arrowstyle="-|>",
                                 mutation_scale=11, lw=1.0, color=RED,
                                 linestyle="--"))
    ax.text(0.30, 0.263, "online adversarial fine-tuning (optional)", fontsize=7.5,
            ha="center", color=RED)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_framework.pdf"))
    fig.savefig(os.path.join(FIG, "fig_framework.png"), dpi=200)
    plt.close(fig)
    print("fig_framework done (16:9)")


if __name__ == "__main__":
    placeholder_motivation()
    placeholder_framework()
