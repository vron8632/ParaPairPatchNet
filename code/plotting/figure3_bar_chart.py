#!/usr/bin/env python3
"""
Figure 3: 实验结果对比柱状图
- Panel a: 物理补丁检测 (我们的方法 vs Baselines)
- Panel b: 经典篡改检测 (3个数据集)
- Panel c: 消融实验 (三分支贡献)

输出: SVG + PNG, 论文级别
"""

import os, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec

# ── 输出目录 ──
OUT_DIR = '/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard/latex/figures'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Nature风格设置 ──
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2

# ── 颜色方案 ──
C_BLUE   = '#0F4D92'
C_GREEN  = '#8BCF8B'
C_RED    = '#B64342'
C_TEAL   = '#42949E'
C_GOLD   = '#D4A017'
C_GRAY   = '#767676'
C_LIGHT  = '#CFCECE'

def save_fig(fig, name):
    for ext in ['.svg', '.png']:
        fig.savefig(os.path.join(OUT_DIR, name + ext), dpi=300, bbox_inches='tight')
    print(f'  ✅ {name} 已保存')

# ===============================================================
# Panel a: 物理补丁检测对比
# ===============================================================
fig = plt.figure(figsize=(14, 5))
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.35)

# --- Panel a: 物理补丁检测 ---
ax = fig.add_subplot(gs[0])
methods = ['Ours\n(Instance)', 'Baseline\n(Global Feat)', 'Baseline\n(ResNet-18)']
domains = ['Digital Patch', 'Physical Patch']
data = np.array([
    [85.8, 91.0],   # Ours
    [83.2, 94.0],   # Global Feat
    [100.0, 100.0], # ResNet-18
])
n_methods, n_domains = data.shape
x = np.arange(n_methods)
w = 0.3
colors = [C_BLUE, C_TEAL]
for i in range(n_domains):
    offset = (i - 0.5) * w
    bars = ax.bar(x + offset, data[:, i], width=w, label=domains[i],
                  color=colors[i], edgecolor='black', linewidth=0.8)
    for bar, val in zip(bars, data[:, i]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=10)
ax.set_ylabel('Detection Accuracy (%)', fontsize=11)
ax.set_title('a  Physical Patch Detection', fontsize=12, fontweight='bold', loc='left')
ax.set_ylim(0, 115)
ax.legend(fontsize=9, loc='lower right')
ax.axhline(91.0, color=C_BLUE, linestyle='--', linewidth=0.8, alpha=0.5)

# --- Panel b: 经典篡改检测 ---
ax = fig.add_subplot(gs[1])
datasets = ['aug.\n+COCO', 'Columbia', 'CASIA v2']
accs = [97.00, 93.22, 81.22]
colors_b = [C_GREEN, C_BLUE, C_RED]
bars = ax.bar(range(len(datasets)), accs, color=colors_b, edgecolor='black', linewidth=0.8, width=0.5)
for bar, val in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_xticks(range(len(datasets)))
ax.set_xticklabels(datasets, fontsize=10)
ax.set_ylabel('Accuracy (%)', fontsize=11)
ax.set_title('b  Classic Tamper Detection', fontsize=12, fontweight='bold', loc='left')
ax.set_ylim(70, 105)

# --- Panel c: 消融实验 ---
ax = fig.add_subplot(gs[2])
ablation_names = ['Full', 'w/o SPN', 'w/o DCT', 'w/o CLIP']
ablation_accs = [80.67, 81.26, 80.39, 61.97]
ablation_colors = [C_BLUE, C_BLUE, C_BLUE, C_RED]
bars = ax.bar(range(len(ablation_names)), ablation_accs, color=ablation_colors,
              edgecolor='black', linewidth=0.8, width=0.5)
for bar, val in zip(bars, ablation_accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_xticks(range(len(ablation_names)))
ax.set_xticklabels(ablation_names, fontsize=10)
ax.set_ylabel('Accuracy (%)', fontsize=11)
ax.set_title('c  Ablation (CASIA v2)', fontsize=12, fontweight='bold', loc='left')
ax.set_ylim(50, 95)
# 标注CLIP的下降
ax.annotate('−19%', xy=(3, 61.97), xytext=(3.5, 55),
            arrowprops=dict(arrowstyle='->', color=C_RED, lw=1.5),
            fontsize=10, color=C_RED, fontweight='bold')

fig.tight_layout(pad=1.5)
save_fig(fig, 'figure3_experiment_results')
plt.close(fig)

print('✅ 图3 柱状图完成')
