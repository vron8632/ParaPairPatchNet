#!/usr/bin/env python3
"""
Figure 1 (v2): 动机图 — 真实数据 + 精致排版
叙事式布局:
  顶部: 大标题 + 一句话洞察
  中部: (a) 场景图(实例掩码+编号) | (b) 实例取证卡片 | (c) 一致性对比
输出: latex/figures/figure1_motivation_v2.png
"""
import os, torch, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from scipy import ndimage
import torch.nn.functional as F

BASE = '/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard'
OUT = os.path.join(BASE, 'latex/figures/figure1_motivation_v2.png')

td = torch.load(f'{BASE}/results/features/patch_dataset_v2/physical_patch_test.pt', map_location='cpu', weights_only=False)
fd = torch.load(f'{BASE}/results/features/physical_patch_test_features.pt', map_location='cpu', weights_only=False)

IDX = 26
img = td['tensors'][IDX].permute(1, 2, 0).numpy()
masks = [m.astype(bool) for m in fd['masks'][IDX]]
spn, dct, clip = fd['spn_feats'][IDX], fd['dct_feats'][IDX], fd['clip_feats'][IDX]
K = len(masks)

def avg_sim(feat):
    fn = F.normalize(feat, p=2, dim=-1)
    sim = fn @ fn.T
    return (sim.sum(1) - 1) / (sim.shape[0] - 1)

sim = {'SPN': avg_sim(spn).numpy(), 'DCT': avg_sim(dct).numpy(), 'CLIP': avg_sim(clip).numpy()}
overall = np.mean([sim['SPN'], sim['DCT'], sim['CLIP']], axis=0)
anom = int(np.argmin(overall))
normal_avg = np.mean(np.delete(overall, anom))
print(f'K={K}, 最不一致实例 #{anom} (overall={overall[anom]:.3f}), 其他平均={normal_avg:.3f}')

# ── 配色 ──
C_BLUE, C_RED, C_GREEN = '#0F4D92', '#C0392B', '#1E8E5A'
C_GRAY, C_LIGHT = '#8A94A6', '#E8EDF4'
tab = plt.cm.tab10(np.linspace(0, 1, K))

fig = plt.figure(figsize=(16, 9), facecolor='white')
gs = fig.add_gridspec(2, 6, height_ratios=[0.12, 1], hspace=0.35, wspace=0.3,
                      left=0.04, right=0.98, top=0.92, bottom=0.08)

# ── 标题区 ──
ax_t = fig.add_subplot(gs[0, :])
ax_t.axis('off')
ax_t.text(0.5, 0.75, 'Physical patches are foreign objects: forensic inconsistency across instances',
          ha='center', va='center', fontsize=19, fontweight='bold', color=C_BLUE)
ax_t.text(0.5, 0.25, f'Instance #{anom} deviates from the other {K-1} instances in cross-instance '
                     f'SPN/DCT/CLIP consistency ({overall[anom]:.3f} vs {normal_avg:.3f} avg)',
          ha='center', va='center', fontsize=12, color=C_GRAY)

# ── (a) 场景图 ──
axa = fig.add_subplot(gs[1, 0:2])
axa.imshow(img)
for k in range(K):
    er = ndimage.binary_erosion(masks[k], iterations=2)
    contour = masks[k] & ~er
    ys, xs = np.where(contour)
    lw = 3.0 if k == anom else 1.5
    axa.scatter(xs, ys, s=0.4, color=(C_RED if k == anom else tab[k]), linewidths=0)
    # 质心编号
    cy, cx = ndimage.center_of_mass(masks[k])
    axa.scatter(cx, cy, s=110, color='white', edgecolor=(C_RED if k == anom else C_BLUE), linewidth=2, zorder=5)
    axa.text(cx, cy, str(k), ha='center', va='center', fontsize=11, fontweight='bold',
             color=(C_RED if k == anom else C_BLUE), zorder=6)
axa.set_title('(a) Scene with instance segmentation', fontsize=13, fontweight='bold', color=C_BLUE, pad=8)
axa.axis('off')

# ── (b) 实例取证卡片 ──
axb = fig.add_subplot(gs[1, 2:4])
axb.set_xlim(0, 1); axb.set_ylim(0, K + 0.6); axb.axis('off')
axb.set_title('(b) Per-instance forensic consistency', fontsize=13, fontweight='bold', color=C_BLUE, pad=8)
for k in range(K):
    y = K - 0.5 - k  # 从上到下
    is_anom = (k == anom)
    edge = C_RED if is_anom else C_GRAY
    fc = '#FDEBEA' if is_anom else '#F6F9FC'
    box = FancyBboxPatch((0.03, y - 0.36), 0.94, 0.72, boxstyle='round,pad=0.02',
                         facecolor=fc, edgecolor=edge, linewidth=(2.5 if is_anom else 1.2))
    axb.add_patch(box)
    # 编号圆
    axb.add_patch(Circle((0.09, y), 0.14, facecolor=(C_RED if is_anom else C_BLUE), edgecolor='white', linewidth=1.5))
    axb.text(0.09, y, str(k), ha='center', va='center', color='white', fontsize=10, fontweight='bold')
    # 三特征小条 (相对1.0)
    bar_x0, bar_w = 0.24, 0.5
    for j, fn in enumerate(['SPN', 'DCT', 'CLIP']):
        v = sim[fn][k]
        bx = bar_x0 + j * 0.18
        axb.barh(y, bar_w, height=0.12, left=bx, color='#D8DFE8', edgecolor='none')
        axb.barh(y, (v - 0.5) / 0.5 * bar_w, height=0.12, left=bx,
                 color=(C_RED if is_anom else C_GREEN), edgecolor='none')
        axb.text(bx + bar_w + 0.03, y, f'{v:.2f}', va='center', fontsize=7.5,
                 color=(C_RED if is_anom else '#5A6472'))
    # 综合分数 + 标记
    axb.text(0.945, y, f'{overall[k]:.3f}', ha='right', va='center', fontsize=10,
             fontweight='bold', color=(C_RED if is_anom else '#3C4757'))
    axb.text(0.97, y, '✗' if is_anom else '✓', ha='center', va='center', fontsize=13,
             fontweight='bold', color=(C_RED if is_anom else C_GREEN))

# ── (c) 一致性对比 ──
axc = fig.add_subplot(gs[1, 4:6])
vals = [normal_avg, overall[anom]]
labels = [f'Other {K-1} instances', f'Instance #{anom}']
colors = [C_GREEN, C_RED]
bars = axc.bar(labels, vals, color=colors, width=0.45, edgecolor='white', linewidth=1.5)
for b, v in zip(bars, vals):
    axc.text(b.get_x() + b.get_width()/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom',
             fontsize=13, fontweight='bold')
axc.set_ylim(0.5, 1.0)
axc.set_ylabel('Avg cross-instance similarity', fontsize=11)
axc.set_title('(c) Cross-instance consistency', fontsize=13, fontweight='bold', color=C_BLUE, pad=8)
axc.axhline(1.0, color='#CCCCCC', lw=0.8, ls=':')
axc.tick_params(axis='x', labelsize=11)
axc.spines[['top', 'right']].set_visible(False)
# 标注
axc.text(0.5, 0.52, 'Forensic inconsistency → detectable', ha='center', fontsize=11,
         fontweight='bold', color=C_RED, transform=axc.transAxes)

fig.savefig(OUT, dpi=200, bbox_inches='tight', facecolor='white')
print(f'✅ 已保存: {OUT}')
