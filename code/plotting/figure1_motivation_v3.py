#!/usr/bin/env python3
"""
Figure 1 (v3): 动机图 — 实例图像墙版
用真实实例裁剪图展示"补丁实例与其他实例不一致":
  (a) 场景图(掩码+编号)  (b) 实例图像墙: 每实例裁剪图 + 一致性分数 + ✓/✗
输出: latex/figures/figure1_motivation_v3.png
"""
import os, torch, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import ndimage
import torch.nn.functional as F

BASE = '/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard'
OUT = os.path.join(BASE, 'latex/figures/figure1_motivation_v3.png')

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

def crop_instance(img, mask, pad=10):
    ys, xs = np.where(mask)
    y0, y1 = max(0, ys.min()-pad), min(img.shape[0], ys.max()+pad+1)
    x0, x1 = max(0, xs.min()-pad), min(img.shape[1], xs.max()+pad+1)
    return img[y0:y1, x0:x1]

# ── 配色 ──
C_BLUE, C_RED, C_GREEN = '#0F4D92', '#C0392B', '#1E8E5A'
C_GRAY = '#8A94A6'
tab = plt.cm.tab10(np.linspace(0, 1, K))

fig = plt.figure(figsize=(17, 9), facecolor='white')
gs = fig.add_gridspec(2, 3, height_ratios=[0.14, 1], hspace=0.3, wspace=0.25,
                      left=0.04, right=0.98, top=0.9, bottom=0.08)

# ── 标题区 ──
ax_t = fig.add_subplot(gs[0, :])
ax_t.axis('off')
ax_t.text(0.5, 0.7, 'Physical patches are foreign objects: forensic inconsistency across instances',
          ha='center', va='center', fontsize=19, fontweight='bold', color=C_BLUE)
ax_t.text(0.5, 0.2, f'Instance #{anom} (red) deviates from the other {K-1} instances in cross-instance '
                     f'SPN/DCT/CLIP consistency ({overall[anom]:.3f} vs {normal_avg:.3f} avg)',
          ha='center', va='center', fontsize=12, color=C_GRAY)

# ── (a) 场景图 (左 40%) ──
axa = fig.add_subplot(gs[1, 0])
axa.imshow(img)
for k in range(K):
    er = ndimage.binary_erosion(masks[k], iterations=2)
    contour = masks[k] & ~er
    ys, xs = np.where(contour)
    lw = 3.0 if k == anom else 1.5
    axa.scatter(xs, ys, s=0.4, color=(C_RED if k == anom else tab[k]), linewidths=0)
    cy, cx = ndimage.center_of_mass(masks[k])
    axa.scatter(cx, cy, s=120, facecolor='white', edgecolor=(C_RED if k == anom else C_BLUE),
                linewidth=2, zorder=5)
    axa.text(cx, cy, str(k), ha='center', va='center', fontsize=11, fontweight='bold',
             color=(C_RED if k == anom else C_BLUE), zorder=6)
axa.set_title('(a) Scene with instance segmentation', fontsize=13, fontweight='bold',
              color=C_BLUE, pad=8)
axa.axis('off')

# ── (b) 实例图像墙 (右 60%) ──
# 每个实例裁剪图 + 分数标注
n_cols = 4
n_rows = int(np.ceil(K / n_cols))
# 用子网格画实例图
gsb = gs[1, 1:].subgridspec(n_rows, n_cols, hspace=0.75, wspace=0.08)
for k in range(K):
    row, col = divmod(k, n_cols)
    ax = fig.add_subplot(gsb[row, col])
    crop = crop_instance(img, masks[k])
    ax.imshow(crop)
    is_anom = (k == anom)
    # 边框
    for sp in ax.spines.values():
        sp.set_color(C_RED if is_anom else C_GRAY)
        sp.set_linewidth(3.0 if is_anom else 1.5)
    ax.set_xticks([]); ax.set_yticks([])
    # 编号 + 分数 + 状态
    ax.set_title(f'#{k}  {overall[k]:.3f}  ' + ('✗' if is_anom else '✓'),
                 fontsize=11, fontweight='bold',
                 color=(C_RED if is_anom else '#3C4757'), pad=4)

# 空位隐藏
for k in range(K, n_rows * n_cols):
    row, col = divmod(k, n_cols)
    fig.add_subplot(gsb[row, col]).axis('off')

# (b) 区域标题
axb_t = fig.add_subplot(gs[1, 1:])
axb_t.axis('off')
axb_t.text(0.5, -0.02, '(b) Per-instance forensic consistency — the patched object deviates',
           ha='center', va='top', fontsize=13, fontweight='bold', color=C_BLUE)

fig.savefig(OUT, dpi=200, bbox_inches='tight', facecolor='white')
print(f'✅ 已保存: {OUT}')
