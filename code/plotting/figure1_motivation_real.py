#!/usr/bin/env python3
"""
Figure 1 (v2): 动机图 — 真实数据版
用真实物理补丁测试图像 (K=8) 拼装动机图:
(a) 原始物理补丁图像
(b) 实例分割掩码轮廓 (8 个实例, 不同颜色)
(c) 最不一致实例高亮 (红色遮罩) — "跨实例取证不一致"
(d) 每实例 SPN/DCT/CLIP 跨实例平均相似度条形图

输出: latex/figures/figure1_motivation_real.png
"""
import os, torch, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F

BASE = '/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard'
OUT = os.path.join(BASE, 'latex/figures/figure1_motivation_real.png')

# 加载数据
td = torch.load(f'{BASE}/results/features/patch_dataset_v2/physical_patch_test.pt', map_location='cpu', weights_only=False)
fd = torch.load(f'{BASE}/results/features/physical_patch_test_features.pt', map_location='cpu', weights_only=False)

IDX = 26  # K=8 的物理补丁图
img = td['tensors'][IDX].permute(1, 2, 0).numpy()          # HWC [0,1]
masks = fd['masks'][IDX]                                    # list of (512,512)
spn, dct, clip = fd['spn_feats'][IDX], fd['dct_feats'][IDX], fd['clip_feats'][IDX]

K = len(masks)
print(f'idx={IDX}: K={K}, img={img.shape}')

# 每实例与其他实例的平均余弦相似度
def avg_sim(feat):
    fn = F.normalize(feat, p=2, dim=-1)
    sim = fn @ fn.T
    return (sim.sum(1) - 1) / (sim.shape[0] - 1)

sim = {'SPN': avg_sim(spn).numpy(), 'DCT': avg_sim(dct).numpy(), 'CLIP': avg_sim(clip).numpy()}
# 综合不一致分数: 三特征平均相似度 (越低越不一致)
overall = np.mean([sim['SPN'], sim['DCT'], sim['CLIP']], axis=0)
anom = int(np.argmin(overall))  # 最不一致实例
print(f'综合相似度: {[round(x,3) for x in overall]}')
print(f'最不一致实例: #{anom} (overall={overall[anom]:.3f})')

# 配色
C_BLUE, C_RED, C_GREEN = '#0F4D92', '#B64342', '#2E9E44'
mask_colors = plt.cm.tab10(np.linspace(0, 1, K))

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 11})

def overlay_contour(ax, base_img, mask, color, lw=2.5):
    ax.imshow(base_img)
    # 轮廓: 用掩码梯度找边界
    from scipy import ndimage
    er = ndimage.binary_erosion(mask, iterations=2)
    contour = mask & ~er
    ys, xs = np.where(contour)
    if len(xs):
        ax.scatter(xs, ys, s=0.5, c=color, linewidths=0)

# (a) 原始图像
ax = axes[0, 0]
ax.imshow(img)
ax.set_title('(a) Physical Patch Image', fontweight='bold', color=C_BLUE)
ax.axis('off')

# (b) 实例分割 (全部掩码轮廓)
ax = axes[0, 1]
ax.imshow(img)
for k in range(K):
    overlay_contour(ax, img, masks[k], mask_colors[k])
ax.set_title(f'(b) Instance Segmentation (K={K})', fontweight='bold', color=C_BLUE)
ax.axis('off')

# (c) 最不一致实例高亮
ax = axes[1, 0]
ax.imshow(img)
for k in range(K):
    overlay_contour(ax, img, masks[k], mask_colors[k], lw=1.5)
# 高亮异常实例: 红色半透明遮罩
m = masks[anom].astype(float)
rgb = np.stack([m, np.zeros_like(m), np.zeros_like(m)], axis=-1)
ax.imshow(rgb, alpha=0.4)
overlay_contour(ax, img, masks[anom], C_RED, lw=4)
ax.set_title(f'(c) Forensic Inconsistency (instance #{anom})', fontweight='bold', color=C_RED)
ax.axis('off')

# (d) 一致性条形图
ax = axes[1, 1]
x = np.arange(K)
width = 0.27
feat_names = ['SPN', 'DCT', 'CLIP']
for j, fn in enumerate(feat_names):
    colors = [C_RED if i == anom else '#9fb3c8' for i in range(K)]
    ax.bar(x + (j-1)*width, sim[fn], width, label=fn, color=colors)
ax.axhline(np.mean([np.delete(sim[fn], anom).mean() for fn in feat_names]),
           color=C_BLUE, ls='--', lw=1.5, label='clean-avg')
ax.set_xticks(x)
ax.set_xticklabels([f'#{i}' for i in range(K)])
ax.set_ylabel('Avg cross-instance similarity')
ax.set_title(f'(d) Cross-Instance Consistency (K={K})', fontweight='bold', color=C_BLUE)
ax.legend(fontsize=9, loc='lower left')
ax.set_ylim(0, 1.05)

fig.suptitle('Motivation: a patch is a foreign object — its forensic features deviate from other instances',
             fontsize=13, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.96])
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches='tight')
print(f'✅ 已保存: {OUT}')
