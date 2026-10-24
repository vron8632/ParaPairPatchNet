#!/usr/bin/env python3
"""
Figure 4: t-SNE特征空间可视化
展示 clean / digital_patch / physical_patch 在特征空间中的分布
每个类别用不同颜色，至少3种颜色
"""

import os, torch, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# ── 路径 ──
BASE = '/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard'
OUT_DIR = os.path.join(BASE, 'latex/figures')
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.size'] = 13
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

C_BLUE   = '#0F4D92'
C_GREEN  = '#8BCF8B'
C_RED    = '#B64342'
C_TEAL   = '#42949E'
C_ORANGE = '#E28E2C'
C_GRAY   = '#767676'

print('加载特征...')
all_feats, all_labels, all_domains = [], [], []
domain_colors = {'clean': C_BLUE, 'digital_patch': C_ORANGE, 'physical_patch': C_RED}
domain_markers = {'clean': 'o', 'digital_patch': 's', 'physical_patch': '^'}

for domain in ['clean', 'digital_patch', 'physical_patch']:
    data = torch.load(os.path.join(BASE, 'results/features', f'{domain}_test_features.pt'),
                      map_location='cpu', weights_only=False)
    n_labels = len(data['labels'])
    n_use = min(n_labels, 300)
    for i in range(n_use):
        spn = data['spn_feats'][i].mean(0).numpy()
        dct = data['dct_feats'][i].mean(0).numpy()
        clip = data['clip_feats'][i].mean(0).numpy()
        feat = np.concatenate([spn, dct, clip])
        feat = feat / (np.linalg.norm(feat) + 1e-10)
        all_feats.append(feat)
        all_labels.append(domain)
    print(f'  {domain}: {n_use} samples')

X = np.array(all_feats)
print(f'特征矩阵: {X.shape}')

print('运行PCA降维 (661→50)...')
from sklearn.decomposition import PCA
pca = PCA(n_components=50, random_state=42)
X_pca = pca.fit_transform(X)
print(f'PCA方差保留: {pca.explained_variance_ratio_.sum():.2%}')

print('运行t-SNE...')
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, 
            random_state=42)
X_tsne = tsne.fit_transform(X_pca)

# ── 画图 ──
fig, ax = plt.subplots(figsize=(10, 8))
for domain in ['clean', 'digital_patch', 'physical_patch']:
    mask = [l == domain for l in all_labels]
    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
               c=domain_colors[domain], marker=domain_markers[domain],
               label=domain.replace('_', ' ').title(),
               s=30, alpha=0.7, edgecolors='black', linewidth=0.3)

ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
ax.set_title('t-SNE Visualization of Forensic Features', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='best', framealpha=0.8)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

fig.tight_layout()
for ext in ['.svg', '.png']:
    fig.savefig(os.path.join(OUT_DIR, 'figure4_tsne' + ext), dpi=300, bbox_inches='tight')
print('✅ t-SNE图已保存')
plt.close(fig)
