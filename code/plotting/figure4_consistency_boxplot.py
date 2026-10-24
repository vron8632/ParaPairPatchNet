#!/usr/bin/env python3
"""Figure 4: 跨实例一致性得分分布箱线图"""
import os, sys, torch, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = '/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard'
sys.path.insert(0, os.path.join(BASE, 'code'))
OUT_DIR = os.path.join(BASE, 'latex/figures')
from models.consistency import CrossInstanceConsistencyV2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cic = CrossInstanceConsistencyV2()

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 13
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
fig.patch.set_facecolor('white')
colors = {'clean': '#0F4D92', 'digital_patch': '#D4A017', 'physical_patch': '#B64342'}
domains = ['clean', 'digital_patch', 'physical_patch']
feat_names = ['SPN', 'DCT', 'CLIP']

for col, (d_name, d_label) in enumerate(zip(['spn','dct','clip'], feat_names)):
    data_vals = {d: [] for d in domains}
    for domain in domains:
        d = torch.load(os.path.join(BASE, 'results/features', f'{domain}_test_features.pt'),
                       map_location='cpu', weights_only=False)
        for i in range(len(d['labels'])):
            feats = d[f'{d_name}_feats'][i]
            if feats.shape[0] >= 2:
                feats_n = torch.nn.functional.normalize(feats, p=2, dim=-1)
                sim = (feats_n @ feats_n.T).cpu().numpy()
                triu = np.triu_indices_from(sim, k=1)
                data_vals[domain].extend(sim[triu].tolist())
    
    bp = axes[col].boxplot([data_vals[d] for d in domains], 
                            patch_artist=True, widths=0.5,
                            medianprops=dict(color='black', lw=1.5))
    for patch, (_, c) in zip(bp['boxes'], colors.items()):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    axes[col].set_xticklabels(['Clean', 'Digital', 'Physical'], fontsize=13)
    axes[col].set_title(f'{d_label} Cross-Instance Similarity', fontsize=13, fontweight='bold')
    axes[col].set_ylabel('Cosine Similarity', fontsize=13)
    axes[col].set_ylim(0, 1.05)

fig.tight_layout(pad=1.5)
fig.savefig(os.path.join(OUT_DIR, 'figure4_consistency.png'), dpi=300, bbox_inches='tight')
print('✅ 一致性箱线图已保存 (替换t-SNE)')
plt.close(fig)
