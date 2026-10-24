#!/usr/bin/env python3
"""
Figure 6: ROC曲线 — 物理补丁检测性能分析
计算AUC并绘制ROC曲线，支持不同阈值下的TPR/FPR分析
"""

import os, sys, torch, numpy as np, matplotlib, sklearn.metrics
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = '/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard'
sys.path.insert(0, os.path.join(BASE, 'code'))
OUT_DIR = os.path.join(BASE, 'latex/figures')
os.makedirs(OUT_DIR, exist_ok=True)

from models.consistency import CrossInstanceConsistencyV2
from models.fusion_classifier import FusionClassifierV2
from scripts.train import compute_fused_features

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cic = CrossInstanceConsistencyV2()

# 加载物理域测试集
data_clean = torch.load(os.path.join(BASE, 'results/features/clean_test_features.pt'),
                         map_location='cpu', weights_only=False)
data_physical = torch.load(os.path.join(BASE, 'results/features/physical_patch_test_features.pt'),
                            map_location='cpu', weights_only=False)

# 加载模型
model_path = os.path.join(BASE, 'results/models/digital_baseline_v2/fusion_best.pth')
model = FusionClassifierV2(input_dim=698)
model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
model = model.to(device).eval()

def get_scores(data):
    """获取所有样本的预测分数"""
    scores = []
    fused = compute_fused_features(data['spn_feats'], data['dct_feats'], data['clip_feats'], cic, device)
    with torch.no_grad():
        logits = model(fused).squeeze(-1)
        scores = torch.sigmoid(logits).cpu().numpy()
    return scores

print('计算clean和physical patch的预测分数...')
scores_clean = get_scores(data_clean)
scores_physical = get_scores(data_physical)
y_true = np.concatenate([np.zeros(len(scores_clean)), np.ones(len(scores_physical))])
y_score = np.concatenate([scores_clean, scores_physical])

# 计算ROC
fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_score)
auc = sklearn.metrics.roc_auc_score(y_true, y_score)
print(f'AUC = {auc:.4f}')

# 找特定threshold的性能
target_fpr = 0.125  # 12.5% FPR
idx = np.argmin(np.abs(fpr - target_fpr))
print(f'tau={thresholds[idx]:.3f}: FPR={fpr[idx]:.3f}, TPR={tpr[idx]:.3f}')
# 打印tau=0.5和0.7的性能
for t in [0.5, 0.7]:
    idx_t = np.argmin(np.abs(thresholds - t))
    print(f'tau={t:.1f}: FPR={fpr[idx_t]:.3f}, TPR={tpr[idx_t]:.3f}')

# 画图
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

fig, ax = plt.subplots(figsize=(8, 4.5))  # 16:9 宽高比

# 1. 随机猜测线以下区域填充 (浅灰色)
ax.fill_between([0, 1], [0, 1], alpha=0.08, color='#767676', label='_nolegend_')
# 2. ROC曲线以上区域填充 (浅蓝色渐变)
ax.fill_between(fpr, tpr, alpha=0.2, color='#0F4D92', label='_nolegend_')
# 3. ROC曲线
ax.plot(fpr, tpr, color='#0F4D92', lw=3.0, label=f'ForensicGuard (AUC = {auc:.3f})')
# 4. 随机猜测线
ax.plot([0, 1], [0, 1], '--', color='#999999', lw=1.5, label='Random chance')

# 5. 标注τ=0.5的点 (默认阈值)
tau_50 = 0.5
idx50 = np.argmin(np.abs(thresholds - tau_50))
ax.scatter(fpr[idx50], tpr[idx50], color='#2E9E44', s=100, zorder=5, 
           edgecolors='white', linewidth=1.5)
ax.annotate(f'$\\tau=0.5$\n(FPR={fpr[idx50]:.1%}, TPR={tpr[idx50]:.1%})',
            xy=(fpr[idx50], tpr[idx50]), xytext=(fpr[idx50]+0.25, tpr[idx50]-0.08),
            arrowprops=dict(arrowstyle='->', color='#2E9E44', lw=1.5),
            fontsize=10, color='#2E9E44', fontweight='bold')

# 6. 标注τ=0.7的点 (高阈值)
tau_70 = 0.7
idx70 = np.argmin(np.abs(thresholds - tau_70))
ax.scatter(fpr[idx70], tpr[idx70], color='#B64342', s=100, zorder=5,
           edgecolors='white', linewidth=1.5)
ax.annotate(f'$\\tau=0.7$\n(FPR={fpr[idx70]:.1%}, TPR={tpr[idx70]:.1%})',
            xy=(fpr[idx70], tpr[idx70]), xytext=(fpr[idx70]+0.2, tpr[idx70]-0.25),
            arrowprops=dict(arrowstyle='->', color='#B64342', lw=1.5),
            fontsize=10, color='#B64342', fontweight='bold')

ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve — Physical Patch Detection', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'figure6_roc.png'), dpi=300, bbox_inches='tight')
print(f'✅ ROC曲线已保存 (AUC={auc:.3f})')
plt.close(fig)
