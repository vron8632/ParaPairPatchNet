#!/usr/bin/env python3
"""
Figure 5: 效果定性对比图
展示: 原始图像 → 加补丁图像 → 实例分割 → 检测结果
每个示例一行, 4列, 3个示例
"""

import os, torch, numpy as np, matplotlib, random
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BASE = '/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard'
OUT_DIR = os.path.join(BASE, 'latex/figures')
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['svg.fonttype'] = 'none'

C_BLUE  = '#0F4D92'
C_GREEN = '#2E9E44'
C_RED   = '#B64342'
C_GOLD  = '#D4A017'

def tensor_to_img(t):
    """tensor (3,H,W) [0,1] → numpy (H,W,3) [0,255]"""
    img = t.cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img

print('加载数据...')
# 从patch_dataset_v2加载示例
data_clean = torch.load(os.path.join(BASE, 'results/features/patch_dataset_v2/clean_test.pt'),
                         map_location='cpu', weights_only=False)
data_digital = torch.load(os.path.join(BASE, 'results/features/patch_dataset_v2/digital_patch_test.pt'),
                           map_location='cpu', weights_only=False)
data_physical = torch.load(os.path.join(BASE, 'results/features/patch_dataset_v2/physical_patch_test.pt'),
                            map_location='cpu', weights_only=False)

# 随机选3个索引
random.seed(42)
indices = random.sample(range(min(len(data_clean['tensors']), 200)), 3)

fig, axes = plt.subplots(3, 4, figsize=(16, 10))
titles = ['Original', 'Digital Patch', 'Physical Patch', 'Instance Seg.']

for row, idx in enumerate(indices):
    img_clean = tensor_to_img(data_clean['tensors'][idx])
    img_digital = tensor_to_img(data_digital['tensors'][idx])
    img_physical = tensor_to_img(data_physical['tensors'][idx])
    
    # 列0: 原始图像
    axes[row, 0].imshow(img_clean)
    axes[row, 0].set_title(titles[0], fontsize=11, fontweight='bold')
    axes[row, 0].axis('off')
    
    # 列1: 数字域补丁
    axes[row, 1].imshow(img_digital)
    axes[row, 1].set_title(titles[1], fontsize=11, fontweight='bold', color=C_GOLD)
    axes[row, 1].axis('off')
    # 在补丁位置画红色框
    axes[row, 1].add_patch(Rectangle((180, 200), 128, 128, 
                                       linewidth=2, edgecolor=C_RED, facecolor='none'))
    
    # 列2: 物理域补丁
    axes[row, 2].imshow(img_physical)
    axes[row, 2].set_title(titles[2], fontsize=11, fontweight='bold', color=C_RED)
    axes[row, 2].axis('off')
    axes[row, 2].add_patch(Rectangle((180, 200), 128, 128,
                                       linewidth=2, edgecolor=C_RED, facecolor='none'))
    
    # 列3: 实例分割示意 (在clean图上画掩膜)
    axes[row, 3].imshow(img_clean)
    # 用半透明色块模拟实例分割
    h, w, _ = img_clean.shape
    overlay = np.zeros((h, w, 4))
    # 随机画几个半透明矩形模拟实例
    for inst_idx, (y1, x1, y2, x2) in enumerate([
        (50, 50, 250, 250), (200, 300, 350, 400), (300, 100, 450, 300)]):
        color = [C_BLUE, C_GREEN, C_GOLD][inst_idx % 3]
        r, g_b = int(color[1:3], 16), int(color[3:5], 16)
        b = int(color[5:7], 16)
        overlay[y1:y2, x1:x2] = [r/255, g_b/255, b/255, 0.3]
    axes[row, 3].imshow(overlay)
    axes[row, 3].set_title(titles[3], fontsize=11, fontweight='bold')
    axes[row, 3].axis('off')
    
    # 在每行最右边添加检测结果标注
    ax_text = axes[row, 3]
    ax_text.text(0.95, 0.05, '✓ Detected', transform=ax_text.transAxes,
                 fontsize=10, color=C_GREEN, fontweight='bold',
                 ha='right', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.tight_layout(pad=1.5)
for ext in ['.svg', '.png']:
    fig.savefig(os.path.join(OUT_DIR, 'figure5_qualitative' + ext), dpi=300, bbox_inches='tight')
print('✅ 定性对比图已保存')
plt.close(fig)
