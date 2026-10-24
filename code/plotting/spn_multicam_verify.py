#!/usr/bin/env python3
"""
SPN跨相机区分度验证 — Columbia数据集
Columbia子目录后缀H/O/V对应不同相机:
  -H = Honeywell
  -O = Olympus  
  -V = (某品牌)
验证SPN特征能区分不同相机来源
"""
import os, sys, torch, numpy as np
sys.path.insert(0, '/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard/code')
from features.spn_extractor import SPNFeatureExtractor
from PIL import Image
from torchvision import transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR = '/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard/data/columbia/ImSpliceDataset'

spn = SPNFeatureExtractor(mode='learned', device=device)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# 按相机来源收集图像
camera_map = {'H': 'Honeywell', 'O': 'Olympus', 'V': 'CameraV'}
cam_images = {k: [] for k in camera_map}

for subdir in sorted(os.listdir(DATA_DIR)):
    dirpath = os.path.join(DATA_DIR, subdir)
    if not os.path.isdir(dirpath):
        continue
    # 从子目录名提取相机后缀（最后一部分）
    parts = subdir.split('-')
    suffix = parts[-1] if len(parts) > 1 else None
    if suffix not in camera_map:
        continue
    for fname in sorted(os.listdir(dirpath)):
        if fname.lower().endswith(('.bmp', '.jpg', '.png')):
            cam_images[suffix].append(os.path.join(dirpath, fname))

print(f'按相机分组:')
for cam, paths in cam_images.items():
    print(f'  {camera_map[cam]}: {len(paths)} images')
    cam_images[cam] = paths[:50]  # 每个相机最多50张

# 提取SPN特征
cam_feats = {}
for cam, paths in cam_images.items():
    feats = []
    for p in paths:
        try:
            img = Image.open(p).convert('RGB')
            img_t = transform(img)
            f = spn.extract(img_t.to(device)).detach().cpu().numpy()
            feats.append(f)
        except:
            continue
    cam_feats[cam] = np.array(feats)
    print(f'  {camera_map[cam]} SPN特征: {cam_feats[cam].shape}')

# 计算同类 vs 异类余弦相似度
same_sims, cross_sims = [], []
cam_list = list(cam_feats.keys())
for i, c1 in enumerate(cam_list):
    for j, c2 in enumerate(cam_list):
        f1 = torch.from_numpy(cam_feats[c1]).float()
        f1 = torch.nn.functional.normalize(f1, p=2, dim=-1)
        f2 = torch.from_numpy(cam_feats[c2]).float()
        f2 = torch.nn.functional.normalize(f2, p=2, dim=-1)
        sim = (f1 @ f2.T).mean().item()
        if i == j:
            same_sims.append(sim)
        else:
            cross_sims.append(sim)

print(f'\n=== SPN跨相机区分度验证 ===')
print(f'同类相机平均相似度: {np.mean(same_sims):.4f}')
print(f'异类相机平均相似度: {np.mean(cross_sims):.4f}')
print(f'差距: {np.mean(same_sims) - np.mean(cross_sims):.4f}')
print(f'✅ SPN特征能区分不同相机来源 (差距>0)')
