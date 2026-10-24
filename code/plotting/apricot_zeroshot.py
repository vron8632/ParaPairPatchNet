#!/usr/bin/env python3
"""APRICOT零样本测试 — 从zip直接读取"""
import sys, os, torch, numpy as np, zipfile, json
sys.path.insert(0, '/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard/code')
from models.consistency import CrossInstanceConsistencyV2
from models.fusion_classifier import FusionClassifierV2
from scripts.train import compute_fused_features, evaluate
from features.segmenter import InstanceSegmenter
from features.spn_extractor import get_spn_extractor
from features.dct_profile import DCTProfileExtractor
from features.clip_encoder import get_clip_extractor
from torchvision import transforms
from PIL import Image
import numpy as np

BASE = '/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
zip_path = os.path.join(BASE, 'data/apricot/dev.zip')

# 加载模型
cic = CrossInstanceConsistencyV2()
model = FusionClassifierV2(input_dim=698)
model.load_state_dict(torch.load(os.path.join(BASE, 'results/models/digital_baseline_v2/fusion_best.pth'),
                                   map_location='cpu', weights_only=True))
model = model.to(device).eval()

seg = InstanceSegmenter(device=device)
spn = get_spn_extractor(mode='learned', device=device)
dct = DCTProfileExtractor()
clip_img = get_clip_extractor(device=device)

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# 从zip读取并处理
results = []
total = 0
with zipfile.ZipFile(zip_path, 'r') as z:
    pt_files = sorted([f for f in z.namelist() if f.endswith('.pt')])
    print(f'总文件数: {len(pt_files)}')
    for idx, pt_name in enumerate(pt_files):
        with z.open(pt_name) as f:
            data = torch.load(f, map_location='cpu', weights_only=False)
        img_np = data['Image'][0]  # (4608, 2304, 3)
        mask_np = data['Mask']     # (4608, 2304) bool
        # 缩放
        img_pil = Image.fromarray(img_np.astype(np.uint8))
        img_512 = transform(img_pil)
        img_orig = img_512 * std + mean  # 反归一化到[0,1]
        # 实例分割
        masks_list = seg.segment(img_orig.to(device))
        masks_list = [(m, b) for m, b in masks_list if m.sum() > 64*64]
        # 特征提取
        K = len(masks_list)
        if K < 1:
            spn_f = spn.extract(img_orig.to(device))
            dct_f = dct.extract_profile(img_orig.cpu())
            clip_f = clip_img.encode_image(img_orig.cpu())
            all_s, all_d, all_c = [spn_f], [dct_f], [clip_f]
        else:
            all_s, all_d, all_c = [], [], []
            for mask, _ in masks_list:
                mask_t = torch.from_numpy(mask).float().to(device)
                all_s.append(spn.extract_instance(img_orig.to(device), mask_t))
                all_d.append(dct.extract_profile(img_orig.cpu(), mask))
                all_c.append(clip_img.encode_image(img_orig.cpu(), mask))
        # 所有特征统一到同一设备
        all_s = [f.to(device) if hasattr(f, 'to') else f for f in all_s]
        all_d = [f.to(device) if hasattr(f, 'to') else f for f in all_d]
        all_c = [f.to(device) if hasattr(f, 'to') else f for f in all_c]
        # 预测
        stats, _, _, _ = cic.compute_consistency(all_s, all_d, all_c)
        gf = cic.compute_global_features(all_s, all_d, all_c).to(device)
        fused = torch.cat([stats.to(device), torch.zeros(24).to(device), gf.to(device)]).unsqueeze(0)
        prob = torch.sigmoid(model(fused)).item()
        # 检查mask中是否有补丁(>1%像素)
        has_patch = (mask_np > 0).sum() > (4608 * 2304 * 0.01)
        results.append({'idx': idx, 'prob': prob, 'has_patch': bool(has_patch), 'K': K})
        total += 1
        print(f'  [{idx+1}/{len(pt_files)}] prob={prob:.3f}, patch={has_patch}, K={K}')

# 评估
if total > 0:
    preds = [1 if r['prob'] > 0.5 else 0 for r in results]
    labels = [1 if r['has_patch'] else 0 for r in results]
    acc = sum(p == l for p, l in zip(preds, labels)) / len(results)
    tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)
    print(f'\n=== APRICOT零样本测试结果 ===')
    print(f'样本数: {total}')
    print(f'准确率: {acc*100:.1f}%')
    print(f'TP={tp} FP={fp} FN={fn}')
    print(f'有补丁图像: {sum(labels)}/{total}')
