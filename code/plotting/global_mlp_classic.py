#!/usr/bin/env python3
"""Global+MLP baseline 在经典篡改数据集上的表现"""
import sys, os, torch, torch.nn as nn
sys.path.insert(0, '/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard/code')
from models.consistency import CrossInstanceConsistencyV2
from models.fusion_classifier import FusionClassifierV2
from scripts.train import compute_fused_features, evaluate

BASE = '/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cic = CrossInstanceConsistencyV2()

def get_global_feats(data):
    """取第一个实例的特征作为全图特征"""
    spn, dct, clip = [], [], []
    for i in range(len(data['labels'])):
        spn.append(data['spn_feats'][i][:1])  # (1, 128)
        dct.append(data['dct_feats'][i][:1])  # (1, 21)
        clip.append(data['clip_feats'][i][:1]) # (1, 512)
    return spn, dct, clip, data['labels']

for ds in ['casia2', 'columbia', 'augmented_coco']:
    print(f'\n=== {ds} (Global+MLP) ===')
    train = torch.load(f'{BASE}/results/features/{ds}_train_features.pt', map_location='cpu', weights_only=False)
    val = torch.load(f'{BASE}/results/features/{ds}_val_features.pt', map_location='cpu', weights_only=False)
    test = torch.load(f'{BASE}/results/features/{ds}_test_features.pt', map_location='cpu', weights_only=False)
    
    s_tr, d_tr, c_tr, l_tr = get_global_feats(train)
    s_val, d_val, c_val, l_val = get_global_feats(val)
    s_te, d_te, c_te, l_te = get_global_feats(test)
    
    print(f'  train: {len(l_tr)}, val: {len(l_val)}, test: {len(l_te)}')
    
    model = FusionClassifierV2(input_dim=698).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    best_acc = 0.0
    for epoch in range(50):
        model.train(); total_loss = 0
        indices = torch.randperm(len(l_tr))
        for start in range(0, len(l_tr), 64):
            end = min(start+64, len(l_tr)); idx = indices[start:end]
            batch_s = [s_tr[i] for i in idx]; batch_d = [d_tr[i] for i in idx]
            batch_c = [c_tr[i] for i in idx]; batch_y = torch.tensor([l_tr[i] for i in idx], dtype=torch.float32, device=device)
            fused = compute_fused_features(batch_s, batch_d, batch_c, cic, device)
            logits = model(fused).squeeze(-1); loss = criterion(logits, batch_y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * len(idx)
        if (epoch+1) % 20 == 0:
            m = evaluate(model, s_val, d_val, c_val, l_val, cic, device)
            if m['accuracy'] > best_acc: best_acc = m['accuracy']
            print(f'  Epoch {epoch+1}/100: loss={total_loss/len(l_tr)*64:.4f}, val_acc={m["accuracy"]*100:.1f}%')
    
    m = evaluate(model, s_te, d_te, c_te, l_te, cic, device)
    print(f'  Test Acc: {m["accuracy"]*100:.2f}%, F1: {m["f1"]*100:.2f}%')
