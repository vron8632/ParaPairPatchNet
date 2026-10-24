#!/usr/bin/env python3
"""
在已提取的v2特征上用FocalLoss训练CASIA v2
（不重新提取特征）
"""
import sys, os, torch, torch.nn as nn
sys.path.insert(0, '/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard/code')
from models.consistency import CrossInstanceConsistencyV2
from models.fusion_classifier import FusionClassifierV2, FocalLoss
from scripts.train import compute_fused_features, evaluate

BASE = '/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard'
FEATURE_DIR = os.path.join(BASE, 'results/features')
MODEL_DIR = os.path.join(BASE, 'results/models/casia2_v2')
os.makedirs(MODEL_DIR, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cic = CrossInstanceConsistencyV2()

print('加载v2特征...')
data = torch.load(f'{FEATURE_DIR}/casia2_v2_train_features.pt', map_location='cpu', weights_only=False)
s_tr, d_tr, c_tr = data['spn_feats'], data['dct_feats'], data['clip_feats']; l_tr = data['labels']
data = torch.load(f'{FEATURE_DIR}/casia2_v2_val_features.pt', map_location='cpu', weights_only=False)
s_val, d_val, c_val = data['spn_feats'], data['dct_feats'], data['clip_feats']; l_val = data['labels']
data = torch.load(f'{FEATURE_DIR}/casia2_v2_test_features.pt', map_location='cpu', weights_only=False)
s_te, d_te, c_te = data['spn_feats'], data['dct_feats'], data['clip_feats']; l_te = data['labels']
print(f'  train: {len(l_tr)}, val: {len(l_val)}, test: {len(l_te)}')

for criterion_name, criterion in [
    ('FocalLoss(alpha=0.75)', FocalLoss(alpha=0.75, gamma=2.0)),
    ('BCEWithLogits (pos_weight=1.5)', nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5]).to(device))),
    ('BCEWithLogits', nn.BCEWithLogitsLoss()),
]:
    print(f'\n=== 训练: {criterion_name} ===')
    model = FusionClassifierV2(input_dim=698).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    best_acc = 0.0
    for epoch in range(300):
        model.train(); total_loss = 0
        indices = torch.randperm(len(l_tr))
        for start in range(0, len(l_tr), 64):
            end = min(start+64, len(l_tr)); idx = indices[start:end]
            batch_s = [s_tr[i] for i in idx]; batch_d = [d_tr[i] for i in idx]; batch_c = [c_tr[i] for i in idx]
            batch_y = torch.tensor([l_tr[i] for i in idx], dtype=torch.float32, device=device)
            fused = compute_fused_features(batch_s, batch_d, batch_c, cic, device)
            logits = model(fused).squeeze(-1); loss = criterion(logits, batch_y)
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total_loss += loss.item() * len(idx)
        if (epoch+1) % 30 == 0:
            m = evaluate(model, s_val, d_val, c_val, l_val, cic, device)
            if m['accuracy'] > best_acc:
                best_acc = m['accuracy']
                torch.save(model.state_dict(), f'{MODEL_DIR}/{criterion_name.replace("(", "_").replace(")", "").replace(" ", "_")}_best.pth')
            print(f'  Epoch {epoch+1}/300: loss={total_loss/len(l_tr):.4f}, val_acc={m["accuracy"]*100:.1f}%')
    print(f'  最佳验证: {best_acc*100:.1f}%')

    # 测试
    model.load_state_dict(torch.load(f'{MODEL_DIR}/{criterion_name.replace("(", "_").replace(")", "").replace(" ", "_")}_best.pth', map_location='cpu', weights_only=True))
    m = evaluate(model, s_te, d_te, c_te, l_te, cic, device)
    print(f'  Test Acc: {m["accuracy"]*100:.2f}%, F1: {m["f1"]*100:.2f}%')
