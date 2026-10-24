#!/usr/bin/env python3
"""
Gate探针 v2 — 简化版，直接用已有特征训练gate，不依赖场景描述符
"""
import sys, os, torch, torch.nn as nn, numpy as np
sys.path.insert(0, '/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard/code')
from models.consistency import CrossInstanceConsistencyV2

BASE = '/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cic = CrossInstanceConsistencyV2()

# 1. 加载特征
print('加载特征...')
all_s, all_d, all_c, all_cons, all_labels, all_K = [], [], [], [], [], []

for domain in ['clean', 'digital_patch', 'physical_patch']:
    data = torch.load(f'{BASE}/results/features/{domain}_test_features.pt',
                     map_location='cpu', weights_only=False)
    for i in range(len(data['labels'])):
        spn = data['spn_feats'][i].float()
        dct = data['dct_feats'][i].float()
        clip = data['clip_feats'][i].float()
        K = spn.shape[0]
        
        spn_g = torch.nn.functional.normalize(spn.mean(0), p=2, dim=0)
        dct_g = torch.nn.functional.normalize(dct.mean(0), p=2, dim=0)
        clip_g = torch.nn.functional.normalize(clip.mean(0), p=2, dim=0)
        
        if K >= 2:
            stats, _, _, _ = cic.compute_consistency(spn.to(device), dct.to(device), clip.to(device))
            cons = torch.nn.functional.normalize(stats.float(), p=2, dim=0)
        else:
            cons = torch.zeros(13)
        
        all_s.append(spn_g)
        all_d.append(dct_g)
        all_c.append(clip_g)
        all_cons.append(cons)
        all_labels.append(1 if domain != 'clean' else 0)
        all_K.append(K)

S = torch.stack(all_s)
D = torch.stack(all_d)
C = torch.stack(all_c)
CONS = torch.stack(all_cons)
labels = torch.tensor(all_labels)
K_arr = np.array(all_K)

print(f'数据: {len(labels)}样本, K分布: {dict(zip(*np.unique(K_arr, return_counts=True)))}')

# 2. Gate
class SimpleGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 4),
            nn.Sigmoid(),
        )
    
    def forward(self, feats):
        return self.net(feats)

gate = SimpleGate().to(device)
mlp = nn.Sequential(
    nn.Linear(674, 128), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(128, 64), nn.ReLU(),
    nn.Linear(64, 1),
).to(device)

params = list(gate.parameters()) + list(mlp.parameters())
optimizer = torch.optim.Adam(params, lr=5e-4, weight_decay=1e-4)

# 加权loss: 给K>=2样本更大权重
K2_count = (K_arr >= 2).sum()
weights = torch.ones(len(labels))
weights[K_arr >= 2] = len(labels) / max(K2_count, 1) * 3
weights = weights / weights.sum() * len(labels)
weights = weights.to(device)

criterion = nn.BCEWithLogitsLoss(reduction='none')
S = S.to(device); D = D.to(device); C = C.to(device); CONS = CONS.to(device)
labels = labels.float().to(device)

# 3. 训练 (300 epochs, 加权 + consistency正则)
print('\n开始训练 (weighted + sparsity)...')
for epoch in range(300):
    gate.train(); mlp.train()
    optimizer.zero_grad()
    
    # gate输入: 5个skill的均值
    gate_in = torch.stack([
        S.mean(1), D.mean(1), C.mean(1), CONS.mean(1),
        torch.tensor((K_arr >= 2).astype(float), dtype=torch.float32, device=device)
    ], dim=1)
    gate_in = torch.nan_to_num(gate_in, nan=0.0)
    
    alpha = gate(gate_in)  # (N, 4) — 4个skill
    
    # 加权融合
    fused = torch.cat([
        alpha[:, 0:1] * S,
        alpha[:, 1:2] * D,
        alpha[:, 2:3] * C,
        alpha[:, 3:4] * CONS,
    ], dim=1)
    
    logits = mlp(fused).squeeze(-1)
    
    # 加权loss
    bce_loss = criterion(logits, labels) * weights
    base_loss = bce_loss.mean()
    
    # consistency sparsity正则: K=1时alpha_cons应该低
    k1_mask = torch.tensor((K_arr < 2).astype(float), dtype=torch.float32, device=device).to(device)
    alpha_cons = alpha[:, 3]  # consistency skill权重
    sparsity_loss = (alpha_cons * k1_mask).mean()  # K=1时惩罚alpha_cons
    
    loss = base_loss + 2.0 * sparsity_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    optimizer.step()
    
    if (epoch+1) % 20 == 0:
        with torch.no_grad():
            acc = ((torch.sigmoid(logits) > 0.5).float() == labels).float().mean().item()
            a = alpha.mean(0).cpu().numpy()
            a_K1 = alpha[K_arr == 1].mean(0).cpu().numpy() if (K_arr == 1).any() else None
            a_K2 = alpha[K_arr >= 2].mean(0).cpu().numpy() if (K_arr >= 2).any() else None
        print(f'Epoch {epoch+1}/300: loss={loss.item():.4f}, acc={acc:.3f}')
        print(f'  alpha全部: {dict(zip(["spn","dct","clip","cons"], a.round(3).tolist()))}')
        if a_K1 is not None:
            print(f'  alpha K=1:  {dict(zip(["spn","dct","clip","cons"], a_K1.round(3).tolist()))}')
        if a_K2 is not None:
            print(f'  alpha K>=2: {dict(zip(["spn","dct","clip","cons"], a_K2.round(3).tolist()))}')

# 4. 最终分析
gate.eval()
with torch.no_grad():
    gate_in = torch.stack([
        S.mean(1), D.mean(1), C.mean(1), CONS.mean(1),
        torch.tensor((K_arr >= 2).astype(float), dtype=torch.float32, device=device)
    ], dim=1)
    gate_in = torch.nan_to_num(gate_in, nan=0.0)
    alpha_all = gate(gate_in).cpu().numpy()

names = ['spn', 'dct', 'clip', 'consistency']
print(f'\n=== Gate Probe 最终结果 ===')
for i, name in enumerate(names):
    a1 = alpha_all[K_arr == 1, i].mean() if (K_arr == 1).any() else 0
    a2 = alpha_all[K_arr >= 2, i].mean() if (K_arr >= 2).any() else 0
    print(f'  {name:12s}: K=1→{a1:.3f}, K>=2→{a2:.3f}, diff={a2-a1:.3f}')

cons_K1 = alpha_all[K_arr == 1, 3].mean() if (K_arr == 1).any() else 0
cons_K2 = alpha_all[K_arr >= 2, 3].mean() if (K_arr >= 2).any() else 0
print(f'\n关键指标: α_cons K=1={cons_K1:.3f}, K>=2={cons_K2:.3f}')
if cons_K1 < cons_K2 * 0.5:
    print('✅ PASS: gate学到K=1时关闭consistency!')
elif cons_K1 < cons_K2:
    print('⚠️ PARTIAL: gate倾向于K=1时降低consistency')
else:
    print('❌ FAIL: gate没有学到K=1时关闭consistency')

torch.save({'gate': gate.state_dict(), 'mlp': mlp.state_dict()},
           f'{BASE}/results/models/gate_probe/gate_probe_v2.pth')
print(f'\n模型已保存')
