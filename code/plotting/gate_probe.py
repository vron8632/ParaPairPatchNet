#!/usr/bin/env python3
"""
Gate探针：简化版 — 用已有特征直接训练gate，不依赖场景描述符
"""
import sys, os, torch, torch.nn as nn, numpy as np
sys.path.insert(0, '/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard/code')
from models.consistency import CrossInstanceConsistencyV2
from models.skill_gate import SkillGate

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
        
        # 全局特征 (L2归一化)
        spn_g = torch.nn.functional.normalize(spn.mean(0), p=2, dim=0)
        dct_g = torch.nn.functional.normalize(dct.mean(0), p=2, dim=0)
        clip_g = torch.nn.functional.normalize(clip.mean(0), p=2, dim=0)
        
        # consistency (K>=2时计算)
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

# 2. 定义简化模型：gate + 加权MLP
gate = SkillGate().to(device)
mlp = nn.Sequential(
    nn.Linear(674, 128), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(128, 64), nn.ReLU(),
    nn.Linear(64, 1),
).to(device)

params = list(gate.parameters()) + list(mlp.parameters())
optimizer = torch.optim.Adam(params, lr=5e-4, weight_decay=1e-4)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([len(all_K)/max(sum(1 for k in all_K if k>=2),1)]).to(device))

S = S.to(device); D = D.to(device); C = C.to(device); CONS = CONS.to(device)
labels = labels.float().to(device)

# 3. 训练
print('\n开始训练 (weighted)...')
for epoch in range(200):
    gate.train(); mlp.train()
    optimizer.zero_grad()
    
    # 用简单启发式构造scene descriptor
    K_norm = torch.tensor(K_arr / 10.0, dtype=torch.float32, device=device).unsqueeze(1)
    spn_var = S.var(1, keepdim=True) if S.shape[1] > 1 else torch.zeros(len(K_arr), 1, device=device)
    dct_var = D.var(1, keepdim=True) if D.shape[1] > 1 else torch.zeros(len(K_arr), 1, device=device)
    clip_var = C.var(1, keepdim=True) if C.shape[1] > 1 else torch.zeros(len(K_arr), 1, device=device)
    g = torch.cat([K_norm, spn_var, dct_var, clip_var, 
                   CONS[:, 0:1], CONS[:, 4:5]], dim=1)  # 简化: 6维
    g = torch.nan_to_num(g, nan=0.0)
    
    alpha = gate(g)  # (N, 5)
    
    # 加权融合
    fused = torch.cat([
        alpha[:, 0:1] * S,
        alpha[:, 1:2] * D,
        alpha[:, 2:3] * C,
        alpha[:, 3:4] * CONS,
    ], dim=1)
    
    logits = mlp(fused).squeeze(-1)
    loss = criterion(logits, labels)
        # 额外加consistency skill关闭的正则
        alpha_cons = alpha[:, 4:5]
        k1_mask = (K_all < 2).float().to(device).unsqueeze(1)
        loss += 0.1 * (alpha_cons * k1_mask).mean()
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    optimizer.step()
    
    if (epoch+1) % 20 == 0:
        with torch.no_grad():
            acc = ((torch.sigmoid(logits) > 0.5).float() == labels).float().mean().item()
            a = alpha.mean(0).cpu().numpy()
            a_K1 = alpha[K_arr == 1].mean(0).cpu().numpy() if (K_arr == 1).any() else None
            a_K2 = alpha[K_arr >= 2].mean(0).cpu().numpy() if (K_arr >= 2).any() else None
        print(f'Epoch {epoch+1}/200: loss={loss.item():.4f}, acc={acc:.3f}')
        print(f'  alpha全部: {dict(zip(["wm","spn","dct","clip","cons"], a.round(3).tolist()))}')
        if a_K1 is not None:
            print(f'  alpha K=1:  {dict(zip(["wm","spn","dct","clip","cons"], a_K1.round(3).tolist()))}')
        if a_K2 is not None:
            print(f'  alpha K>=2: {dict(zip(["wm","spn","dct","clip","cons"], a_K2.round(3).tolist()))}')

# 4. 最终分析
gate.eval()
with torch.no_grad():
    alpha_all = gate(g).cpu().numpy()

names = ['watermark', 'spn', 'dct', 'clip', 'consistency']
print(f'\n=== Gate Probe 最终结果 ===')
for i, name in enumerate(names):
    a1 = alpha_all[K_arr == 1, i].mean() if (K_arr == 1).any() else 0
    a2 = alpha_all[K_arr >= 2, i].mean() if (K_arr >= 2).any() else 0
    print(f'  {name:12s}: K=1→{a1:.3f}, K>=2→{a2:.3f}, diff={a2-a1:.3f}')

cons_K1 = alpha_all[K_arr == 1, 4].mean() if (K_arr == 1).any() else 0
cons_K2 = alpha_all[K_arr >= 2, 4].mean() if (K_arr >= 2).any() else 0
print(f'\n关键指标: α_cons K=1={cons_K1:.3f}, K>=2={cons_K2:.3f}')
if cons_K1 < cons_K2 * 0.5:
    print('✅ PASS: gate学到K=1时关闭consistency!')
elif cons_K1 < cons_K2:
    print('⚠️ PARTIAL: gate倾向于K=1时降低consistency，但差异不够大')
else:
    print('❌ FAIL: gate没有学到K=1时关闭consistency')

# 保存
torch.save({'gate': gate.state_dict(), 'mlp': mlp.state_dict()},
           f'{BASE}/results/models/gate_probe/gate_probe_final.pth')
print(f'\n模型已保存')
