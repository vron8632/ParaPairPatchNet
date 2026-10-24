"""
Skill Gate Module G_φ — 轻量二元选择器
根据场景描述符决定4个skill是否参与融合

输入: g ∈ R^6 (场景描述符)
输出: α ∈ [0,1]^5 (5个skill的激活权重)

参数: 6×16+16+16×5+5 = 157 参数 (0.6KB)
"""

import torch
import torch.nn as nn
import numpy as np


class SkillGate(nn.Module):
    """
    轻量skill门控模块 G_φ
    输入: g ∈ R^6 (场景描述符)
    输出: α ∈ [0,1]^5 (sigmoid后的门控权重)
    
    输入特征:
    - K_norm: 实例数/10 (归一化)
    - patch_score_mean: YOLO检测器平均目标分数
    - patch_score_max: YOLO检测器最大目标分数
    - img_var: 图像方差 (低=纯色背景)
    - spn_std_norm: SPN信号强度归一化
    - dct_energy_norm: DCT中频能量归一化
    """
    
    def __init__(self):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 5),
            nn.Sigmoid(),
        )
        
        # 初始化: consistency skill在K=1时倾向于低激活
        with torch.no_grad():
            self.gate[-2].bias[4] = -2.0  # 初始bias让consistency倾向关闭
    
    def forward(self, g):
        """
        输入: g ∈ R^6
        输出: α ∈ [0,1]^5
        """
        return self.gate(g)
    
    def get_skill_weights(self, g):
        """返回5个skill的权重字典"""
        alpha = self.forward(g)
        names = ['watermark', 'spn', 'dct', 'clip', 'consistency']
        return {name: alpha[:, i].item() if alpha.dim() > 1 else alpha[i].item() 
                for i, name in enumerate(names)}


def compute_scene_descriptor(spn_feats_list, dct_feats_list, clip_feats_list,
                             consistency_vec, img_tensor, K):
    """
    计算场景描述符 g ∈ R^6
    
    参数:
      spn_feats_list: list of per-instance SPN特征 (K×128)
      dct_feats_list: list of per-instance DCT特征 (K×21)  
      clip_feats_list: list of per-instance CLIP特征 (K×512)
      consistency_vec: 13-d consistency向量
      img_tensor: 原始图像 (3,H,W)
      K: 实例数量
    """
    # 1. K_norm: 归一化实例数
    K_norm = K / 10.0
    
    # 2. patch_score: 用全局特征的一致性作为代理
    if len(spn_feats_list) > 0:
        spn_stack = torch.stack(spn_feats_list)  # (K, 128)
        dct_stack = torch.stack(dct_feats_list)  # (K, 21)
        clip_stack = torch.stack(clip_feats_list) # (K, 512)
        
        # 计算实例间差异作为patch-like信号
        spn_mean = spn_stack.mean(0)
        dct_mean = dct_stack.mean(0)
        clip_mean = clip_stack.mean(0)
        
        # patch_score: 实例间特征的方差作为"不一致性信号"
        spn_var = spn_stack.var(0).mean().item()
        dct_var = dct_stack.var(0).mean().item()
        clip_var = clip_stack.var(0).mean().item()
        patch_score_mean = (spn_var + dct_var + clip_var) / 3.0
        patch_score_max = max(spn_var, dct_var, clip_var)
    else:
        patch_score_mean = 0.0
        patch_score_max = 0.0
    
    # 3. img_var: 图像方差
    img_var = img_tensor.var().item() if img_tensor is not None else 0.5
    
    # 4. spn_std_norm: SPN信号强度
    spn_std_norm = spn_var / 0.01  # 归一化到合理范围
    
    # 5. dct_energy_norm: DCT能量
    dct_energy_norm = dct_var / 0.1  # 归一化到合理范围
    
    return torch.tensor([K_norm, patch_score_mean, patch_score_max,
                         img_var, spn_std_norm, dct_energy_norm],
                        dtype=torch.float32)


# ============================================================
# 探针: 离线训练gate看α分布
# ============================================================
def run_gate_probe(feature_dir, output_dir, epochs=100, lr=1e-3):
    """
    离线探针: 冻结所有特征提取器，只训gate和MLP
    验证K=1时α_cons是否自然为0
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 加载所有数据
    from models.consistency import CrossInstanceConsistencyV2
    cic = CrossInstanceConsistencyV2()
    
    all_g, all_labels, all_K = [], [], []
    all_s, all_d, all_c, all_cons = [], [], [], []
    
    for domain in ['clean', 'digital_patch', 'physical_patch']:
        data = torch.load(f'{feature_dir}/{domain}_test_features.pt', 
                         map_location='cpu', weights_only=False)
        for i in range(len(data['labels'])):
            spn = data['spn_feats'][i]
            dct = data['dct_feats'][i]
            clip = data['clip_feats'][i]
            K = spn.shape[0]
            
            # 计算consistency
            if K >= 2:
                stats, _, _, _ = cic.compute_consistency(
                    spn.to(device), dct.to(device), clip.to(device))
                cons = stats
            else:
                cons = torch.zeros(13)
            
            # 计算全局特征
            global_f = cic.compute_global_features(spn.to(device), dct.to(device), clip.to(device))
            
            # 场景描述符
            g = compute_scene_descriptor(
                [spn[j] for j in range(K)],
                [dct[j] for j in range(K)],
                [clip[j] for j in range(K)],
                cons, None, K)
            
            all_g.append(g)
            all_labels.append(1 if domain != 'clean' else 0)
            all_K.append(K)
            all_s.append(spn.mean(0))
            all_d.append(dct.mean(0))
            all_c.append(clip.mean(0))
            all_cons.append(cons)
    
    G = torch.stack(all_g)
    L = torch.tensor(all_labels)
    K_all = np.array(all_K)
    S = torch.stack(all_s)
    D = torch.stack(all_d)
    C = torch.stack(all_c)
    CONS = torch.stack(all_cons)
    
    print(f'数据: {len(L)}样本, K分布: {dict(zip(*np.unique(K_all, return_counts=True)))}')
    
    # L2归一化特征
    S = torch.nn.functional.normalize(S, p=2, dim=1)
    D = torch.nn.functional.normalize(D, p=2, dim=1)
    C = torch.nn.functional.normalize(C, p=2, dim=1)
    CONS = torch.nn.functional.normalize(CONS, p=2, dim=1)
    
    # 训练gate
    gate = SkillGate().to(device)
    fusion = nn.Sequential(
        nn.LayerNorm(674),
        nn.Linear(674, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(256, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 1),
    ).to(device)
    
    optimizer = torch.optim.Adam(list(gate.parameters()) + list(fusion.parameters()), 
                                 lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    G = G.to(device)
    S = S.to(device)
    D = D.to(device)
    C = C.to(device)
    CONS = CONS.to(device)
    
    # 训练
    for epoch in range(epochs):
        gate.train(); fusion.train()
        optimizer.zero_grad()
        
        # gate输出
        alpha = gate(G)  # (N, 5)
        
        # 加权特征
        fused = torch.cat([
            alpha[:, 0:1] * S,
            alpha[:, 1:2] * D,
            alpha[:, 2:3] * C,
            alpha[:, 3:4] * CONS,
        ], dim=1)
        
        # 简单MLP分类
        logits = fusion(fused).squeeze(-1)
        loss = criterion(logits, L.float().to(device))
        
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 20 == 0:
            with torch.no_grad():
                acc = ((torch.sigmoid(logits) > 0.5).float() == L.float().to(device)).float().mean().item()
                alpha_mean = alpha.mean(0).cpu().numpy()
                alpha_K1 = alpha[K_all == 1].mean(0).cpu().numpy() if (K_all == 1).any() else None
                alpha_K2 = alpha[K_all >= 2].mean(0).cpu().numpy() if (K_all >= 2).any() else None
            print(f'Epoch {epoch+1}/{epochs}: loss={loss.item():.4f}, acc={acc:.3f}')
            print(f'  alpha全部: {dict(zip(["wm","spn","dct","clip","cons"], alpha_mean.round(3).tolist()))}')
            if alpha_K1 is not None:
                print(f'  alpha K=1:  {dict(zip(["wm","spn","dct","clip","cons"], alpha_K1.round(3).tolist()))}')
            if alpha_K2 is not None:
                print(f'  alpha K>=2: {dict(zip(["wm","spn","dct","clip","cons"], alpha_K2.round(3).tolist()))}')
    
    # 保存结果
    torch.save({'gate': gate.state_dict(), 'fusion': fusion.state_dict()},
               f'{output_dir}/gate_probe.pth')
    
    # 统计alpha分布
    gate.eval()
    with torch.no_grad():
        alpha_all = gate(G).cpu().numpy()
    
    names = ['watermark', 'spn', 'dct', 'clip', 'consistency']
    print(f'\n=== Gate Probe 最终结果 ===')
    for name, i in zip(names, range(5)):
        a1 = alpha_all[K_all == 1, i].mean()
        a2 = alpha_all[K_all >= 2, i].mean() if (K_all >= 2).any() else 0
        print(f'  {name:12s}: K=1→{a1:.3f}, K>=2→{a2:.3f}, diff={a2-a1:.3f}')
    
    # 关键指标: consistency在K=1时是否低
    cons_K1 = alpha_all[K_all == 1, 4].mean() if (K_all == 1).any() else 0
    cons_K2 = alpha_all[K_all >= 2, 4].mean() if (K_all >= 2).any() else 0
    print(f'\n关键指标: α_cons K=1={cons_K1:.3f}, K>=2={cons_K2:.3f}')
    if cons_K1 < cons_K2 * 0.5:
        print('✅ PASS: gate学到K=1时关闭consistency!')
    else:
        print('❌ FAIL: gate没有学到K=1时关闭consistency')
    
    return gate, alpha_all, K_all


if __name__ == '__main__':
    BASE = '/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard'
    gate, alpha, K_all = run_gate_probe(
        f'{BASE}/results/features',
        f'{BASE}/results/models/gate_probe',
        epochs=200, lr=1e-3)
