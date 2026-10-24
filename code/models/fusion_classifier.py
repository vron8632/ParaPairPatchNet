"""
MLP融合分类器 v2 — 支持增强特征输入

v2 更新:
- 输入维度: 697 (13跨实例一致性 + 24子区域一致性 + 660全局特征)
- 可选: 类别加权loss (解决clean/patch不平衡)
- 可选: focal loss (难例挖掘)
- v1 接口完全兼容 (input_dim=674仍可用)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionClassifierV2(nn.Module):
    """
    增强版MLP融合分类器 v2

    输入: 697-dim (cross_instance 13 + subregion 24 + global 660)
    输出: p ∈ [0,1] 篡改/补丁概率

    架构: 697 → 256 → 128 → 64 → 1
    v1相比: 更宽的第一层, 加入LayerNorm和更激进的dropout
    """

    def __init__(self, input_dim=698, hidden1=256, hidden2=128, 
                 hidden3=64, dropout=0.4):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.75),
            nn.Linear(hidden2, hidden3),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden3, 1),
        )

        # 不确定性估计头
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden3, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

    def predict_proba(self, x):
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def predict(self, x, threshold=0.5):
        proba = self.predict_proba(x)
        return (proba > threshold).float()


class FocalLoss(nn.Module):
    """Focal Loss: 解决类别不平衡"""
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probas = torch.sigmoid(logits)
        p_t = probas * targets + (1 - probas) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = focal_weight * alpha_weight * bce_loss
        return loss.mean()


# 兼容旧接口
FusionClassifier = FusionClassifierV2


class ForensicFeaturePipelineV2:
    """
    完整的v2取证特征提取 + 一致性分析 + 分类 Pipeline
    将各模块串联起来
    """

    def __init__(self, segmenter, spn_extractor, dct_extractor,
                 clip_extractor, consistency, classifier, device='cuda'):
        self.segmenter = segmenter
        self.spn_extractor = spn_extractor
        self.dct_extractor = dct_extractor
        self.clip_extractor = clip_extractor
        self.consistency = consistency
        self.classifier = classifier
        self.device = device

    @torch.no_grad()
    def process_image(self, image_tensor):
        """
        处理单张图像 (v2: 包含子区域一致性分析)
        """
        # 1. 实例分割
        masks_list = self.segmenter.segment(image_tensor)
        H, W = image_tensor.shape[-2:]
        min_area = 32 * 32
        masks_list = [(m, b) for m, b in masks_list if m.sum() > min_area]

        if len(masks_list) < 1:
            # 没有实例, 使用整图特征
            spn_f = self.spn_extractor.extract(image_tensor).unsqueeze(0)
            dct_f = self.dct_extractor.extract_profile(image_tensor).unsqueeze(0)
            clip_f = self.clip_extractor.encode_image(image_tensor).unsqueeze(0)
            all_spn, all_dct, all_clip = [spn_f], [dct_f], [clip_f]
        else:
            all_spn, all_dct, all_clip = [], [], []
            for mask, bbox in masks_list:
                mask_t = torch.from_numpy(mask).float().to(self.device)
                spn_f = self.spn_extractor.extract_instance(image_tensor.to(self.device), mask_t)
                dct_f = self.dct_extractor.extract_profile(image_tensor, mask_t.cpu())
                clip_f = self.clip_extractor.encode_image(image_tensor, mask)
                all_spn.append(spn_f)
                all_dct.append(dct_f)
                all_clip.append(clip_f)

        # 2. 跨实例一致性 (v1)
        cross_stats, _, _, _ = self.consistency.compute_consistency(all_spn, all_dct, all_clip)

        # 3. 实例子区域一致性 (v2)
        subregion_stats = self.consistency.compute_subregion_stats(
            image_tensor, masks_list, self.spn_extractor, 
            self.dct_extractor, self.clip_extractor)

        # 4. 全局平均特征
        global_f = self.consistency.compute_global_features(all_spn, all_dct, all_clip)

        # 5. v2融合
        fused = self.consistency.fuse_features_v2(cross_stats, subregion_stats, global_f)

        # 6. 分类
        fused = fused.unsqueeze(0).to(self.device)
        proba = self.classifier.predict_proba(fused)
        pred = (proba > 0.5).long()

        return pred.item(), proba.item(), fused
