"""
跨实例一致性分析模块 v2 — 新增实例子区域一致性分析

v2 新增功能:
1. Intra-Instance Sub-Region Consistency: 将每个实例分为N×N子网格，
   分析子网格间取证特征的一致性。补丁所在的子网格会表现为离群值。
2. Patch-Instance Difference: 计算补丁局部特征 vs 宿主实例特征的差异向量
3. 兼容v1接口 (CrossInstanceConsistency)

核心假设:
- 自然图像中所有实例的取证特征应高度一致
- 每个实例内部的子区域也应取证一致
- 物理补丁作为一个外来对象，会在实例内部产生局部取证不一致
"""

import torch
import torch.nn.functional as F
import numpy as np
import math


class SubRegionConsistency:
    """
    实例子区域一致性分析

    将每个语义实例划分为 N×N 子网格，提取每个子网格的取证特征，
    分析子网格间的特征一致性。补丁区域表现为离群值。

    输出统计量:
    - intra_instance_std: 各实例内部子网格特征的标准差 (均值越高=越不一致)
    - max_deviation: 每个实例中偏离最大的子网格的偏差值
    - patch_likelihood: 基于子网格一致性的补丁可能性评分
    """

    def __init__(self, grid_size=4):
        """
        grid_size: 将实例划分为 grid_size × grid_size 个子网格
        """
        self.grid_size = grid_size

    def compute_subregion_consistency(self, image_tensor, masks_list,
                                       spn_extractor, dct_extractor, clip_extractor):
        """
        计算实例子区域一致性

        输入:
          image_tensor: (3, H, W) 原始图像
          masks_list: [(mask, bbox), ...] 实例分割结果
          spn_extractor: SPN特征提取器
          dct_extractor: DCT特征提取器
          clip_extractor: CLIP特征提取器

        输出:
          subregion_stats: (24,) tensor - 子区域一致性统计量
            格式: [每实例的SPN子区域std (K个), 每实例的DCT子区域std (K个),
                   每实例的CLIP子区域std (K个), 每实例的max偏差 (K个),
                   实例数K, 含补丁实例数, 总体不一致性得分]
            填充到固定维度 24
        """
        K = len(masks_list)
        if K < 1:
            return torch.zeros(24)

        device = image_tensor.device
        all_intra_std = {'spn': [], 'dct': [], 'clip': []}
        all_max_dev = []
        instance_patch_likelihood = []

        for mask, bbox in masks_list:
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).float().to(device)

            # 找到实例的有效边界框
            indices = torch.where(mask > 0.5)
            if len(indices[0]) < 64:
                continue  # 实例太小，跳过

            y_min, y_max = indices[0].min().item(), indices[0].max().item()
            x_min, x_max = indices[1].min().item(), indices[1].max().item()
            h, w = y_max - y_min, x_max - x_min

            if h < self.grid_size or w < self.grid_size:
                continue

            # 将实例划分为 grid_size × grid_size 子网格
            sub_features = {'spn': [], 'dct': [], 'clip': []}

            for gy in range(self.grid_size):
                for gx in range(self.grid_size):
                    sy = y_min + int(gy * h / self.grid_size)
                    ey = y_min + int((gy + 1) * h / self.grid_size)
                    sx = x_min + int(gx * w / self.grid_size)
                    ex = x_min + int((gx + 1) * w / self.grid_size)

                    # 子网格区域掩膜
                    sub_mask = torch.zeros_like(mask)
                    sub_mask[sy:ey, sx:ex] = mask[sy:ey, sx:ex]

                    if sub_mask.sum() < 32:
                        continue

                    # 提取子网格的取证特征
                    img_cropped = image_tensor * sub_mask.unsqueeze(0).to(device)

                    try:
                        spn_f = spn_extractor.extract(img_cropped)
                        dct_f = dct_extractor.extract_profile(img_cropped.cpu())
                        clip_f = clip_extractor.encode_image(img_cropped.cpu(), 
                                                             sub_mask.cpu().numpy())
                        sub_features['spn'].append(spn_f)
                        sub_features['dct'].append(dct_f)
                        sub_features['clip'].append(clip_f)
                    except Exception:
                        continue

            # 计算子网格间的特征一致性 (标准差)
            for feat_name in ['spn', 'dct', 'clip']:
                if len(sub_features[feat_name]) >= 2:
                    feats = torch.stack(sub_features[feat_name])  # (N, D)
                    feats = F.normalize(feats, p=2, dim=-1)
                    # 计算 pairwise 距离
                    sim_matrix = feats @ feats.T  # (N, N)
                    triu = torch.triu_indices(len(feats), len(feats), offset=1)
                    vals = sim_matrix[triu[0], triu[1]]
                    intra_std = vals.std().item() if len(vals) > 1 else 0.0
                    all_intra_std[feat_name].append(intra_std)
                else:
                    all_intra_std[feat_name].append(0.0)

            # 计算每个子网格与实例平均特征的偏差
            if len(sub_features['spn']) >= 3:
                spn_stack = F.normalize(torch.stack(sub_features['spn']), p=2, dim=-1)
                avg_feat = spn_stack.mean(0, keepdim=True)
                deviations = (1 - (spn_stack @ avg_feat.T).squeeze(-1)).mean().item()
                all_max_dev.append(deviations)

        # 构建输出统计量
        K_actual = len(all_intra_std['spn'])
        if K_actual == 0:
            return torch.zeros(24)

        # 填充到固定维度
        stats = []
        for feat_name in ['spn', 'dct', 'clip']:
            vals = all_intra_std[feat_name][:8]  # 最多8个实例
            while len(vals) < 8:
                vals.append(0.0)
            stats.extend(vals)

        max_dev = all_max_dev[:8]
        while len(max_dev) < 8:
            max_dev.append(0.0)
        stats.extend(max_dev)

        # 全局统计量
        overall_inconsistency = np.mean(all_intra_std['spn']) if all_intra_std['spn'] else 0.0
        stats.extend([float(K_actual), 
                     float(sum(1 for d in all_max_dev if d > 0.15)),
                     float(overall_inconsistency)])

        return torch.tensor(stats[:24])


class CrossInstanceConsistencyV2:
    """
    增强版跨实例一致性分析 v2

    融合:
    1. v1: 跨实例一致性 (12个统计量 + K)
    2. v2: 实例子区域一致性 (24个统计量)
    3. v2: 如果已知补丁位置, 补丁-实例差异特征

    输出: (13 + 24 = 37) 维特征向量
    """

    def __init__(self, subregion_consistency=None):
        self.subregion = subregion_consistency or SubRegionConsistency(grid_size=4)

    def compute_consistency(self, spn_feats, dct_feats, clip_feats):
        """v1兼容: 跨实例一致性 (原始12+1维)"""
        # 复用v1逻辑
        if isinstance(spn_feats, list):
            spn_feats = torch.stack(spn_feats) if spn_feats else torch.zeros(0, 128)
        if isinstance(dct_feats, list):
            dct_feats = torch.stack(dct_feats) if dct_feats else torch.zeros(0, 21)
        if isinstance(clip_feats, list):
            clip_feats = torch.stack(clip_feats) if clip_feats else torch.zeros(0, 512)

        K = spn_feats.shape[0]

        if K < 2:
            stats = torch.tensor([0.0] * 12 + [float(K)])
            return stats, None, None, None

        spn_feats = F.normalize(spn_feats, p=2, dim=-1)
        dct_feats = F.normalize(dct_feats, p=2, dim=-1)
        clip_feats = F.normalize(clip_feats, p=2, dim=-1)

        S_spn = spn_feats @ spn_feats.T
        S_dct = dct_feats @ dct_feats.T
        S_clip = clip_feats @ clip_feats.T

        triu_indices = torch.triu_indices(K, K, offset=1)
        
        def _stats(vals):
            if len(vals) == 0:
                return [0.0, 0.0, 0.0, 0.0]
            return [vals.mean().item(), vals.std().item() if len(vals) > 1 else 0.0,
                    vals.min().item(), vals.max().item()]

        stats = torch.tensor(
            _stats(S_spn[triu_indices[0], triu_indices[1]]) +
            _stats(S_dct[triu_indices[0], triu_indices[1]]) +
            _stats(S_clip[triu_indices[0], triu_indices[1]]) +
            [float(K)]
        )
        return stats, S_spn, S_dct, S_clip

    def compute_subregion_stats(self, image_tensor, masks_list,
                                 spn_extractor, dct_extractor, clip_extractor):
        """计算实例子区域一致性统计量 (v2新增)"""
        return self.subregion.compute_subregion_consistency(
            image_tensor, masks_list, spn_extractor, dct_extractor, clip_extractor)

    def fuse_features_v2(self, cross_instance_stats, subregion_stats, global_feats):
        """
        v2融合: 跨实例一致性(13) + 子区域一致性(24) + 全局特征(660) = 697维
        """
        return torch.cat([cross_instance_stats, subregion_stats, global_feats])

    def compute_global_features(self, spn_feats, dct_feats, clip_feats):
        """v1兼容: 全局平均特征"""
        if isinstance(spn_feats, list):
            spn_feats = torch.stack(spn_feats) if spn_feats else torch.zeros(1, 128)
        if isinstance(dct_feats, list):
            dct_feats = torch.stack(dct_feats) if dct_feats else torch.zeros(1, 21)
        if isinstance(clip_feats, list):
            clip_feats = torch.stack(clip_feats) if clip_feats else torch.zeros(1, 512)

        avg_spn = spn_feats.mean(0) if spn_feats.shape[0] > 0 else torch.zeros(128)
        avg_dct = dct_feats.mean(0) if dct_feats.shape[0] > 0 else torch.zeros(21)
        avg_clip = clip_feats.mean(0) if clip_feats.shape[0] > 0 else torch.zeros(512)

        # L2归一化到单位长度 (消除SPN/DCT/CLIP之间的尺度差异)
        avg_spn = F.normalize(avg_spn.unsqueeze(0), p=2, dim=-1).squeeze(0)
        avg_dct = F.normalize(avg_dct.unsqueeze(0), p=2, dim=-1).squeeze(0)
        avg_clip = F.normalize(avg_clip.unsqueeze(0), p=2, dim=-1).squeeze(0)

        return torch.cat([avg_spn, avg_dct, avg_clip])


# 兼容旧接口
CrossInstanceConsistency = CrossInstanceConsistencyV2
