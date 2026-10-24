"""
DCT频域剖面特征提取模块
提取图像每个实例区域的DCT频带能量分布
"""

import torch
import numpy as np
from scipy.fftpack import dct


class DCTProfileExtractor:
    """
    DCT频域剖面特征提取器
    对图像或实例区域提取10频段能量分布 (参考ICASSP论文)

    DCT频带划分 (zigzag顺序):
    频带0: 直流DC
    频带1: 最低频 (1,0), (0,1)
    频带2-9: 按zigzag顺序划分
    """

    def __init__(self, block_size=8, n_bands=10):
        self.block_size = block_size
        self.n_bands = n_bands
        # 预计算zigzag频带映射
        self._init_band_mapping()

    def _init_band_mapping(self):
        """初始化频带索引映射"""
        n = self.block_size
        # zigzag顺序索引
        zigzag = []
        for s in range(2 * n - 1):
            if s % 2 == 0:
                for i in range(min(s, n - 1), max(-1, s - n), -1):
                    zigzag.append((i, s - i))
            else:
                for i in range(max(0, s - n + 1), min(s + 1, n)):
                    zigzag.append((i, s - i))

        # 频带划分: 第一个是DC, 其余按zigzag分n_bands-1组
        self.band_indices = [[] for _ in range(self.n_bands)]
        for idx, (u, v) in enumerate(zigzag):
            if idx == 0:
                band = 0  # DC
            else:
                band = 1 + (idx - 1) * (self.n_bands - 1) // (n * n - 1)
                band = min(band, self.n_bands - 1)
            self.band_indices[band].append((u, v))

    def extract_profile(self, image_tensor, mask=None):
        """
        提取DCT频域剖面特征
        输入:
          image_tensor: (3,H,W) torch tensor [0,1]
          mask: (H,W) 二值掩膜 (numpy bool), 可选
        输出: (20,) 特征 (10频段能量均值 + 10频段能量方差)
        """
        # 转为灰度并转到numpy
        gray = image_tensor.mean(0).cpu().numpy()  # (H,W)

        if mask is not None:
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            gray = gray * mask

        H, W = gray.shape
        bs = self.block_size

        # 分块DCT
        energy_bands = []
        for i in range(0, H - bs + 1, bs):
            for j in range(0, W - bs + 1, bs):
                block = gray[i:i + bs, j:j + bs]
                dct_block = dct(dct(block, axis=0, norm='ortho'),
                                axis=1, norm='ortho')
                dct_energy = dct_block ** 2

                # 提取每个频带的能量
                block_profile = []
                for band in range(self.n_bands):
                    energy = 0
                    for (u, v) in self.band_indices[band]:
                        if u < bs and v < bs:
                            energy += dct_energy[u, v]
                    block_profile.append(energy)

                # 归一化: 总能量
                total = sum(block_profile)
                if total > 1e-10:
                    block_profile = [e / total for e in block_profile]
                energy_bands.append(block_profile)

        if len(energy_bands) == 0:
            return torch.zeros(self.n_bands * 2)

        # 聚合: 每个频带的均值和方差
        energy_np = np.array(energy_bands)  # (num_blocks, n_bands)
        mean_profile = energy_np.mean(axis=0)  # (n_bands,)
        var_profile = energy_np.var(axis=0)  # (n_bands,)

        # 额外: KL散度 vs 参考分布
        # 参考分布: 均匀分布 (理想自然图像的近似)
        ref = np.ones(self.n_bands) / self.n_bands
        kl_div = np.sum(mean_profile * np.log(mean_profile / ref + 1e-10))

        # 拼接: mean(10) + var(10) + kl(1)
        profile = np.concatenate([mean_profile, var_profile, [kl_div]])
        return torch.from_numpy(profile).float()

    def extract_instances(self, image_tensor, masks_list):
        """
        批量提取多个实例的DCT特征
        输入:
          image_tensor: (3,H,W)
          masks_list: [(mask, bbox), ...]
        输出: [(20,) tensor, ...]
        """
        features = []
        for mask, bbox in masks_list:
            f = self.extract_profile(image_tensor, mask)
            features.append(f)
        return features


if __name__ == '__main__':
    extractor = DCTProfileExtractor()
    dummy = torch.randn(3, 512, 512)
    profile = extractor.extract_profile(dummy)
    print(f"DCT profile shape: {profile.shape}")
    print(f"Profile value: {profile}")
