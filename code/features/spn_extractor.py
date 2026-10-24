"""
SPN噪声残差特征提取模块
使用经典SPN相关方法 (Lukas et al. 2006)
配合DuetGuard预训练SPN提取器加速
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import uniform_filter


class SPNFeatureExtractor:
    """
    SPN特征提取器
    支持两种模式:
    1. 'classic': 经典SPN相关方法 (Lukas 2006)
    2. 'learned': 使用DuetGuard预训练SPN提取器 (推荐)
    """

    def __init__(self, mode='learned', weights_path=None, device='cuda'):
        self.mode = mode
        self.device = device if torch.cuda.is_available() else 'cpu'

        if mode == 'learned':
            self._init_learned(weights_path)
        else:
            self._init_classic()

    def _init_learned(self, weights_path):
        """初始化学习式SPN提取器 (从DuetGuard复用)"""
        import sys
        import os
        import importlib.util
        # 使用绝对路径导入，避免与 code/models/ 包名冲突
        apjf_path = '/media/oyp/工作/Projects/042_image_forensic/DuetGuard/apjf_src'
        module_path = os.path.join(apjf_path, 'models', 'spn_extractor.py')
        if os.path.exists(module_path):
            spec = importlib.util.spec_from_file_location(
                "duetguard_spn_extractor", module_path)
            spn_module = importlib.util.module_from_spec(spec)
            sys.modules["duetguard_spn_extractor"] = spn_module
            spec.loader.exec_module(spn_module)
            SPNExtractor = spn_module.SPNExtractor
        else:
            raise FileNotFoundError(
                f"DuetGuard SPN extractor not found at {module_path}")

        self.model = SPNExtractor(fp_dim=128, base_ch=64,
                                  num_blocks=2, expand_ratio=4)
        self.model.eval()

        if weights_path is None:
            weights_path = '/media/oyp/工作/Projects/042_image_forensic/DuetGuard/weights/spn_extractor_best.pth'

        if os.path.exists(weights_path):
            state = torch.load(weights_path, map_location='cpu',
                               weights_only=True)
            # 兼容不同保存格式
            if 'model_state_dict' in state:
                self.model.load_state_dict(state['model_state_dict'])
            else:
                self.model.load_state_dict(state)
            print(f"  [SPN] 加载权重: {weights_path}")
        else:
            print(f"  [SPN] 警告: 权重 {weights_path} 未找到, 使用随机初始化")

        self.model = self.model.to(self.device)

    def _init_classic(self):
        """初始化经典SPN方法"""
        # 使用固定高通滤波器
        kernel = torch.tensor([[[[-1, -1, -1],
                                  [-1, 8, -1],
                                  [-1, -1, -1]]]], dtype=torch.float32) / 8.0
        self.register_buffer('hpf_kernel', kernel)

    @torch.no_grad()
    def extract(self, image_tensor):
        """
        提取SPN特征
        输入: image_tensor - (3,H,W) torch tensor, 值范围 [0,1]
        输出: spn_feature - (128,) torch tensor (学习式) 或 (1,) 相关值 (经典式)
        """
        if self.mode == 'learned':
            return self._extract_learned(image_tensor)
        else:
            return self._extract_classic(image_tensor)

    @torch.no_grad()
    def _extract_learned(self, image_tensor):
        """学习式SPN提取"""
        # 确保输入形状为 (1,3,H,W)
        if image_tensor.dim() == 3:
            img = image_tensor.unsqueeze(0)  # (1,3,H,W)
        else:
            img = image_tensor

        # 缩放到256 (SPN提取器输入尺寸)
        if img.shape[-1] != 256:
            img = F.interpolate(img, size=(256, 256), mode='bilinear',
                                align_corners=False)

        img = img.to(self.device)
        fingerprint, noise_map = self.model(img)
        return fingerprint.squeeze(0)  # (128,)

    @torch.no_grad()
    def _extract_classic(self, image_tensor):
        """经典SPN相关法"""
        # 简化的噪声残差提取
        gray = image_tensor.mean(0)  # (H,W)
        # 使用均值滤波近似去噪
        if gray.dim() == 2:
            gray_np = gray.cpu().numpy()
            denoised = uniform_filter(gray_np, size=5)
            noise = torch.from_numpy(gray_np - denoised).float()
            # 返回噪声标准差作为简单特征
            return torch.tensor([noise.std().item()])
        return torch.zeros(1)

    def extract_instance(self, image_tensor, mask):
        """
        提取实例区域的SPN特征
        输入:
          image_tensor: (3,H,W)
          mask: (H,W) 二值掩膜 (numpy bool)
        输出: (128,) 特征向量
        """
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).float()

        # 裁剪到实例区域
        mask_3ch = mask.unsqueeze(0).repeat(3, 1, 1)  # (3,H,W)
        masked_img = image_tensor * mask_3ch

        return self.extract(masked_img)


# 全局单例
_spn_extractor = None


def get_spn_extractor(mode='learned', device='cuda'):
    """获取全局SPN提取器（单例）"""
    global _spn_extractor
    if _spn_extractor is None:
        _spn_extractor = SPNFeatureExtractor(mode=mode, device=device)
    return _spn_extractor
