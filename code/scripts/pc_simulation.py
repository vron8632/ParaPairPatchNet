"""
Print-Camera (PC) 失真仿真管线 v2 (增强版)
模拟物理打印-拍摄过程中的各类失真 — 更强更真实的仿真

改进 v2:
  1. 更强的色域偏移 (模拟不同打印机特性)
  2. 随机纸张纹理/打印条纹
  3. 增强半色调抖动 + 墨滴扩散
  4. 镜头炫光/渐晕模拟
  5. 透视失真增强 (随机角度)
  6. 多级severity: mild/medium/strong/extreme
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import io
import random
import math


class PCSimulationV2:
    """
    增强版 Print-Camera 失真仿真器 v2

    使用方式:
        pc = PCSimulationV2(severity='medium')
        distorted = pc(image_tensor)
    """

    def __init__(self, severity='medium', seed=None):
        self.severity = severity
        self.rng = random.Random(seed)

        # 各严重等级的失真参数范围 (v2增强版)
        self.params = {
            'mild': {
                'blur_sigma': (0.3, 0.8),
                'noise_sigma': (0.005, 0.015),
                'jpeg_quality': (80, 95),
                'perspective_angle': (0.5, 2.0),
                'color_shift': (0.02, 0.05),
                'halftone_strength': (0.01, 0.03),
                'vignette': (0.0, 0.05),
            },
            'medium': {
                'blur_sigma': (0.8, 1.8),
                'noise_sigma': (0.015, 0.035),
                'jpeg_quality': (55, 80),
                'perspective_angle': (1.0, 4.0),
                'color_shift': (0.05, 0.10),
                'halftone_strength': (0.03, 0.06),
                'vignette': (0.05, 0.12),
            },
            'strong': {
                'blur_sigma': (1.5, 3.0),
                'noise_sigma': (0.03, 0.06),
                'jpeg_quality': (35, 60),
                'perspective_angle': (2.0, 6.0),
                'color_shift': (0.08, 0.15),
                'halftone_strength': (0.05, 0.10),
                'vignette': (0.10, 0.20),
            },
            'extreme': {
                'blur_sigma': (2.5, 4.5),
                'noise_sigma': (0.05, 0.10),
                'jpeg_quality': (20, 45),
                'perspective_angle': (3.0, 8.0),
                'color_shift': (0.12, 0.25),
                'halftone_strength': (0.08, 0.15),
                'vignette': (0.15, 0.30),
            }
        }

    def _sample_param(self, param_name):
        lo, hi = self.params[self.severity][param_name]
        return lo + self.rng.random() * (hi - lo)

    @torch.no_grad()
    def simulate(self, image_tensor):
        """
        对输入图像应用增强PC失真管线
        输入: (3, H, W) torch.Tensor, 值范围 [0, 1]
        输出: (3, H, W) torch.Tensor, 值范围 [0, 1]
        """
        img = image_tensor.clone()

        # === 步骤1: 强色域映射 (模拟印刷色彩偏移) ===
        img = self._color_gamut_mapping(img)

        # === 步骤2: 增强半色调抖动 + 墨滴扩散 ===
        img = self._halftoning(img)

        # === 步骤3: 纸张纹理叠加 ===
        img = self._paper_texture(img)

        # === 步骤4: 镜头模糊 ===
        blur_sigma = self._sample_param('blur_sigma')
        img = self._gaussian_blur(img, blur_sigma)

        # === 步骤5: 复合传感器噪声 (高斯+泊松+颜色噪声) ===
        noise_sigma = self._sample_param('noise_sigma')
        img = self._sensor_noise(img, noise_sigma)

        # === 步骤6: JPEG压缩 ===
        jpeg_q = int(self._sample_param('jpeg_quality'))
        img = self._jpeg_compress(img, jpeg_q)

        # === 步骤7: 透视失真 ===
        angle = self._sample_param('perspective_angle')
        img = self._perspective_distortion(img, angle)

        # === 步骤8: 渐晕效果 ===
        vignette = self._sample_param('vignette')
        img = self._vignette(img, vignette)

        # === 步骤9: 最终色彩偏移 (模拟白平衡偏差) ===
        img = self._white_balance_shift(img)

        return img.clamp(0, 1)

    def _color_gamut_mapping(self, img):
        """增强色域映射: 模拟CMYK印刷色彩偏移"""
        batch = img.unsqueeze(0)
        strength = self._sample_param('color_shift')
        
        # 非均匀颜色变换矩阵 (模拟不同打印机的色彩特性)
        matrix = torch.eye(3) + torch.randn(3, 3) * strength * 2
        matrix = matrix.clamp(-0.5, 0.5)
        matrix[0, 0] += 0.90 + self.rng.random() * 0.10
        matrix[1, 1] += 0.90 + self.rng.random() * 0.10
        matrix[2, 2] += 0.90 + self.rng.random() * 0.10

        c, h, w = batch.shape[1:]
        flat = batch.permute(0, 2, 3, 1)
        transformed = flat @ matrix.T.to(flat.device)
        # 添加非线性gamma偏移
        transformed = torch.pow(transformed.clamp(0.01, 1), 1.0 / (1.0 + self.rng.random() * 0.3))
        result = transformed.permute(0, 3, 1, 2)
        return result.squeeze(0).clamp(0, 1)

    def _halftoning(self, img):
        """增强半色调抖动 + 墨滴扩散效果"""
        strength = self._sample_param('halftone_strength')
        _, H, W = img.shape
        
        # 高频抖动噪声 (模拟印刷网点)
        dither = torch.randn_like(img) * strength
        
        # 低频墨滴扩散 (模拟墨水在纸张上的扩散)
        grid_size = max(4, int(8 - strength * 20))
        ink_noise = torch.randn(3, H // grid_size + 1, W // grid_size + 1)
        ink_noise = F.interpolate(ink_noise.unsqueeze(0), size=(H, W), 
                                   mode='bilinear', align_corners=False).squeeze(0)
        
        return (img + dither + ink_noise * strength * 0.5).clamp(0, 1)

    def _paper_texture(self, img):
        """叠加纸张纹理"""
        _, H, W = img.shape
        # 生成纸张纤维纹理
        texture = torch.randn(1, H, W)
        # 使用多次模糊模拟纤维结构
        for _ in range(2):
            k = 5
            kernel = torch.ones(1, 1, k, k) / (k * k)
            texture = F.conv2d(texture.unsqueeze(0), kernel.to(texture.device), 
                               padding=k//2).squeeze(0)
        texture = texture / texture.std() * 0.03
        return (img + texture.expand(3, -1, -1).to(img.device)).clamp(0, 1)

    def _gaussian_blur(self, img, sigma):
        """高斯模糊模拟镜头失焦"""
        if sigma < 0.3:
            return img
        kernel_size = int(2 * round(sigma * 3) + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(3, min(kernel_size, 21))

        ax = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()

        kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(3, 1, 1, 1)
        padded = F.pad(img.unsqueeze(0), (kernel_size // 2,) * 4, mode='reflect')
        result = F.conv2d(padded, kernel.to(img.device), groups=3)
        return result.squeeze(0).clamp(0, 1)

    def _sensor_noise(self, img, sigma):
        """复合传感器噪声: 高斯 + 泊松 + 颜色串扰"""
        # 高斯噪声
        gaussian = torch.randn_like(img) * sigma
        # 散粒噪声 (信号依赖)
        shot = torch.sqrt(img.clamp(0.01, 1)) * torch.randn_like(img) * sigma * 0.5
        # 颜色通道串扰噪声
        crosstalk = torch.randn(3, 1, 1) * sigma * 0.3
        return (img + gaussian + shot + crosstalk).clamp(0, 1)

    def _jpeg_compress(self, img, quality):
        """JPEG压缩模拟"""
        img_np = (img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        try:
            pil_img = Image.fromarray(img_np)
            buffer = io.BytesIO()
            pil_img.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            pil_decoded = Image.open(buffer)
            decoded_np = np.array(pil_decoded).astype(np.float32) / 255.0
            result = torch.from_numpy(decoded_np).permute(2, 0, 1).to(img.device)
            return result.clamp(0, 1)
        except Exception:
            return img

    def _perspective_distortion(self, img, angle_deg):
        """透视失真 + 随机裁剪"""
        if angle_deg < 0.5:
            return img
        _, h, w = img.shape
        
        # 随机透视变换
        from torchvision.transforms.functional import perspective
        # 模拟轻微偏转视角
        shift_h = int(h * angle_deg * 0.008)
        shift_w = int(w * angle_deg * 0.008)
        
        startpoints = [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
        endpoints = [
            [shift_w, shift_h],
            [w - 1 - shift_w, shift_h],
            [w - 1 - shift_w * 0.5, h - 1 - shift_h],
            [shift_w * 0.5, h - 1 - shift_h]
        ]
        
        try:
            img_pil = Image.fromarray((img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
            img_pil = img_pil.transform(img_pil.size, 
                                         Image.Transform.PERSPECTIVE,
                                         [p for pts in endpoints for p in pts] +
                                         [p for pts in startpoints for p in pts],
                                         Image.Resampling.BILINEAR)
            result = torch.from_numpy(np.array(img_pil).astype(np.float32) / 255.0).permute(2, 0, 1).to(img.device)
            return result.clamp(0, 1)
        except Exception:
            return img

    def _vignette(self, img, strength):
        """渐晕效果 (边缘暗角)"""
        if strength < 0.01:
            return img
        _, H, W = img.shape
        # 创建径向渐变
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=img.device),
            torch.linspace(-1, 1, W, device=img.device),
            indexing='ij'
        )
        radius = torch.sqrt(x ** 2 + y ** 2)
        vignette_mask = 1 - strength * (radius / radius.max())
        vignette_mask = vignette_mask.clamp(0.7, 1.0)
        return (img * vignette_mask).clamp(0, 1)

    def _white_balance_shift(self, img):
        """模拟白平衡偏移"""
        r_shift = 1.0 + (self.rng.random() - 0.5) * 0.1
        g_shift = 1.0 + (self.rng.random() - 0.5) * 0.08
        b_shift = 1.0 + (self.rng.random() - 0.5) * 0.1
        scale = torch.tensor([r_shift, g_shift, b_shift], device=img.device).view(3, 1, 1)
        return (img * scale).clamp(0, 1)


# 兼容旧接口
PCSimulation = PCSimulationV2


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from dataset.instance_dataset import InstanceForensicDataset
    from torchvision.utils import save_image

    DATA_DIR = '/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard/data'
    ds = InstanceForensicDataset(DATA_DIR, 'casia2', 'test', img_size=512)
    pc = PCSimulationV2(severity='medium')
    
    img, label, path = ds[0]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_orig = img * std + mean
    distorted = pc.simulate(img_orig)
    
    psnr = 20 * torch.log10(1.0 / (img_orig - distorted).pow(2).mean().sqrt())
    print(f'原始: [{img_orig.min():.3f}, {img_orig.max():.3f}]')
    print(f'失真: [{distorted.min():.3f}, {distorted.max():.3f}]')
    print(f'PSNR: {psnr:.2f} dB')
    
    save_image(torch.stack([img_orig, distorted]),
               '/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard/figures/pc_simulation_v2.png')
    print('对比图像已保存')
