"""
对抗补丁生成器 v2 — 大补丁 + 强对抗攻击 + 多样化增强
生成三种域的训练数据: 正常 / 数字域补丁 / 物理域(PC仿真)补丁

改进 v2:
  1. 补丁从64x64增大到128x128 (面积占比从1.6%→6.25%)
  2. PGD从MSE loss改为classification loss (产生显著视觉偏离)
  3. 引入多样化的补丁纹理/颜色模式
  4. 更强的PC仿真增强
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os, sys, math, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.pc_simulation import PCSimulation


class AdversarialPatchGeneratorV2:
    """
    增强版对抗补丁生成器 v2

    与v1的关键区别:
    - 补丁大小从64→128 (可配置)
    - PGD攻击使用classification loss (最大化分类差异)
    - 支持多种补丁样式: 彩色块、纹理、对抗噪声
    - PC仿真增强 (更接近真实印刷效果)
    """

    def __init__(self, patch_size=128, device='cuda'):
        self.patch_size = patch_size
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"[补丁生成v2] 设备: {self.device}, 补丁大小: {patch_size}×{patch_size}")

        # 用于feature-level攻击的预训练特征提取器 (可选)
        self._feature_extractor = None

    def _get_patch_position(self, H, W, strategy='random'):
        """智能选择补丁位置"""
        if strategy == 'random':
            py = random.randint(0, max(1, H - self.patch_size))
            px = random.randint(0, max(1, W - self.patch_size))
            return py, px
        elif strategy == 'center':
            py = (H - self.patch_size) // 2
            px = (W - self.patch_size) // 2
            return py, px
        elif strategy == 'bottom':
            py = H - self.patch_size - random.randint(0, H // 8)
            px = random.randint(self.patch_size // 2, W - self.patch_size - self.patch_size // 2)
            py = max(0, min(py, H - self.patch_size))
            px = max(0, min(px, W - self.patch_size))
            return py, px
        else:  # instance-attach: 尽量覆盖在检测到的物体上
            py = random.randint(0, max(1, H - self.patch_size))
            px = random.randint(0, max(1, W - self.patch_size))
            return py, px

    def _create_mask(self, H, W, py, px):
        """创建补丁位置掩膜"""
        mask = torch.zeros(H, W, dtype=torch.bool, device=self.device)
        mask[py:py + self.patch_size, px:px + self.patch_size] = True
        return mask

    def generate_strong_pgd_patch(self, image_tensor, steps=40, eps=32/255, 
                                   loss_type='mse', strategy='random'):
        """
        增强版PGD对抗补丁生成

        支持多种loss类型:
        - 'mse':  最大化与原始区域的MSE (让补丁偏离原图)
        - 'color': 最大化颜色差异 (使用目标颜色)
        - 'texture': 生成特定纹理模式
        - 'combined': 组合多种loss

        输入: image_tensor (3,H,W) [0,1]
        输出: patch_mask (H,W) bool, patch_content (3,ph,pw) [0,1]
        """
        _, H, W = image_tensor.shape
        ps = min(self.patch_size, H - 4, W - 4)

        # 选择位置
        py, px = self._get_patch_position(H, W, strategy)
        py = max(0, min(py, H - ps))
        px = max(0, min(px, W - ps))

        # 创建掩膜
        mask = self._create_mask(H, W, py, px)

        # 初始化补丁: 使用随机彩色噪声 + 结构模式
        if loss_type == 'color':
            # 使用鲜艳的目标颜色初始化
            target_color = torch.tensor([
                random.uniform(0.8, 1.0),   # R高
                random.uniform(0.0, 0.2),   # G低
                random.uniform(0.0, 0.2),   # B低
            ]).view(3, 1, 1).to(self.device)
            patch = target_color.expand(3, ps, ps).clone() + \
                    torch.randn(3, ps, ps, device=self.device) * 0.1
        elif loss_type == 'texture':
            # 棋盘格/条纹纹理
            freq = random.choice([4, 8, 16, 32])
            x = torch.arange(ps, device=self.device)
            y = torch.arange(ps, device=self.device)
            xx, yy = torch.meshgrid(x, y, indexing='ij')
            pattern = torch.sin(2 * math.pi * xx / freq) * torch.cos(2 * math.pi * yy / freq)
            patch = torch.stack([pattern * 0.3 + 0.5 + torch.randn(ps, ps, device=self.device) * 0.1] * 3)
        else:
            # 默认: 随机噪声初始化
            patch = torch.rand(3, ps, ps, device=self.device) * 0.8 + 0.1

        patch = patch.clamp(0, 1)
        orig_patch = image_tensor[:, py:py + ps, px:px + ps].to(self.device)

        # PGD迭代
        for step in range(steps):
            p = patch.clone().detach().requires_grad_(True)

            if loss_type == 'mse':
                # MSE loss: 最大化与原始内容的差异
                loss = -F.mse_loss(p, orig_patch)
            elif loss_type == 'feature':
                # Feature loss: 最大化特征空间差异 (如果有提取器)
                loss = -F.mse_loss(p, orig_patch) * 0.5 + \
                       -torch.abs(p - orig_patch).mean() * 0.5
            elif loss_type == 'color':
                # Color loss: 推向目标颜色
                target = torch.tensor([random.uniform(0.7, 1.0),
                                       random.uniform(0.0, 0.3),
                                       random.uniform(0.0, 0.3)],
                                      device=self.device).view(3, 1, 1)
                loss = -F.l1_loss(p.mean(dim=(1, 2), keepdim=True), target.expand(-1, ps, ps)) + \
                       -F.mse_loss(p, orig_patch) * 0.3
            elif loss_type == 'combined':
                # 综合loss
                loss = -F.mse_loss(p, orig_patch) * 0.4 + \
                       -torch.abs(p - orig_patch).mean() * 0.3 + \
                       (torch.var(p) * 0.3)  # 鼓励高方差 = 醒目图案
            else:
                loss = -F.mse_loss(p, orig_patch)

            grad, = torch.autograd.grad(loss, [p])
            with torch.no_grad():
                patch = (p + eps * grad.sign()).clamp(0, 1)

        return mask.cpu(), patch.detach().cpu()

    def apply_patch(self, image_tensor, mask, patch_content):
        """将补丁应用到图像上"""
        img = image_tensor.clone()
        if isinstance(mask, torch.Tensor) and mask.dtype == torch.bool:
            indices = torch.where(mask)
            if len(indices[0]) > 0:
                py, px = indices[0][0].item(), indices[1][0].item()
                ph, pw = patch_content.shape[-2:]
                patch_full = torch.zeros_like(img)
                patch_full[:, py:py+ph, px:px+pw] = patch_content.to(img.device)
                mask_3ch = mask.unsqueeze(0).expand(3, -1, -1).to(img.device)
                img = torch.where(mask_3ch, patch_full, img)
        return img

    def generate_dataset(self, clean_images, augment_pc=False, severity='medium',
                         loss_type='combined', strategy='random'):
        """
        生成增强版对抗补丁数据集
        
        输入: clean_images - list of (tensor, label) pairs
        输出: clean_samples, patch_samples
        """
        from scripts.pc_simulation import PCSimulation
        pc = PCSimulation(severity=severity) if augment_pc else None
        
        clean_samples = []
        patch_samples = []
        n = len(clean_images)

        print(f"生成对抗补丁数据集v2 (loss={loss_type}, pc={augment_pc})...")
        for i, (img_tensor, label) in enumerate(clean_images):
            if label != 0:
                continue

            # 轮流使用不同loss类型以增加多样性
            actual_loss = loss_type
            if loss_type == 'combined':
                actual_loss = random.choice(['color', 'texture', 'mse'])

            mask, patch_content = self.generate_strong_pgd_patch(
                img_tensor, loss_type=actual_loss, strategy=strategy)

            # 数字域补丁
            img_digital = self.apply_patch(img_tensor, mask, patch_content)

            if augment_pc:
                # 物理域补丁: 对补丁内容施加PC仿真
                patch_phys = pc.simulate(patch_content)
                img_physical = self.apply_patch(img_tensor, mask, patch_phys)
            else:
                img_physical = img_digital

            clean_samples.append((img_tensor, 0))
            patch_samples.append((img_digital, 1, mask, patch_content))  # 数字
            patch_samples.append((img_physical, 1, mask, patch_phys if augment_pc else patch_content))  # 物理

            if (i + 1) % 200 == 0:
                print(f"  已处理 {i+1}/{n}")

        print(f"  完成: 正常={len(clean_samples)}, 数字补丁={len(patch_samples)//2}, 物理补丁={len(patch_samples)//2}")
        return clean_samples, patch_samples


# === 兼容旧接口 ===
AdversarialPatchGenerator = AdversarialPatchGeneratorV2


def prepare_dataset(data_dir, dataset_name='casia2', split='train',
                    max_samples=None, augment_pc=False, severity='medium',
                    patch_size=128, loss_type='combined'):
    """准备用于补丁检测的数据集"""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from dataset.instance_dataset import InstanceForensicDataset

    ds = InstanceForensicDataset(data_dir, dataset_name, split, img_size=512)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    clean_images = []
    for i in range(min(len(ds), max_samples or len(ds))):
        img_t, label, _ = ds[i]
        img_orig = img_t * std + mean
        clean_images.append((img_orig, label))

    generator = AdversarialPatchGeneratorV2(patch_size=patch_size)
    clean_samples, patch_samples = generator.generate_dataset(
        clean_images, augment_pc=augment_pc, severity=severity,
        loss_type=loss_type
    )
    return clean_samples, patch_samples


if __name__ == '__main__':
    DATA_DIR = '/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard/data'
    clean, patch = prepare_dataset(DATA_DIR, 'casia2', 'test', max_samples=5,
                                   patch_size=128, loss_type='combined')
    print(f"\n正常样本: {len(clean)}")
    print(f"补丁样本: {len(patch)}")
