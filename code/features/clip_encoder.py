"""
CLIP语义嵌入特征提取模块
使用 transformers 库中的 CLIP 模型
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image


class CLIPFeatureExtractor:
    """
    CLIP语义特征提取器
    使用 transformers 的 CLIPModel (替代 openai/CLIP, 无需额外安装)

    输入: 图像区域
    输出: 512-dim 归一化嵌入向量
    """

    def __init__(self, model_name='openai/clip-vit-base-patch16', device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'

        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained(
            model_name, local_files_only=True)
        self.processor = CLIPProcessor.from_pretrained(
            model_name, local_files_only=True)
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"  [CLIP] 加载 {model_name}")

    @torch.no_grad()
    def encode_image(self, image_tensor, mask=None):
        """
        编码图像为CLIP嵌入
        输入:
          image_tensor: (3,H,W) torch tensor [0,1]
          mask: (H,W) 二值掩膜, 可选
        输出: (512,) 归一化嵌入向量
        """
        if isinstance(image_tensor, torch.Tensor):
            # tensor -> numpy -> PIL
            img_np = image_tensor.cpu().numpy().transpose(1, 2, 0)
            # 反归一化
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = img_np * std + mean
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

            # 应用掩膜
            if mask is not None:
                if isinstance(mask, torch.Tensor):
                    mask_np = mask.cpu().numpy()
                else:
                    mask_np = mask
                # 将掩膜外区域设为灰色 (避免黑色影响CLIP统计)
                gray = np.ones_like(img_np) * 128
                img_np = img_np * mask_np[:, :, None] + \
                         gray * (~mask_np)[:, :, None]
                img_np = img_np.astype(np.uint8)

            pil_img = Image.fromarray(img_np)
        else:
            pil_img = image_tensor

        # CLIP预处理 - 只提取 pixel_values
        inputs = self.processor(images=pil_img, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.device)

        # 编码 - get_image_features 返回 tensor
        outputs = self.model.get_image_features(pixel_values=pixel_values)
        # 确保是张量 (处理 tuple 或 BaseModelOutputWithPooling 的情况)
        if isinstance(outputs, tuple):
            embedding = outputs[0]
        elif hasattr(outputs, 'pooler_output'):
            embedding = outputs.pooler_output
        else:
            embedding = outputs
        # L2归一化
        embedding = F.normalize(embedding, p=2, dim=-1)
        return embedding.squeeze(0)  # (512,)

    def encode_instances(self, image_tensor, masks_list):
        """
        批量编码多个实例
        输入:
          image_tensor: (3,H,W)
          masks_list: [(mask, bbox), ...]
        输出: [(512,) tensor, ...]
        """
        features = []
        for mask, bbox in masks_list:
            f = self.encode_image(image_tensor, mask)
            features.append(f)
        return features


# 全局单例
_clip_extractor = None


def get_clip_extractor(device='cuda'):
    global _clip_extractor
    if _clip_extractor is None:
        _clip_extractor = CLIPFeatureExtractor(device=device)
    return _clip_extractor
