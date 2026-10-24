"""
YOLO26-seg 实例分割模块
对输入图像进行实例分割，返回每个实例的掩膜和边界框
"""

import torch
import numpy as np
from PIL import Image


class InstanceSegmenter:
    """
    YOLO26n-seg 实例分割器
    输入: 图像 (B,3,H,W) 或 (3,H,W) tensor, 或 PIL Image
    输出: 实例掩膜列表, 每项为 (mask: np.array(H,W), bbox: [x1,y1,x2,y2])
    """

    def __init__(self, weights_path=None, device='cuda', conf_thresh=0.25):
        from ultralytics import YOLO

        if weights_path is None:
            # 自动查找权重文件
            import os
            candidates = [
                '/media/oyp/工作/Projects/042_image_forensic/DuetGuard/watermark-yolo26-icassp/code/yolo26n-seg.pt',
                '/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard/code/yolo26n-seg.pt',
                'yolo26n-seg.pt',
            ]
            for c in candidates:
                if os.path.exists(c):
                    weights_path = c
                    break
            if weights_path is None:
                raise FileNotFoundError(
                    "YOLO26 weights not found. Please specify weights_path.")

        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(weights_path)
        self.conf_thresh = conf_thresh

    @torch.no_grad()
    def segment(self, image_tensor):
        """
        输入: image_tensor - (3,H,W) torch tensor, 归一化到 [0,1] 或 ImageNet标准
        输出: masks_list - [(mask, bbox), ...], mask=ndarray(H,W) bool, bbox=[x1,y1,x2,y2]
        """
        # 转换 tensor 为 numpy 用于 YOLO
        if isinstance(image_tensor, torch.Tensor):
            img_np = image_tensor.cpu().numpy().transpose(1, 2, 0)  # (H,W,3)
            # 反归一化 (如果使用 ImageNet 标准化)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = img_np * std + mean
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        else:
            img_np = image_tensor

        results = self.model(img_np, conf=self.conf_thresh, verbose=False)

        # 获取输入图像的原始尺寸
        if isinstance(image_tensor, torch.Tensor):
            orig_h, orig_w = image_tensor.shape[-2:]
        else:
            orig_h, orig_w = img_np.shape[:2]

        masks_list = []
        if results[0].masks is not None:
            masks_np = results[0].masks.data.cpu().numpy()  # (N, H_yolo, W_yolo)
            boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else None

            from skimage.transform import resize
            for i in range(len(masks_np)):
                # 将掩膜缩放到原始图像尺寸
                mask_resized = resize(
                    masks_np[i], (orig_h, orig_w),
                    order=0, preserve_range=True,
                    anti_aliasing=False
                ) > 0.5
                bbox = boxes[i].tolist() if boxes is not None else None
                masks_list.append((mask_resized, bbox))

        return masks_list

    def segment_batch(self, batch_tensor, batch_size=4):
        """批量分割，返回每张图的实例列表"""
        all_masks = []
        for i in range(0, len(batch_tensor), batch_size):
            sub_batch = batch_tensor[i:i + batch_size]
            for img in sub_batch:
                masks = self.segment(img)
                all_masks.append(masks)
        return all_masks


if __name__ == '__main__':
    # 测试
    segmenter = InstanceSegmenter(device='cpu')
    dummy = torch.randn(3, 512, 512)
    masks = segmenter.segment(dummy)
    print(f"检测到 {len(masks)} 个实例")
    for i, (mask, bbox) in enumerate(masks):
        print(f"  实例{i}: mask面积={mask.sum()}, bbox={bbox}")
