"""
批量特征提取脚本
对数据集中的所有图像提取实例级取证特征，保存到磁盘
"""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.instance_dataset import InstanceForensicDataset
from features.segmenter import InstanceSegmenter
from features.spn_extractor import get_spn_extractor
from features.dct_profile import DCTProfileExtractor
from features.clip_encoder import get_clip_extractor
from models.consistency import CrossInstanceConsistency


class BatchFeatureExtractor:
    """批量特征提取器"""

    def __init__(self, data_dir, feature_dir, device='cuda'):
        self.data_dir = data_dir
        self.feature_dir = feature_dir
        self.device = device if torch.cuda.is_available() else 'cpu'

        print(f"[特征提取] 设备: {self.device}")
        print(f"[特征提取] 数据目录: {data_dir}")
        print(f"[特征提取] 特征保存: {feature_dir}")
        os.makedirs(feature_dir, exist_ok=True)

        # 初始化各模块
        print("初始化模块...")
        self.segmenter = InstanceSegmenter(device=self.device)
        self.spn_extractor = get_spn_extractor(mode='learned',
                                               device=self.device)
        self.dct_extractor = DCTProfileExtractor()
        self.clip_extractor = get_clip_extractor(device=self.device)

        print("所有模块初始化完成")

    @torch.no_grad()
    def extract_dataset(self, dataset_name='casia2', split='train',
                        max_samples=None):
        """
        提取整个数据集的实例级特征

        保存内容:
          features.npz 包含:
            - spn_feats: list of (K, 128) tensors
            - dct_feats: list of (K, 21) tensors
            - clip_feats: list of (K, 512) tensors
            - labels: list of ints
            - paths: list of str
            - instance_counts: list of ints
        """
        print(f"\n{'='*60}")
        print(f"提取 {dataset_name}/{split}")
        print(f"{'='*60}")

        dataset = InstanceForensicDataset(
            self.data_dir, dataset_name, split,
            img_size=512
        )

        loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False,
            num_workers=2, pin_memory=True,
        )

        all_spn, all_dct, all_clip = [], [], []
        all_labels, all_paths, all_counts = [], [], []

        for batch_idx, (img, label, path) in enumerate(
                tqdm(loader, desc=f"Extracting {dataset_name}/{split}")):
            if max_samples and batch_idx >= max_samples:
                break

            img = img.squeeze(0)  # (3, H, W)
            label = label.item()

            # 1. 实例分割
            masks_list = self.segmenter.segment(img)
            # 过滤小实例
            H, W = img.shape[-2:]
            min_area = 32 * 32
            masks_list = [(m, b) for m, b in masks_list
                          if m.sum() > min_area]

            K = len(masks_list)
            all_counts.append(K)

            if K < 2:
                # 少于2个实例, 使用整图特征并复制一份
                img_d = img.to(self.device)
                spn_f = self.spn_extractor.extract(img_d)
                dct_f = self.dct_extractor.extract_profile(img)
                clip_f = self.clip_extractor.encode_image(img)
                all_spn.append(spn_f.unsqueeze(0))  # (1, 128)
                all_dct.append(dct_f.unsqueeze(0))  # (1, 21)
                all_clip.append(clip_f.unsqueeze(0))  # (1, 512)
            else:
                # 逐实例提取
                spn_list, dct_list, clip_list = [], [], []
                for mask, bbox in masks_list:
                    mask_t = torch.from_numpy(mask).float().to(self.device)
                    img_d = img.to(self.device)

                    spn_f = self.spn_extractor.extract_instance(img_d, mask_t)
                    dct_f = self.dct_extractor.extract_profile(img, mask)
                    clip_f = self.clip_extractor.encode_image(img, mask)

                    spn_list.append(spn_f)
                    dct_list.append(dct_f)
                    clip_list.append(clip_f)

                all_spn.append(torch.stack(spn_list))
                all_dct.append(torch.stack(dct_list))
                all_clip.append(torch.stack(clip_list))

            all_labels.append(label)
            all_paths.append(path[0] if isinstance(path, list) else path)

            # 每100张保存一次中间结果
            if (batch_idx + 1) % 100 == 0:
                self._save_checkpoint(
                    dataset_name, split, batch_idx + 1,
                    all_spn, all_dct, all_clip,
                    all_labels, all_paths, all_counts
                )

        # 最终保存
        save_path = os.path.join(
            self.feature_dir, f'{dataset_name}_{split}_features.pt')
        # 收集掩膜列表 (v2子区域一致性)
        all_masks = getattr(self, '_all_masks', None)
        
        save_dict = {
            'spn_feats': all_spn,
            'dct_feats': all_dct,
            'clip_feats': all_clip,
            'labels': all_labels,
            'paths': all_paths,
            'instance_counts': all_counts,
            'dataset': dataset_name,
            'split': split,
            'num_samples': len(all_labels),
        }
        if all_masks is not None:
            save_dict['masks'] = all_masks
        
        torch.save(save_dict, save_path)
        print(f"  特征已保存: {save_path} ({len(all_labels)} samples)")

        return save_path

    def _save_checkpoint(self, dataset_name, split, n_processed,
                         all_spn, all_dct, all_clip,
                         all_labels, all_paths, all_counts):
        """保存中间检查点"""
        ckpt_path = os.path.join(
            self.feature_dir,
            f'{dataset_name}_{split}_ckpt_{n_processed}.pt')
        torch.save({
            'spn_feats': all_spn,
            'dct_feats': all_dct,
            'clip_feats': all_clip,
            'labels': all_labels,
            'paths': all_paths,
            'instance_counts': all_counts,
            'n_processed': n_processed,
        }, ckpt_path)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=
        '/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard/data')
    parser.add_argument('--feature_dir', type=str, default=
        '/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard/results/features')
    parser.add_argument('--datasets', nargs='+',
                        default=['casia2', 'columbia'])
    parser.add_argument('--splits', nargs='+',
                        default=['train', 'val', 'test'])
    parser.add_argument('--max_samples', type=int, default=None,
                        help='每个子集最大样本数 (用于调试)')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    extractor = BatchFeatureExtractor(
        data_dir=args.data_dir,
        feature_dir=args.feature_dir,
        device=args.device,
    )

    for dataset_name in args.datasets:
        for split in args.splits:
            extractor.extract_dataset(
                dataset_name=dataset_name,
                split=split,
                max_samples=args.max_samples,
            )


if __name__ == '__main__':
    main()
