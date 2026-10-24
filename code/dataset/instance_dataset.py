"""
Instance-Aware Forensic Feature Fusion - Dataset Loader
支持 CASIA v2 和 Columbia 数据集
"""

import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class InstanceForensicDataset(Dataset):
    """
    统一数据集接口，支持 CASIA v2 和 Columbia
    返回: image (3,H,W), label (0=真实, 1=篡改), image_path
    """

    def __init__(self, data_dir, dataset_name='casia2', split='train',
                 img_size=512, split_ratio=(0.6, 0.2, 0.2), seed=42):
        self.data_dir = data_dir
        self.img_size = img_size
        self.rng = np.random.RandomState(seed)

        if dataset_name == 'casia2':
            self.samples = self._load_casia2()
        elif dataset_name == 'columbia':
            self.samples = self._load_columbia()
        elif dataset_name == 'aigc_test':
            self.samples = self._load_aigc_test()
        elif dataset_name == 'augmented_coco':
            self.samples = self._load_augmented_coco()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # 划分数据集
        self.rng.shuffle(self.samples)
        n = len(self.samples)
        n_train = int(n * split_ratio[0])
        n_val = int(n * split_ratio[1])

        if split == 'train':
            self.samples = self.samples[:n_train]
        elif split == 'val':
            self.samples = self.samples[n_train:n_train + n_val]
        elif split == 'test':
            self.samples = self.samples[n_train + n_val:]
        else:
            raise ValueError(f"Unknown split: {split}")

        # 图像变换
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        print(f"[{dataset_name}] {split}: {len(self.samples)} images")

    def _load_casia2(self):
        """加载 CASIA v2: Au/ (真实), Tp/ (篡改)"""
        base = os.path.join(self.data_dir, 'casia2', 'CASIA2.0_revised')
        samples = []
        for label, subdir in [(0, 'Au'), (1, 'Tp')]:
            dir_path = os.path.join(base, subdir)
            if not os.path.isdir(dir_path):
                continue
            for fname in sorted(os.listdir(dir_path)):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.bmp')):
                    samples.append((os.path.join(dir_path, fname), label))
        return samples

    def _load_columbia(self):
        """加载 Columbia: Au-* (真实), Au-T* 也是真实, 拼接伪造在特定子目录? 使用彩色图像"""
        base = os.path.join(self.data_dir, 'columbia', 'ImSpliceDataset')
        samples = []
        # 所有子目录中的图片
        for subdir in sorted(os.listdir(base)):
            dir_path = os.path.join(base, subdir)
            if not os.path.isdir(dir_path):
                continue
            # 根据子目录名判断: 含"T"的是真实, 含"S"的是拼接
            # Au-T* = 真实, Au-S* = 拼接
            is_tampered = 'S' in subdir and 'T' not in subdir.split('S')[0]
            # 更精确: Au-S 系列是拼接
            is_tampered = subdir.startswith('Au-S')
            for fname in sorted(os.listdir(dir_path)):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.bmp')):
                    samples.append((os.path.join(dir_path, fname),
                                    1 if is_tampered else 0))
        return samples

    def _load_aigc_test(self):
        """加载 aigc_test: authentic/ (Au), forgery/ (Tp)"""
        base = os.path.join(self.data_dir, 'aigc_test')
        samples = []
        for label, subdir in [(0, 'authentic'), (1, 'forgery')]:
            dir_path = os.path.join(base, subdir)
            if not os.path.isdir(dir_path):
                continue
            for fname in sorted(os.listdir(dir_path)):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    samples.append((os.path.join(dir_path, fname), label))
        return samples

    def _load_augmented_coco(self):
        """
        加载 augmented_val (Tp) + COCO val2017 (Au)
        augmented_val 包含 5 种篡改类型: copy_move, inpainting, rect_paste, removal, splicing
        每种 200 张 = 1000 Tp
        COCO val2017 取前 1000 张作为 Au
        """
        # Tp: augmented_val 下所有子目录(除masks)
        aug_base = '/media/oyp/工作/Projects/042_image_forensic/DuetGuard/data/augmented_val'
        samples = []
        for subdir in sorted(os.listdir(aug_base)):
            if subdir == 'masks':
                continue
            dir_path = os.path.join(aug_base, subdir)
            if not os.path.isdir(dir_path):
                continue
            for fname in sorted(os.listdir(dir_path)):
                if fname.lower().endswith('.jpg'):
                    samples.append((os.path.join(dir_path, fname), 1))

        # Au: COCO val2017 (取前1000张以平衡)
        coco_base = '/media/oyp/工作/Projects/042_image_forensic/DuetGuard/data/coco/val2017'
        if os.path.isdir(coco_base):
            count = 0
            for fname in sorted(os.listdir(coco_base)):
                if fname.lower().endswith('.jpg'):
                    samples.append((os.path.join(coco_base, fname), 0))
                    count += 1
                    if count >= 1000:
                        break

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
            img = self.transform(img)
            return img, label, path
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # 返回下一个有效样本
            return self.__getitem__((idx + 1) % len(self.samples))


def get_dataloader(data_dir, dataset_name='casia2', split='train',
                   batch_size=32, img_size=512, num_workers=4):
    """获取数据加载器"""
    dataset = InstanceForensicDataset(
        data_dir=data_dir, dataset_name=dataset_name,
        split=split, img_size=img_size,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=(split == 'train'),
        num_workers=num_workers, pin_memory=True,
    )
    return loader


def compute_dataset_stats(data_dir, dataset_name='casia2'):
    """打印数据集统计信息"""
    for split in ['train', 'val', 'test']:
        ds = InstanceForensicDataset(data_dir, dataset_name, split, img_size=512)
        labels = [s[1] for s in ds.samples]
        n_real = sum(1 for l in labels if l == 0)
        n_fake = sum(1 for l in labels if l == 1)
        print(f"  {split}: {len(ds)} images "
              f"(real={n_real}, tampered={n_fake}, "
              f"ratio={n_fake/(n_real+1):.2f})")


if __name__ == '__main__':
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else \
        '/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard/data'
    print("=== CASIA v2 ===")
    compute_dataset_stats(data_dir, 'casia2')
    print("\n=== Columbia ===")
    compute_dataset_stats(data_dir, 'columbia')
