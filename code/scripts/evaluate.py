"""
评估脚本: 全面评估模型性能，生成论文结果表格
"""

import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.consistency import CrossInstanceConsistency
from models.fusion_classifier import FusionClassifier
from scripts.train import load_features, compute_fused_features, evaluate


def compute_baseline_pad(feature_dir, dataset, split):
    """
    模拟PAD基线: 使用语义独立性分析
    简化实现: 基于SPN一致性的最小值的检测
    """
    data = load_features(feature_dir, dataset, split)
    if data is None:
        return None

    spn = data['spn_feats']
    labels = data['labels']
    consistency = CrossInstanceConsistency()

    preds = []
    for i in range(len(spn)):
        if len(spn[i]) < 2:
            preds.append(0.5)  # 默认
            continue
        # 计算SPN一致性
        S = spn[i] @ spn[i].T
        K = S.shape[0]
        triu = torch.triu_indices(K, K, offset=1)
        vals = S[triu[0], triu[1]]
        min_sim = vals.min().item()
        # PAD-like: 低一致性 → 预测为篡改
        preds.append(1.0 if min_sim < 0.5 else 0.0)

    preds = np.array(preds)
    labels = np.array(labels)
    acc = (preds == labels).mean()
    return acc


def run_full_evaluation(feature_dir, model_path, device='cuda'):
    """运行完整评估，输出论文所需的所有表格"""
    device = device if torch.cuda.is_available() else 'cpu'
    consistency = CrossInstanceConsistency()

    # 加载模型
    input_dim = 13 + 128 + 21 + 512  # 674
    model = FusionClassifier(input_dim=input_dim)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device,
                                         weights_only=True))
        model = model.to(device)
        print(f"模型已加载: {model_path}")
    else:
        print(f"警告: 模型不存在: {model_path}")
        return

    datasets = ['casia2', 'columbia']
    all_metrics = {}

    print(f"\n{'='*70}")
    print("主结果: 各数据集上的检测精度")
    print(f"{'='*70}")
    print(f"{'Dataset':<15} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}")
    print("-" * 50)

    for ds in datasets:
        test_data = load_features(feature_dir, ds, 'test')
        if test_data is None:
            # 尝试 val
            test_data = load_features(feature_dir, ds, 'val')
        if test_data:
            metrics = evaluate(
                model, test_data['spn_feats'], test_data['dct_feats'],
                test_data['clip_feats'], test_data['labels'],
                consistency, device
            )
            all_metrics[ds] = metrics
            print(f"{ds:<15} {metrics['accuracy']*100:>7.1f}% "
                  f"{metrics['precision']*100:>7.1f}% "
                  f"{metrics['recall']*100:>7.1f}% "
                  f"{metrics['f1']*100:>7.1f}%")

    # 消融实验
    print(f"\n{'='*70}")
    print("消融实验: 特征贡献分析 (CASIA v2 test)")
    print(f"{'='*70}")

    test_data = load_features(feature_dir, 'casia2', 'test')
    if test_data:
        spn, dct, clip = test_data['spn_feats'], test_data['dct_feats'], test_data['clip_feats']
        labels = test_data['labels']

        configs = [
            ("Full (SPN+DCT+CLIP)", spn, dct, clip),
            ("w/o SPN", [torch.zeros_like(s) for s in spn], dct, clip),
            ("w/o DCT", spn, [torch.zeros_like(d) for d in dct], clip),
            ("w/o CLIP", spn, dct, [torch.zeros_like(c) for c in clip]),
        ]

        for name, s, d, c in configs:
            m = evaluate(model, s, d, c, labels, consistency, device)
            print(f"  {name:<25} acc={m['accuracy']*100:.1f}%  "
                  f"f1={m['f1']*100:.1f}%")

    # 跨数据集泛化
    print(f"\n{'='*70}")
    print("跨数据集泛化: CASIA v2 train → 其他数据集 test")
    print(f"{'='*70}")

    for ds in ['columbia']:
        test_data = load_features(feature_dir, ds, 'test')
        if test_data:
            m = evaluate(model, test_data['spn_feats'], test_data['dct_feats'],
                        test_data['clip_feats'], test_data['labels'],
                        consistency, device)
            print(f"  CASIA v2 → {ds:<15} acc={m['accuracy']*100:.1f}%")

    print(f"\n评估完成!")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_dir', type=str, default=
        '/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard/results/features')
    parser.add_argument('--model_path', type=str, default=
        '/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard/results/models/fusion_best.pth')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    run_full_evaluation(args.feature_dir, args.model_path, args.device)


if __name__ == '__main__':
    main()
