"""
训练脚本 v2 — 支持增强特征(697维)、子区域一致性、Focal Loss
"""

import os, sys, torch, torch.nn as nn
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.consistency import CrossInstanceConsistencyV2, SubRegionConsistency
from models.fusion_classifier import FusionClassifierV2, FocalLoss


def compute_fused_features(spn_batch, dct_batch, clip_batch,
                           consistency, device='cpu',
                           image_tensors=None, masks_batch=None,
                           spn_extractor=None, dct_extractor=None, clip_extractor=None):
    """
    从批次特征计算v2融合特征 (697维)
    
    如果提供了image_tensors和masks, 额外计算子区域一致性
    """
    batch_fused = []
    for i in range(len(spn_batch)):
        stats, _, _, _ = consistency.compute_consistency(
            spn_batch[i], dct_batch[i], clip_batch[i])
        
        # 子区域一致性 (如果提供图像和掩膜)
        if image_tensors is not None and masks_batch is not None and i < len(masks_batch):
            sub_stats = consistency.compute_subregion_stats(
                image_tensors[i], masks_batch[i],
                spn_extractor, dct_extractor, clip_extractor)
        else:
            sub_stats = torch.zeros(24)
        
        global_f = consistency.compute_global_features(
            spn_batch[i], dct_batch[i], clip_batch[i])
        fused = torch.cat([stats.to(device), sub_stats.to(device), global_f.to(device)])
        batch_fused.append(fused)

    return torch.stack(batch_fused)


def load_features(feature_dir, dataset_name='casia2', split='train'):
    """加载预提取的特征"""
    path = os.path.join(feature_dir, f'{dataset_name}_{split}_features.pt')
    if not os.path.exists(path):
        print(f"  特征文件不存在: {path}")
        return None
    data = torch.load(path, map_location='cpu', weights_only=True)
    print(f"  加载 {dataset_name}/{split}: {len(data['labels'])} samples")
    return data


def train_epoch(model, optimizer, criterion, spn_data, dct_data, clip_data,
                labels, consistency, device, batch_size=64,
                image_tensors=None, masks_data=None,
                spn_extractor=None, dct_extractor=None, clip_extractor=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    n = len(labels)
    indices = torch.randperm(n)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        idx = indices[start:end]
        
        spn_batch = [spn_data[i] for i in idx]
        dct_batch = [dct_data[i] for i in idx]
        clip_batch = [clip_data[i] for i in idx]
        label_batch = torch.tensor([labels[i] for i in idx],
                                   dtype=torch.float32, device=device)
        
        imgs = [image_tensors[i] for i in idx] if image_tensors is not None else None
        masks = [masks_data[i] for i in idx] if masks_data is not None else None
        
        fused = compute_fused_features(
            spn_batch, dct_batch, clip_batch, consistency, device,
            imgs, masks, spn_extractor, dct_extractor, clip_extractor)
        
        logits = model(fused).squeeze(-1)
        loss = criterion(logits, label_batch)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(idx)

    return total_loss / n


@torch.no_grad()
def evaluate(model, spn_data, dct_data, clip_data,
             labels, consistency, device, threshold=0.5,
             image_tensors=None, masks_data=None,
             spn_extractor=None, dct_extractor=None, clip_extractor=None):
    """评估模型"""
    model.eval()
    n = len(labels)
    all_preds, all_probas = [], []

    for i in range(n):
        spn = [spn_data[i]]
        dct = [dct_data[i]]
        clip = [clip_data[i]]
        
        imgs = [image_tensors[i]] if image_tensors is not None else None
        masks = [masks_data[i]] if masks_data is not None else None
        
        fused = compute_fused_features(
            spn, dct, clip, consistency, device,
            imgs, masks, spn_extractor, dct_extractor, clip_extractor)
        
        proba = torch.sigmoid(model(fused))
        pred = (proba > threshold).float()
        all_preds.append(pred.item())
        all_probas.append(proba.item())

    all_preds = np.array(all_preds)
    all_labels = np.array(labels)
    all_probas = np.array(all_probas)

    acc = (all_preds == all_labels).mean()
    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    tn = ((all_preds == 0) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {'accuracy': float(acc), 'precision': float(precision),
            'recall': float(recall), 'f1': float(f1),
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)}


def train(dataset_name='casia2', feature_dir=None, output_dir=None,
          epochs=100, lr=1e-3, batch_size=64, device='cuda',
          eval_datasets=None, use_focal=False):
    """完整训练流程"""
    if feature_dir is None:
        feature_dir = '/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard/results/features'
    if output_dir is None:
        output_dir = '/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard/results/models'
    os.makedirs(output_dir, exist_ok=True)
    device = device if torch.cuda.is_available() else 'cpu'
    print(f"[训练] 设备: {device}, 数据集: {dataset_name}")

    consistency = CrossInstanceConsistencyV2()
    
    train_data = load_features(feature_dir, dataset_name, 'train')
    if train_data is None:
        print("错误: 训练特征未找到")
        return
    spn_train, dct_train, clip_train = train_data['spn_feats'], train_data['dct_feats'], train_data['clip_feats']
    labels_train = train_data['labels']

    val_data = load_features(feature_dir, dataset_name, 'val')
    if val_data:
        spn_val, dct_val, clip_val = val_data['spn_feats'], val_data['dct_feats'], val_data['clip_feats']
        labels_val = val_data['labels']
    else:
        spn_val, dct_val, clip_val, labels_val = [], [], [], []

    # v2输入维度: 13(跨实例) + 24(子区域) + 660(全局) = 697
    input_dim = 13 + 24 + 128 + 21 + 512  # = 698
    print(f"模型输入维度(v2): {input_dim}")
    
    model = FusionClassifierV2(input_dim=input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = FocalLoss(alpha=0.75) if use_focal else nn.BCEWithLogitsLoss()

    best_val_acc = 0.0
    for epoch in range(epochs):
        loss = train_epoch(model, optimizer, criterion,
                          spn_train, dct_train, clip_train, labels_train,
                          consistency, device, batch_size)
        scheduler.step()

        if val_data and (epoch + 1) % 10 == 0:
            val_metrics = evaluate(model, spn_val, dct_val, clip_val, labels_val,
                                  consistency, device)
            val_acc = val_metrics['accuracy']
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), os.path.join(output_dir, 'fusion_best.pth'))
                print(f"  Epoch {epoch+1}: loss={loss:.4f}, val_acc={val_acc:.4f} (new best!)")
            else:
                print(f"  Epoch {epoch+1}: loss={loss:.4f}, val_acc={val_acc:.4f}")

    torch.save(model.state_dict(), os.path.join(output_dir, 'fusion_final.pth'))
    print(f"训练完成! 最佳验证精度: {best_val_acc:.4f}")

    # 测试
    test_data = load_features(feature_dir, dataset_name, 'test')
    if test_data:
        model.load_state_dict(torch.load(os.path.join(output_dir, 'fusion_best.pth'),
                                          map_location=device, weights_only=True))
        test_metrics = evaluate(model, test_data['spn_feats'], test_data['dct_feats'],
                               test_data['clip_feats'], test_data['labels'],
                               consistency, device)
        print(f"\n{dataset_name} 测试集:")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # 跨数据集泛化
    if eval_datasets:
        print(f"\n跨数据集泛化:")
        for eval_ds in eval_datasets:
            eval_data = load_features(feature_dir, eval_ds, 'test')
            if eval_data:
                m = evaluate(model, eval_data['spn_feats'], eval_data['dct_feats'],
                            eval_data['clip_feats'], eval_data['labels'],
                            consistency, device)
                print(f"  {dataset_name} → {eval_ds}: acc={m['accuracy']:.4f}, f1={m['f1']:.4f}")

    return model


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='casia2')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--eval_datasets', nargs='+', default=['columbia'])
    parser.add_argument('--feature_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--use_focal', action='store_true')
    args = parser.parse_args()
    train(dataset_name=args.dataset, epochs=args.epochs, lr=args.lr,
          batch_size=args.batch_size, device=args.device,
          eval_datasets=args.eval_datasets, feature_dir=args.feature_dir,
          output_dir=args.output_dir, use_focal=args.use_focal)
