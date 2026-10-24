#!/usr/bin/env python3
"""
Bootstrap 置信区间 — ForensicGuard

对每个数据集的 test 预测做 bootstrap 重采样 (B=1000)，计算
检测准确率的 95% 置信区间。基于已训练模型推理，无需重新训练。

用法:
    python3 code/eval_bootstrap_ci.py
输出:
    results/bootstrap_ci.json
"""
import sys, os, json, collections, torch
import numpy as np

BASE = '/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard'
SCRIPT_DIR = os.path.join(BASE, 'code')
FEATURE_DIR = os.path.join(BASE, 'results/features')
MODEL_DIR = os.path.join(BASE, 'results/models')
OUT_FILE = os.path.join(BASE, 'results/bootstrap_ci.json')

sys.path.insert(0, SCRIPT_DIR)
from models.consistency import CrossInstanceConsistencyV2
from models.fusion_classifier import FusionClassifierV2
from scripts.train import compute_fused_features

device = 'cuda' if torch.cuda.is_available() else 'cpu'
B = 1000  # bootstrap 重采样次数
ALPHA = 0.95


def get_predictions(dataset):
    """加载特征+模型，返回所有样本的 (pred, label)"""
    feat_path = f'{FEATURE_DIR}/{dataset}_test_features.pt'
    data = torch.load(feat_path, map_location='cpu', weights_only=False)
    s_te, d_te, c_te, l_te = (data['spn_feats'], data['dct_feats'],
                               data['clip_feats'], data['labels'])
    model = FusionClassifierV2(input_dim=698).to(device)
    model.load_state_dict(torch.load(f'{MODEL_DIR}/{dataset}/fusion_best.pth',
                                     map_location='cpu', weights_only=False))
    model.eval()
    cic = CrossInstanceConsistencyV2()

    preds, labels = [], []
    with torch.no_grad():
        for i in range(len(l_te)):
            spn, dct, clip = [s_te[i]], [d_te[i]], [c_te[i]]
            fused = compute_fused_features(spn, dct, clip, cic, device)
            proba = torch.sigmoid(model(fused)).item()
            preds.append(1 if proba > 0.5 else 0)
            labels.append(l_te[i])
    return np.array(preds), np.array(labels)


def bootstrap_ci(preds, labels, n_boot=B):
    """bootstrap 重采样计算准确率 95% CI"""
    n = len(labels)
    rng = np.random.default_rng(42)
    accs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        accs.append((preds[idx] == labels[idx]).mean())
    accs = np.array(accs)
    lo = np.percentile(accs, (1 - ALPHA) / 2 * 100)
    hi = np.percentile(accs, (1 + ALPHA) / 2 * 100)
    return float((preds == labels).mean() * 100), float(lo * 100), float(hi * 100)


if __name__ == '__main__':
    results = {}
    for ds in ['casia2', 'columbia', 'augmented_coco']:
        print(f'[bootstrap] {ds}: 推理 + 重采样...')
        preds, labels = get_predictions(ds)
        acc, lo, hi = bootstrap_ci(preds, labels)
        results[ds] = {
            'n_test': int(len(labels)),
            'accuracy_pct': round(acc, 2),
            'ci95_lo_pct': round(lo, 2),
            'ci95_hi_pct': round(hi, 2),
            'n_bootstrap': B,
        }
        print(f'  acc={acc:.2f}%  95% CI=[{lo:.2f}, {hi:.2f}]')

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, 'w') as f:
        json.dump({'method': 'bootstrap (B=1000, alpha=0.95)',
                   'results': results}, f, indent=2)
    print(f'\n[保存] {OUT_FILE}')
