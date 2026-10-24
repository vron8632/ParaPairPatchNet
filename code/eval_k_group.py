#!/usr/bin/env python3
"""
K 分组评估 — ForensicGuard

对 CASIA v2 test 集按实例数 K 分组评估检测准确率，
验证跨实例一致性在不同 K 下的表现，并将结果存储为 JSON。

用法:
    python3 code/eval_k_group.py
输出:
    results/casia2_k_group_eval.json
"""
import sys, os, json, collections, torch

BASE = '/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard'
SCRIPT_DIR = os.path.join(BASE, 'code')
FEATURE_DIR = os.path.join(BASE, 'results/features')
MODEL_DIR = os.path.join(BASE, 'results/models')
OUT_FILE = os.path.join(BASE, 'results/casia2_k_group_eval.json')

sys.path.insert(0, SCRIPT_DIR)
from models.consistency import CrossInstanceConsistencyV2
from models.fusion_classifier import FusionClassifierV2
from scripts.train import compute_fused_features

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'[eval] device: {device}')


def evaluate(dataset='casia2'):
    """加载特征 + 模型，按 K 分组评估，返回结果 dict"""
    feat_path = f'{FEATURE_DIR}/{dataset}_test_features.pt'
    data = torch.load(feat_path, map_location='cpu', weights_only=False)
    s_te, d_te, c_te, l_te = (data['spn_feats'], data['dct_feats'],
                               data['clip_feats'], data['labels'])
    ic = data.get('instance_counts')
    n = len(l_te)
    print(f'[eval] {dataset}: {n} test samples, feature: {feat_path}')

    model_path = f'{MODEL_DIR}/{dataset}/fusion_best.pth'
    model = FusionClassifierV2(input_dim=698).to(device)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False))
    model.eval()
    print(f'[eval] model: {model_path}')

    cic = CrossInstanceConsistencyV2()

    # 分组: K -> [correct, total]
    grp = collections.defaultdict(lambda: [0, 0])
    # 统计每类的样本数和正确数 (0=Au, 1=Tp)
    cls = collections.defaultdict(lambda: [0, 0])

    with torch.no_grad():
        for i in range(n):
            spn, dct, clip = [s_te[i]], [d_te[i]], [c_te[i]]
            fused = compute_fused_features(spn, dct, clip, cic, device)
            proba = torch.sigmoid(model(fused)).item()
            pred = 1 if proba > 0.5 else 0
            K = ic[i] if ic else len(s_te[i])

            grp[K][1] += 1
            grp[K][0] += int(pred == l_te[i])
            cls[l_te[i]][1] += 1
            cls[l_te[i]][0] += int(pred == l_te[i])

    # 汇总
    def acc(c, t):
        return round(100.0 * c / t, 2) if t > 0 else None

    k0 = grp.get(0, [0, 0])
    k1 = grp.get(1, [0, 0])
    kge2_c = sum(v[0] for k, v in grp.items() if k >= 2)
    kge2_n = sum(v[1] for k, v in grp.items() if k >= 2)
    kle1_c = k0[0] + k1[0]
    kle1_n = k0[1] + k1[1]

    result = {
        'dataset': dataset,
        'n_samples': n,
        'overall': {'correct': sum(v[0] for v in grp.values()),
                    'total': n,
                    'accuracy_pct': acc(sum(v[0] for v in grp.values()), n)},
        'per_class': {
            'authentic': {'correct': cls[0][0], 'total': cls[0][1],
                          'accuracy_pct': acc(cls[0][0], cls[0][1])},
            'tampered': {'correct': cls[1][0], 'total': cls[1][1],
                         'accuracy_pct': acc(cls[1][0], cls[1][1])},
        },
        'k_groups': {
            'K=0': {'correct': k0[0], 'total': k0[1],
                    'accuracy_pct': acc(k0[0], k0[1]),
                    'proportion_pct': acc(k0[1], n)},
            'K=1': {'correct': k1[0], 'total': k1[1],
                    'accuracy_pct': acc(k1[0], k1[1]),
                    'proportion_pct': acc(k1[1], n)},
            'K>=2': {'correct': kge2_c, 'total': kge2_n,
                     'accuracy_pct': acc(kge2_c, kge2_n),
                     'proportion_pct': acc(kge2_n, n)},
            'K<=1': {'correct': kle1_c, 'total': kle1_n,
                     'accuracy_pct': acc(kle1_c, kle1_n),
                     'proportion_pct': acc(kle1_n, n)},
        },
        'k_detail': {
            str(k): {'correct': v[0], 'total': v[1],
                     'accuracy_pct': acc(v[0], v[1])}
            for k, v in sorted(grp.items())
        },
        'model': model_path,
        'features': feat_path,
    }

    # 打印
    print(f'\n=== {dataset} K 分组评估 ===')
    print(f'整体: {result["overall"]["correct"]}/{n} = {result["overall"]["accuracy_pct"]}%')
    for k in ['K=0', 'K=1', 'K>=2', 'K<=1']:
        g = result['k_groups'][k]
        print(f'  {k}: {g["correct"]}/{g["total"]} = {g["accuracy_pct"]}% (占比 {g["proportion_pct"]}%)')
    print(f'  Au: {cls[0][0]}/{cls[0][1]} = {acc(cls[0][0], cls[0][1])}%')
    print(f'  Tp: {cls[1][0]}/{cls[1][1]} = {acc(cls[1][0], cls[1][1])}%')

    return result


if __name__ == '__main__':
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    result = evaluate('casia2')
    with open(OUT_FILE, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f'\n[保存] 结果已存储: {OUT_FILE}')
