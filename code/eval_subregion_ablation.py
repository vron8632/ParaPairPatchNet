#!/usr/bin/env python3
"""
Sub-region (24-d) ablation on CASIA v2 — mirrors run_all.sh stage 6 protocol.

Variant A: Full model (698-d = 13 consistency + 24 sub-region + 661 global).
Variant B: w/o sub-region (674-d = 13 consistency + 661 global).

Both use the exact stage-6 recipe: FocalLoss(alpha=0.75, gamma=2.0),
AdamW(lr=1e-4, wd=1e-4), 300 epochs, batch 64, grad clip 1.0,
best-val-acc checkpoint selection, test evaluation with fusion_best.
Seeds are fixed for fair comparison (stage 6 itself was unseeded).
"""
import sys, os, json, torch, torch.nn as nn
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from models.consistency import CrossInstanceConsistencyV2
from models.fusion_classifier import FusionClassifierV2, FocalLoss
from scripts.train import compute_fused_features, evaluate

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
FEATURE_DIR = os.path.join(BASE, 'results/features')
OUT_JSON = os.path.join(BASE, 'results/subregion_ablation.json')

torch.manual_seed(42)
import numpy as np
np.random.seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cic = CrossInstanceConsistencyV2()


def load(split):
    d = torch.load(f'{FEATURE_DIR}/casia2_{split}_features.pt', map_location='cpu', weights_only=False)
    return d['spn_feats'], d['dct_feats'], d['clip_feats'], d['labels']


s_tr, d_tr, c_tr, l_tr = load('train')
s_val, d_val, c_val, l_val = load('val')
s_te, d_te, c_te, l_te = load('test')
n_tr = len(l_tr)
print(f'train={n_tr} val={len(l_val)} test={len(l_te)}')


def fused_nosub(batch_s, batch_d, batch_c, device):
    """13-d consistency + 661-d global, without the 24-d sub-region vector."""
    out = []
    for i in range(len(batch_s)):
        stats, _, _, _ = cic.compute_consistency(batch_s[i], batch_d[i], batch_c[i])
        g = cic.compute_global_features(batch_s[i], batch_d[i], batch_c[i])
        out.append(torch.cat([stats.to(device), g.to(device)]))
    return torch.stack(out)


def evaluate_nosub(model, s_data, d_data, c_data, labels, device, threshold=0.5):
    """evaluate() with the 674-d fused features (no 24-d sub-region vector)."""
    model.eval()
    n = len(labels)
    all_preds, all_probas = [], []
    with torch.no_grad():
        for i in range(n):
            fused = fused_nosub([s_data[i]], [d_data[i]], [c_data[i]], device)
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


def run(tag, use_subregion):
    torch.manual_seed(42)
    np.random.seed(42)
    input_dim = 698 if use_subregion else 674
    model = FusionClassifierV2(input_dim=input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    best_acc, best_state = 0.0, None
    for epoch in range(300):
        model.train()
        total_loss = 0.0
        indices = torch.randperm(n_tr)
        for start in range(0, n_tr, 64):
            end = min(start + 64, n_tr)
            idx = indices[start:end]
            bs, bd, bc = [s_tr[i] for i in idx], [d_tr[i] for i in idx], [c_tr[i] for i in idx]
            by = torch.tensor([l_tr[i] for i in idx], dtype=torch.float32, device=device)
            fused = compute_fused_features(bs, bd, bc, cic, device) if use_subregion else fused_nosub(bs, bd, bc, device)
            logits = model(fused).squeeze(-1)
            loss = criterion(logits, by)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(idx)
        if (epoch + 1) % 10 == 0:
            m = evaluate(model, s_val, d_val, c_val, l_val, cic, device) if use_subregion else evaluate_nosub(model, s_val, d_val, c_val, l_val, device)
            if m['accuracy'] > best_acc:
                best_acc = m['accuracy']
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f'  [{tag}] Epoch {epoch+1}/300: loss={total_loss/n_tr*64:.4f}, val_acc={m["accuracy"]*100:.1f}%')
    model.load_state_dict(best_state)
    model.eval()
    mt = evaluate(model, s_te, d_te, c_te, l_te, cic, device) if use_subregion else evaluate_nosub(model, s_te, d_te, c_te, l_te, device)
    print(f'=== {tag}: Test Acc={mt["accuracy"]*100:.2f}% F1={mt["f1"]*100:.2f}% (best_val={best_acc*100:.2f}%)')
    return {'tag': tag, 'input_dim': input_dim, 'best_val_acc_pct': round(best_acc*100, 2),
            'test_acc_pct': round(mt['accuracy']*100, 2), 'test_f1_pct': round(mt['f1']*100, 2),
            'tp': int(mt['tp']), 'fp': int(mt['fp']), 'tn': int(mt['tn']), 'fn': int(mt['fn'])}


if __name__ == '__main__':
    results = [run('full_698d', use_subregion=True), run('wo_subregion_674d', use_subregion=False)]
    with open(OUT_JSON, 'w') as f:
        json.dump({'protocol': 'stage6 (FocalLoss a=0.75, 300ep, lr=1e-4, b64, seed=42)',
                   'results': results}, f, indent=1)
    print(f'\nsaved -> {OUT_JSON}')
