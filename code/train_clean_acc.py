#!/usr/bin/env python3
"""
Clean Accuracy 提升实验 — ForensicGuard
================================================
目标: 通过不同训练配置提高 Clean Acc（当前基线 ~29%，FPR ~70%）。

实验矩阵:
  E0 baseline  : clean+digital 训练, BCE, clean权=1   (复现当前基线)
  E1 clean优先 : clean+digital 训练, BCE, clean权=2
  E2 +物理域   : clean+digital+physical 训练, BCE, clean权=2
  E3 强clean   : clean+digital+physical 训练, BCE, clean权=4
  E4 focal     : clean+digital+physical 训练, Focal(gamma=2), clean权=2

输出:
  results/clean_acc_experiments.json  — 汇总结果
  results/logs/clean_acc_E{N}.log     — 每实验详细日志
  results/models/clean_acc/E{N}/fusion_best.pth — 每实验最佳模型

终端输出: 每 10 epoch 打印 loss/val指标/剩余时分秒(ETA)
"""
import sys, os, time, json, torch, torch.nn as nn
import numpy as np

BASE = '/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard'
sys.path.insert(0, os.path.join(BASE, 'code'))
FEATURE_DIR = os.path.join(BASE, 'results/features')
LOG_DIR = os.path.join(BASE, 'results/logs')
MODEL_ROOT = os.path.join(BASE, 'results/models/clean_acc')
OUT_JSON = os.path.join(BASE, 'results/clean_acc_experiments.json')

from models.consistency import CrossInstanceConsistencyV2
from models.fusion_classifier import FusionClassifierV2
from scripts.train import compute_fused_features

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'[env] device={device}', flush=True)

# ───────────────────────── 工具函数 ─────────────────────────
def load_domain(domain, split):
    d = torch.load(f'{FEATURE_DIR}/{domain}_{split}_features.pt', map_location='cpu', weights_only=False)
    return d['spn_feats'], d['dct_feats'], d['clip_feats'], d['labels']

def merge(domains, split):
    ss, ds, cs, ls = [], [], [], []
    for dom in domains:
        s, d, c, l = load_domain(dom, split)
        ss.extend(s); ds.extend(d); cs.extend(c); ls.extend(l)
    return ss, ds, cs, ls

def eval_domains(model, cic, domains, split='test'):
    """对每个域评估: acc / FPR@tau / TPR@tau"""
    res = {}
    model.eval()
    with torch.no_grad():
        for dom in domains:
            s, d, c, l = load_domain(dom, split)
            preds, probs = [], []
            for i in range(len(l)):
                fused = compute_fused_features([s[i]], [d[i]], [c[i]], cic, device)
                p = torch.sigmoid(model(fused)).item()
                probs.append(p); preds.append(1 if p > 0.5 else 0)
            acc = np.mean([p == y for p, y in zip(preds, l)])
            res[dom] = {
                'acc_pct': round(100*acc, 2),
                'n': len(l),
                'pred_tampered_pct': round(100*np.mean(preds), 2),
            }
    return res

def eval_auc_fpr(model, cic):
    """clean vs (digital+physical) 二分类: FPR@tau=0.5, AUC"""
    model.eval()
    ys, probs = [], []
    with torch.no_grad():
        for dom, lab in [('clean', 0), ('digital_patch', 1), ('physical_patch', 1)]:
            s, d, c, l = load_domain(dom, 'test')
            for i in range(len(l)):
                fused = compute_fused_features([s[i]], [d[i]], [c[i]], cic, device)
                probs.append(torch.sigmoid(model(fused)).item())
                ys.append(lab)
    ys = np.array(ys); probs = np.array(probs)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(ys, probs)
    fpr = np.mean(probs[ys == 0] > 0.5)
    tpr = np.mean(probs[ys == 1] > 0.5)
    return {'auc': round(float(auc), 4), 'fpr_tau05_pct': round(100*fpr, 2),
            'tpr_tau05_pct': round(100*tpr, 2)}

def fmt_eta(secs):
    secs = int(max(0, secs))
    h, rem = divmod(secs, 3600); m, s = divmod(rem, 60)
    return f'{h:02d}:{m:02d}:{s:02d}'

# ───────────────────────── 训练 ─────────────────────────
def train_experiment(exp_id, cfg, logf):
    """训练一个配置, 返回评估结果"""
    t0 = time.time()
    logf(f'\n===== {exp_id}: {cfg["desc"]} =====')
    logf(f'训练域={cfg["train_domains"]} 权重(clean,tamper)={cfg["w_clean"]},{cfg["w_tamper"]} '
         f'loss={cfg["loss"]} epochs={cfg["epochs"]}')

    # 数据
    s_tr, d_tr, c_tr, l_tr = merge(cfg['train_domains'], 'train')
    s_val, d_val, c_val, l_val = merge(['clean', 'digital_patch', 'physical_patch'], 'val')
    n_tr = len(l_tr)
    logf(f'训练样本: {n_tr}, 验证样本: {len(l_val)}')

    cic = CrossInstanceConsistencyV2()
    model = FusionClassifierV2(input_dim=698).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])

    # loss
    if cfg['loss'] == 'focal':
        def criterion(logits, y, w):
            p = torch.sigmoid(logits)
            gamma = 2.0
            pt = torch.where(y > 0.5, p, 1 - p)
            wv = torch.where(y > 0.5, torch.tensor(cfg['w_tamper'], device=device),
                             torch.tensor(cfg['w_clean'], device=device))
            return (-wv * (1 - pt) ** gamma * torch.log(pt + 1e-8)).mean()
    else:
        def criterion(logits, y, w):
            return nn.functional.binary_cross_entropy_with_logits(
                logits, y, weight=w, pos_weight=torch.tensor(
                    cfg['w_tamper'] / cfg['w_clean'] if cfg['w_tamper'] > cfg['w_clean'] else 1.0,
                    device=device))

    best_val, best_clean = 0.0, 0.0
    for epoch in range(1, cfg['epochs'] + 1):
        model.train(); total_loss = 0.0
        indices = torch.randperm(n_tr)
        for start in range(0, n_tr, 64):
            end = min(start + 64, n_tr); idx = indices[start:end]
            bs = [s_tr[i] for i in idx]; bd = [d_tr[i] for i in idx]
            bc = [c_tr[i] for i in idx]
            by = torch.tensor([l_tr[i] for i in idx], dtype=torch.float32, device=device)
            w = torch.tensor([cfg['w_clean'] if l_tr[i] == 0 else cfg['w_tamper'] for i in idx],
                             dtype=torch.float32, device=device)
            fused = compute_fused_features(bs, bd, bc, cic, device)
            logits = model(fused).squeeze(-1)
            loss = criterion(logits, by, w)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total_loss += loss.item() * len(idx)

        if epoch % 10 == 0 or epoch == cfg['epochs']:
            # 验证（分域）
            model.eval()
            vclean = eval_domains(model, cic, ['clean'], 'val')['clean']['acc_pct']
            voverall_c = 0
            with torch.no_grad():
                for i in range(len(l_val)):
                    fused = compute_fused_features([s_val[i]], [d_val[i]], [c_val[i]], cic, device)
                    pred = (torch.sigmoid(model(fused)) > 0.5).item()
                    voverall_c += (pred == l_val[i])
            voverall = 100 * voverall_c / len(l_val)
            if vclean > best_clean:
                best_clean = vclean
                torch.save(model.state_dict(), f'{MODEL_ROOT}/{exp_id}/fusion_best.pth')
            if voverall > best_val:
                best_val = voverall
            elapsed = time.time() - t0
            eta = fmt_eta(elapsed / epoch * (cfg['epochs'] - epoch))
            line = (f'  Epoch {epoch}/{cfg["epochs"]}: loss={total_loss/n_tr*64:.4f} '
                    f'val_acc={voverall:.1f}% val_clean={vclean:.1f}% '
                    f'ETA={eta} elapsed={fmt_eta(elapsed)}')
            print(line, flush=True); logf(line)

    # 评估最佳模型（val clean 最优）
    model.load_state_dict(torch.load(f'{MODEL_ROOT}/{exp_id}/fusion_best.pth',
                                     map_location='cpu', weights_only=False))
    model = model.to(device).eval()
    dom_res = eval_domains(model, cic, ['clean', 'digital_patch', 'physical_patch'], 'test')
    roc = eval_auc_fpr(model, cic)
    result = {'experiment': exp_id, 'config': {**cfg, 'train_domains': list(cfg['train_domains'])},
              'test': dom_res, 'roc': roc, 'best_val_clean_pct': round(best_clean, 2),
              'best_val_acc_pct': round(best_val, 2), 'total_secs': round(time.time()-t0, 1)}
    logf(f'→ test: {dom_res}')
    logf(f'→ roc: {roc}')
    return result

# ───────────────────────── 主程序 ─────────────────────────
EXPERIMENTS = [
    {'id': 'E0', 'desc': 'baseline clean+digital, BCE w=1:1',
     'train_domains': ['clean', 'digital_patch'], 'w_clean': 1, 'w_tamper': 1,
     'loss': 'bce', 'epochs': 300, 'lr': 1e-4, 'wd': 1e-4},
    {'id': 'E1', 'desc': 'clean 优先, clean+digital, BCE w=2:1',
     'train_domains': ['clean', 'digital_patch'], 'w_clean': 2, 'w_tamper': 1,
     'loss': 'bce', 'epochs': 300, 'lr': 1e-4, 'wd': 1e-4},
    {'id': 'E2', 'desc': '+物理域, BCE w=2:1',
     'train_domains': ['clean', 'digital_patch', 'physical_patch'], 'w_clean': 2, 'w_tamper': 1,
     'loss': 'bce', 'epochs': 300, 'lr': 1e-4, 'wd': 1e-4},
    {'id': 'E3', 'desc': '+物理域, 强clean, BCE w=4:1',
     'train_domains': ['clean', 'digital_patch', 'physical_patch'], 'w_clean': 4, 'w_tamper': 1,
     'loss': 'bce', 'epochs': 300, 'lr': 1e-4, 'wd': 1e-4},
    {'id': 'E4', 'desc': '+物理域, Focal gamma=2, w=2:1',
     'train_domains': ['clean', 'digital_patch', 'physical_patch'], 'w_clean': 2, 'w_tamper': 1,
     'loss': 'focal', 'epochs': 300, 'lr': 1e-4, 'wd': 1e-4},
]

def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_ROOT, exist_ok=True)
    all_results = []
    print(f'\n=== Clean Acc 提升实验 (共 {len(EXPERIMENTS)} 组) ===', flush=True)
    for cfg in EXPERIMENTS:
        exp_id = cfg['id']
        os.makedirs(f'{MODEL_ROOT}/{exp_id}', exist_ok=True)
        log_path = f'{LOG_DIR}/clean_acc_{exp_id}.log'
        with open(log_path, 'w') as lf:
            def logf(msg):
                lf.write(msg + '\n')
            res = train_experiment(exp_id, cfg, logf)
        all_results.append(res)
        print(f'  ✓ {exp_id} 完成: clean={res["test"]["clean"]["acc_pct"]}% '
              f'physical={res["test"]["physical_patch"]["acc_pct"]}% '
              f'FPR={res["roc"]["fpr_tau05_pct"]}% AUC={res["roc"]["auc"]}', flush=True)

    with open(OUT_JSON, 'w') as f:
        json.dump({'experiments': all_results}, f, indent=2, ensure_ascii=False)
    print(f'\n✅ 全部完成, 结果已保存: {OUT_JSON}', flush=True)
    print('\n=== 汇总 ===')
    print(f'{"ID":<4}{"desc":<38}{"Clean%":>8}{"Digital%":>9}{"Physical%":>10}{"FPR@.5%":>9}{"AUC":>7}')
    for r in all_results:
        print(f'{r["experiment"]:<4}{r["config"]["desc"]:<38}'
              f'{r["test"]["clean"]["acc_pct"]:>8.1f}'
              f'{r["test"]["digital_patch"]["acc_pct"]:>9.1f}'
              f'{r["test"]["physical_patch"]["acc_pct"]:>10.1f}'
              f'{r["roc"]["fpr_tau05_pct"]:>9.1f}'
              f'{r["roc"]["auc"]:>7.3f}')

if __name__ == '__main__':
    main()
