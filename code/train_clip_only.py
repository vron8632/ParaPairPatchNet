#!/usr/bin/env python3
"""
CLIP-only 3 类分类实验 — ForensicGuard
================================================
验证 Table 1 中 "CLIP-only + MLP (global-pooled, 512-d)" 在 3 类
(Clean / Digital / Physical) 协议下的真实表现。

方法:
  - 每图 CLIP 特征 = 所有实例的 CLIP embedding 平均池化 (512-d)
    (与论文 "global average-pooled" 一致; K=0 时用零向量)
  - 3 类 MLP: 512 → 128 → 3, CrossEntropy
  - 报告每类 (Clean/Digital/Physical) 的 test 准确率 + 整体

输出:
  results/clip_only_experiment.json — 完整结果
  results/logs/clip_only.log        — 训练日志 (含 ETA)
  results/models/clip_only/fusion_best.pth
"""
import sys, os, time, json, torch, torch.nn as nn
import numpy as np

BASE = '/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard'
sys.path.insert(0, os.path.join(BASE, 'code'))
FEATURE_DIR = os.path.join(BASE, 'results/features')
LOG_DIR = os.path.join(BASE, 'results/logs')
OUT_JSON = os.path.join(BASE, 'results/clip_only_experiment.json')
MODEL_DIR = os.path.join(BASE, 'results/models/clip_only')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
DOMAINS = ['clean', 'digital_patch', 'physical_patch']  # 0,1,2

def log(msg):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(f'{LOG_DIR}/clip_only.log', 'a') as f:
        f.write(line + '\n')

def fmt_eta(s):
    s = int(max(0, s)); h, r = divmod(s, 3600); m, s2 = divmod(r, 60)
    return f'{h:02d}:{m:02d}:{s2:02d}'

def load_pooled(domain, split):
    """加载 clip 特征并全局平均池化 -> (n, 512) + labels"""
    d = torch.load(f'{FEATURE_DIR}/{domain}_{split}_features.pt', map_location='cpu', weights_only=False)
    clips, labels = d['clip_feats'], list(d['labels'])
    pooled = []
    for cf in clips:
        if cf is None or len(cf) == 0:
            pooled.append(torch.zeros(512))
        else:
            pooled.append(cf.float().mean(0))
    X = torch.stack(pooled)
    y = torch.tensor([labels[i] for i in range(len(labels))])  # 0/1, tamper 域
    return X, y

def load_all(split):
    """三域合并, 标签 0/1/2"""
    Xs, ys = [], []
    for di, dom in enumerate(DOMAINS):
        X, y = load_pooled(dom, split)
        Xs.append(X); ys.append(y + di if False else y)  # 保留域标签
    # 实际: clean 标签0, digital/physical 标签1(原标签) -> 需转3类
    return None

def load_3class(split):
    """每域单独标签: clean=0, digital=1, physical=2"""
    Xs, ys = [], []
    for di, dom in enumerate(DOMAINS):
        X, y = load_pooled(dom, split)
        Xs.append(X); ys.append(torch.full_like(y, di, dtype=torch.long))
    return torch.cat(Xs), torch.cat(ys)

class ClipOnlyMLP(nn.Module):
    def __init__(self, in_dim=512, hidden=128, n_cls=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, n_cls))
    def forward(self, x):
        return self.net(x)

def main():
    os.makedirs(LOG_DIR, exist_ok=True); os.makedirs(MODEL_DIR, exist_ok=True)
    open(f'{LOG_DIR}/clip_only.log', 'w').close()
    log('=== CLIP-only 3 类分类实验 ===')
    log(f'device={device}')

    X_tr, y_tr = load_3class('train')
    X_va, y_va = load_3class('val')
    X_te, y_te = load_3class('test')
    n_tr = len(y_tr)
    log(f'train={n_tr} (clean={sum(y_tr==0)}, digital={sum(y_tr==1)}, physical={sum(y_tr==2)})')
    log(f'val={len(y_va)}, test={len(y_te)}')

    model = ClipOnlyMLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    best_acc, best_clean, best_path = 0.0, 0.0, f'{MODEL_DIR}/fusion_best.pth'
    t0 = time.time()
    for epoch in range(1, 201):
        model.train(); total_loss = 0.0
        idx = torch.randperm(n_tr)
        for st in range(0, n_tr, 64):
            en = min(st + 64, n_tr); ids = idx[st:en]
            xb = X_tr[ids].to(device); yb = y_tr[ids].to(device)
            out = model(xb)
            loss = crit(out, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * len(ids)
        if epoch % 10 == 0 or epoch == 200:
            model.eval()
            with torch.no_grad():
                pred = model(X_va.to(device)).argmax(1).cpu()
            vacc = (pred == y_va).float().mean().item() * 100
            # 每类 val acc
            cls_acc = {}
            for di, dom in enumerate(DOMAINS):
                m = (y_va == di)
                cls_acc[dom] = (pred[m] == y_va[m]).float().mean().item() * 100 if m.sum() > 0 else 0
            if vacc > best_acc:
                best_acc = vacc
                torch.save(model.state_dict(), best_path)
            el = time.time() - t0
            eta = fmt_eta(el / epoch * (200 - epoch))
            log(f'  Epoch {epoch}/200: loss={total_loss/n_tr*64:.4f} val_acc={vacc:.1f}% '
                f'val_clean={cls_acc["clean"]:.1f}% ETA={eta}')

    # test 评估
    model.load_state_dict(torch.load(best_path, map_location='cpu', weights_only=True))
    model = model.to(device).eval()
    with torch.no_grad():
        pred = model(X_te.to(device)).argmax(1).cpu()
    overall = (pred == y_te).float().mean().item() * 100
    per_class, pred_tampered = {}, {}
    for di, dom in enumerate(DOMAINS):
        m = (y_te == di)
        per_class[dom] = round((pred[m] == y_te[m]).float().mean().item() * 100, 1)
        pred_tampered[dom] = round((pred[m] != di).float().mean().item() * 100, 1)  # 判错比例
    log(f'\n=== Test 结果 ===')
    log(f'整体准确率: {overall:.1f}%')
    for dom in DOMAINS:
        log(f'  {dom:<15}: acc={per_class[dom]}%  判错={pred_tampered[dom]}%')

    result = {'protocol': '3-class (Clean/Digital/Physical), global-pooled CLIP 512-d',
              'train': {'clean': int((y_tr==0).sum()), 'digital': int((y_tr==1).sum()),
                        'physical': int((y_tr==2).sum())},
              'test_overall_acc_pct': round(overall, 1),
              'test_per_class_acc_pct': per_class,
              'test_per_class_misclass_pct': pred_tampered,
              'random_baseline_3class_pct': 33.3,
              'model': best_path}
    with open(OUT_JSON, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    log(f'\n已保存: {OUT_JSON}')

if __name__ == '__main__':
    main()
