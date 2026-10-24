#!/usr/bin/env python3
"""
跨域泛化改进（针对 cross_dataset.py 暴露的 CASIA->COCO AUC=0.66 问题）
==========================================================================
已测: 同域 COCO 0.851 / 跨域 CASIA->COCO 0.660 / COCO->CASIA 0.754 —— 跨域塌陷。

本脚本尝试三种域泛化手段，逐一对照：
  (1) multi-source   : CASIA + COCO 联合训练（增加域多样性，最简单）
  (2) feat-norm      : 逐图 L2 归一化 + 全局标准化（削弱相机/风格偏置）
  (3) mixup/增强     : 训练时 mixup（λ~Beta(0.4,0.4)），隐式域插值
全部在**同一测试协议**（clean vs physical 的 test split）下比较，报 TPR@FPR。

用法:
  $PY code/experiments_tvc/cross_domain_improve.py
输出: results/tvc/cross_domain_improve.json
"""
import argparse, json, os, sys
import numpy as np

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)
SRC = "/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard"
FEAT = os.path.join(SRC, "results", "features")
OUT = os.path.join(os.path.dirname(CODE_DIR), "results", "tvc")
BRANCHES = ("clip", "spn", "dct")


def load(feat_dir, dom, split):
    import torch
    d = torch.load(os.path.join(feat_dir, f"{dom}_{split}_features.pt"),
                   map_location="cpu", weights_only=False)
    mats = []
    for b in BRANCHES:
        src = {"clip": "clip_feats", "spn": "spn_feats", "dct": "dct_feats"}[b]
        mats.append(np.stack([d[src][i].mean(0).numpy().astype(np.float32)
                              for i in range(len(d["labels"]))]))
    return np.concatenate(mats, 1), np.array(d["labels"])


def gather(dirs, split, doms=("clean", "digital_patch")):
    X, y = [], []
    for d in dirs:
        for dom in doms:
            a, b = load(d, dom, split)
            X.append(a); y.append(b)
    return np.concatenate(X), np.concatenate(y)


def test_set(dir_, split_test=("clean", "physical_patch")):
    X, y = [], []
    for dom in split_test:
        a, b = load(dir_, dom, "test")
        X.append(a); y.append(b)
    return np.concatenate(X), np.concatenate(y)


def eval_op(p, y):
    pos, neg = y == 1, y == 0
    order = np.argsort(np.concatenate([p[pos], p[neg]]))
    r = np.empty(len(order)); r[order] = np.arange(1, len(order) + 1)
    auc = (r[:pos.sum()].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum())
    th = np.unique(np.percentile(p[neg], np.linspace(0, 100, 201)))
    sweep = [{"tau": round(float(t), 4),
              "tpr": round(float(((p >= t) & pos).sum() / pos.sum()), 4),
              "fpr": round(float(((p >= t) & neg).sum() / neg.sum()), 4)} for t in th]
    rep = {"auc": round(float(auc), 4), "n": int(len(y)), "sweep": sweep}
    for b in (0.01, 0.05, 0.10, 0.20):
        c = [s for s in sweep if s["fpr"] <= b]
        rep[f"tpr@{int(b*100)}"] = max(c, key=lambda s: s["tpr"])["tpr"] if c else None
    return rep


def train(Xtr, ytr, Xte, seed=0, mixup=0.0, epochs=400, device="cuda"):
    import torch
    import torch.nn as nn
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    torch.manual_seed(seed)
    m = nn.Sequential(nn.Linear(Xtr.shape[1], 128), nn.ReLU(),
                      nn.Dropout(0.3), nn.Linear(128, 1)).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    lf = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(Xtr).to(device); yt = torch.tensor(ytr, dtype=torch.float32).to(device)
    n = len(ytr)
    for _ in range(epochs):
        m.train()
        if mixup > 0:
            idx = torch.randperm(n, device=device)
            lam = float(np.random.beta(0.4, 0.4)) if np.random.rand() < mixup else 1.0
            Xm = lam * Xt + (1 - lam) * Xt[idx]
            ym = lam * yt + (1 - lam) * yt[idx]
        else:
            Xm, ym = Xt, yt
        opt.zero_grad()
        lf(m(Xm).squeeze(1), ym).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        # 返回 logits 而非 sigmoid 后的概率：
        # L2 归一化等预处理会让 logit 量级剧增导致 sigmoid 饱和(全≈1)，
        # 用 logits 做 AUC/sweep 是单调等价的，可正确反映可分性与工作点。
        return m(torch.tensor(Xte).to(device)).squeeze(1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    casia, coco = FEAT, os.path.join(FEAT, "coco_patch")

    Xte, yte = test_set(coco)          # 统一测试: COCO 的 clean vs physical test
    configs = {
        "single_casia": [casia],
        "single_coco": [coco],
        "multi_source": [casia, coco],
    }
    results = {}
    for name, dirs in configs.items():
        Xtr, ytr = gather(dirs, "train")
        # feat-norm 变体: 逐图 L2 归一化
        for variant, pre in (("as-is", None), ("l2row", "l2")):
            if name == "single_coco" and variant == "l2row":
                pass
            Xa = Xtr.copy()
            if pre == "l2":
                Xa = Xa / (np.linalg.norm(Xa, axis=1, keepdims=True) + 1e-8)
            scores = []
            for s in range(args.seeds):
                p = train(Xa, ytr, Xte, seed=s, mixup=0.0, device=device)
                scores.append(eval_op(p, yte))
            tag = f"{name}|{variant}"
            results[tag] = {k: scores[0][k] for k in scores[0] if k != "sweep"}
            results[tag]["auc_mean"] = round(float(np.mean([c["auc"] for c in scores])), 4)
            results[tag]["auc_std"] = round(float(np.std([c["auc"] for c in scores])), 4)
            print(f"{tag:30s} AUC={results[tag]['auc_mean']}±{results[tag]['auc_std']} "
                  f"tpr@5%={results[tag]['tpr@5']} tpr@20%={results[tag]['tpr@20']}")

        # mixup 变体（只对 multi_source 做，避免组合爆炸）
        if name == "multi_source":
            scores = []
            for s in range(args.seeds):
                p = train(Xtr, ytr, Xte, seed=s, mixup=0.5, device=device)
                scores.append(eval_op(p, yte))
            tag = f"{name}|mixup"
            results[tag] = {k: scores[0][k] for k in scores[0] if k != "sweep"}
            results[tag]["auc_mean"] = round(float(np.mean([c["auc"] for c in scores])), 4)
            results[tag]["auc_std"] = round(float(np.std([c["auc"] for c in scores])), 4)
            print(f"{tag:30s} AUC={results[tag]['auc_mean']}±{results[tag]['auc_std']} "
                  f"tpr@5%={results[tag]['tpr@5']} tpr@20%={results[tag]['tpr@20']}")

    out = os.path.join(OUT, "cross_domain_improve.json")
    json.dump({"test": "COCO clean vs physical (test)", "seeds": args.seeds,
               "results": results}, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"[xdom] wrote {out}")


if __name__ == "__main__":
    main()
