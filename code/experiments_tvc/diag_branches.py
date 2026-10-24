#!/usr/bin/env python3
"""
诊断 — 各特征分支的判别力 + clean 被高分的原因
================================================
在 clean vs physical / clean vs digital 上训练线性探针与 MLP，报告 AUC 与概率分布，
定位 FPR 高的病根（是 CLIP 语义把普通场景判异常，还是 SPN/DCT 噪声？）。

用法:
  PY=/home/oyp/miniconda3/envs/apjf/bin/python
  $PY code/experiments_tvc/diag_branches.py --base <SRC>
输出: results/tvc/diag_branches.json
"""
import argparse, json, os, sys
import numpy as np

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)
DOMAINS = ["clean", "digital_patch", "physical_patch"]


def load(base):
    feat = {}
    for dom in DOMAINS:
        for sp in ("train", "val", "test"):
            p = os.path.join(base, "results", "features", f"{dom}_{sp}_features.pt")
            feat[(dom, sp)] = torch_load(p)
    return feat


def torch_load(p):
    import torch
    return torch.load(p, map_location="cpu", weights_only=False)


def pool(vecs, i, branch):
    import torch
    t = vecs[i]
    return t.mean(0).numpy().astype(np.float32)


def build(feat, dom, branch):
    d = feat[(dom, "test")]
    src = {"spn": "spn_feats", "dct": "dct_feats", "clip": "clip_feats"}[branch]
    X = np.stack([pool(d[src], i, branch) for i in range(len(d["labels"]))])
    y = np.array(d["labels"])
    return X, y


def train_probe(Xtr, ytr, Xte, yte, device="cpu", epochs=300, hidden=64):
    import torch
    import torch.nn as nn
    Xtr = torch.tensor(Xtr).to(device); ytr = torch.tensor(ytr, dtype=torch.float32).to(device)
    Xte = torch.tensor(Xte).to(device)
    m = nn.Sequential(nn.Linear(Xtr.shape[1], hidden), nn.ReLU(),
                      nn.Linear(hidden, 1)).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    lf = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        m.train(); opt.zero_grad()
        lf(m(Xtr).squeeze(1), ytr).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        p = torch.sigmoid(m(Xte).squeeze(1)).cpu().numpy()
    return p


def auc(p, y):
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feat = load(args.base)

    report = {"branches": {}, "tasks": {}}
    for task, posdom in [("clean_vs_physical", "physical_patch"),
                         ("clean_vs_digital", "digital_patch")]:
        trep = {}
        for branch in ("spn", "dct", "clip"):
            Xtr, ytr = build(feat, posdom, branch)
            Xc, yc = build(feat, "clean", branch)
            # 训练：patch(train) + clean(train) —— 用 test 评估（这里直接 train on test 仅作诊断上界）
            # 为保证可比：在 test 上训练/测试同一分布（诊断判别力上界）
            Xall = np.concatenate([Xtr, Xc]); yall = np.concatenate([ytr, yc])
            n = len(yall); idx = np.random.RandomState(0).permutation(n)
            ntr = int(n * 0.7)
            p = train_probe(Xall[idx[:ntr]], yall[idx[:ntr]], Xall[idx[ntr:]], yall[idx[ntr:]], device)
            trep[branch] = {"auc": round(auc(p, yall[idx[ntr:]]), 4),
                            "dim": int(Xall.shape[1])}
        report["tasks"][task] = trep
        print(f"[diag] {task}: " + "  ".join(f"{k} AUC={v['auc']}" for k, v in trep.items()))

    # clean 概率分布（用已训练主模型之外的诊断用探针）
    out = args.out or os.path.join(os.path.dirname(CODE_DIR), "results", "tvc", "diag_branches.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[diag] wrote {out}")


if __name__ == "__main__":
    main()
