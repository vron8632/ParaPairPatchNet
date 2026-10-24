#!/usr/bin/env python3
"""
通用训练+评估 — 支持任意特征组合，报告 operating point
========================================================
任务: clean(0) vs patch(1)。默认训练 clean+digital，测试 clean+physical（公平零样本设定）。
可自由组合特征: dinov2_cls / dinov2_pmean / dinov2_pstd / clip / spn / dct / dinov2_pt（patch-token 拼接）

用法:
  PY=/home/oyp/miniconda3/envs/apjf/bin/python
  $PY code/experiments_tvc/train_eval.py --base <SRC> --features dinov2_cls clip \
      --train-domains clean digital_patch --test-pos physical_patch --tag dino+clip
输出: results/tvc/train_eval_<tag>.json
"""
import argparse, json, os, sys
import numpy as np

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)
SPLITS = ["train", "val", "test"]


def _load_feat(base, dom, sp, name):
    """返回 per-sample 特征矩阵或 None。"""
    import torch
    fd = os.path.join(base, "results", "features")
    if name.startswith("dinov2"):
        p = os.path.join(fd, f"dinov2_{dom}_{sp}.pt")
        if not os.path.exists(p):
            return None
        d = torch.load(p, map_location="cpu", weights_only=False)
        key = {"dinov2_cls": "cls", "dinov2_pmean": "patch_mean",
               "dinov2_pstd": "patch_std"}[name]
        return d[key].float().numpy()
    p = os.path.join(fd, f"{dom}_{sp}_features.pt")
    if not os.path.exists(p):
        return None
    d = torch.load(p, map_location="cpu", weights_only=False)
    src = {"spn": "spn_feats", "dct": "dct_feats", "clip": "clip_feats"}[name]
    out = []
    for i in range(len(d["labels"])):
        out.append(d[src][i].mean(0).numpy().astype(np.float32))
    return np.stack(out)


def load_domains(base, dom, sp, names):
    mats, y = [], None
    import torch
    for nm in names:
        m = _load_feat(base, dom, sp, nm)
        if m is None:
            raise FileNotFoundError(f"缺少特征 {nm} for {dom}/{sp}")
        mats.append(m)
    # label
    if names[0].startswith("dinov2"):
        d = torch.load(os.path.join(base, "results", "features", f"dinov2_{dom}_{sp}.pt"),
                       map_location="cpu", weights_only=False)
        y = np.array(d["labels"])
    else:
        d = torch.load(os.path.join(base, "results", "features", f"{dom}_{sp}_features.pt"),
                       map_location="cpu", weights_only=False)
        y = np.array(d["labels"])
    X = np.concatenate(mats, axis=1).astype(np.float32)
    return X, y


def build(base, names, dom, sp):
    if not names:
        return None, None
    return load_domains(base, dom, sp, names)


def build_extra_clean(base, names):
    """clean_extra_train.pt 只有 spn/dct/clip（无 dinov2）。"""
    import torch
    p = os.path.join(base, "results", "features", "clean_extra_train_features.pt")
    if not os.path.exists(p):
        return None, None
    if any(n.startswith("dinov2") for n in names):
        return None, None
    d = torch.load(p, map_location="cpu", weights_only=False)
    mats = []
    for nm in names:
        src = {"spn": "spn_feats", "dct": "dct_feats", "clip": "clip_feats"}[nm]
        mats.append(np.stack([d[src][i].mean(0).numpy().astype(np.float32)
                              for i in range(len(d["labels"]))]))
    return np.concatenate(mats, 1), np.array(d["labels"])


def standardize(Xtr, Xs):
    mu = Xtr.mean(0, keepdims=True); sd = Xtr.std(0, keepdims=True) + 1e-6
    return (Xtr - mu) / sd, [(X - mu) / sd for X in Xs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--features", nargs="+", required=True)
    ap.add_argument("--train-domains", nargs="+", default=["clean", "digital_patch"])
    ap.add_argument("--test-pos", default="physical_patch")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--extra-clean", action="store_true",
                    help="把 clean_extra_train (1400 张) 加入训练负样本")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tr_X, tr_y = [], []
    for dom in args.train_domains:
        X, y = build(args.base, args.features, dom, "train")
        tr_X.append(X); tr_y.append(y)
    if args.extra_clean:
        Xe, ye = build_extra_clean(args.base, args.features)
        if Xe is not None:
            tr_X.append(Xe); tr_y.append(ye)
            print(f"[{args.tag}] + clean_extra_train: {len(ye)} 张")
    va_X, va_y = [], []
    for dom in args.train_domains:
        X, y = build(args.base, args.features, dom, "val")
        va_X.append(X); va_y.append(y)
    te_X = [build(args.base, args.features, "clean", "test")[0],
            build(args.base, args.features, args.test_pos, "test")[0]]
    te_y = [build(args.base, args.features, "clean", "test")[1],
            build(args.base, args.features, args.test_pos, "test")[1]]

    Xtr = np.concatenate(tr_X); ytr = np.concatenate(tr_y)
    Xva = np.concatenate(va_X); yva = np.concatenate(va_y)
    Xte = np.concatenate(te_X); yte = np.concatenate(te_y)
    Xtr, (Xva, Xte) = standardize(Xtr, [Xva, Xte])

    def to_t(a): return torch.tensor(a).to(device)
    Xtr_t, ytr_t = to_t(Xtr), to_t(ytr.astype(np.float32))
    Xva_t, Xte_t = to_t(Xva), to_t(Xte)

    model = nn.Sequential(
        nn.Linear(Xtr.shape[1], args.hidden), nn.ReLU(), nn.Dropout(args.dropout),
        nn.Linear(args.hidden, 64), nn.ReLU(), nn.Linear(64, 1)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lf = nn.BCEWithLogitsLoss()
    best = None
    for ep in range(args.epochs):
        model.train(); opt.zero_grad()
        lf(model(Xtr_t).squeeze(1), ytr_t).backward(); opt.step()

    model.eval()
    with torch.no_grad():
        logit_te = model(Xte_t).squeeze(1)
        logit_va = model(Xva_t).squeeze(1)
    # 温度校准：在 val 上最小化 NLL（只调一个标量 T，不改变排序 => AUC 不变，但改善 FPR/TPR 工作点）
    yva_t = to_t(yva.astype(np.float32))
    T = torch.nn.Parameter(torch.ones(1, device=device))
    optT = torch.optim.LBFGS([T], lr=0.1, max_iter=50)

    def _closure():
        optT.zero_grad()
        loss = lf(logit_va / T, yva_t)
        loss.backward()
        return loss

    optT.step(_closure)
    Tval = float(T.detach().item())
    with torch.no_grad():
        p = torch.sigmoid(logit_te / Tval).cpu().numpy()

    order = np.argsort(np.concatenate([p[yte == 1], p[yte == 0]]))
    pos, neg = p[yte == 1], p[yte == 0]
    r = np.empty(len(order)); r[order] = np.arange(1, len(order) + 1)
    aucv = (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

    # 用负类分数的分位数作为阈值，覆盖完整 ROC（固定 τ 扫描在概率集中时会失效）
    thresholds = np.unique(np.percentile(p[yte == 0], np.linspace(0, 100, 201)))
    sweep = []
    for t in thresholds:
        pred = p >= t
        sweep.append({"tau": round(float(t), 4),
                      "tpr": round(float((pred & (yte == 1)).sum() / max((yte == 1).sum(), 1)), 4),
                      "fpr": round(float((pred & (yte == 0)).sum() / max((yte == 0).sum(), 1)), 4)})
    rep = {"tag": args.tag, "features": args.features,
           "train_domains": args.train_domains, "test_pos": args.test_pos,
           "n_train": int(len(ytr)), "n_test": int(len(yte)),
           "dim": int(Xtr.shape[1]), "auc": round(float(aucv), 4), "sweep": sweep}
    for b in (0.01, 0.05, 0.10, 0.20):
        cand = [s for s in sweep if s["fpr"] <= b]
        rep[f"best_tpr_at_fpr<={int(b*100)}pct"] = max(cand, key=lambda s: s["tpr"]) if cand else None

    out = args.out or os.path.join(os.path.dirname(CODE_DIR), "results", "tvc",
                                   f"train_eval_{args.tag}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print(f"[{args.tag}] dim={rep['dim']} AUC={rep['auc']}  " +
          "  ".join(f"TPR@{k.split('<=')[1]}={rep[k]['tpr'] if rep[k] else None}"
                    for k in ("best_tpr_at_fpr<=1pct", "best_tpr_at_fpr<=5pct",
                              "best_tpr_at_fpr<=10pct", "best_tpr_at_fpr<=20pct")))
    print(f"         wrote {out}")


if __name__ == "__main__":
    main()
