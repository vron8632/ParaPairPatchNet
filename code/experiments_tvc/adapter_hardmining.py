#!/usr/bin/env python3
"""
Adapter + 难例挖掘 综合训练评估
==============================
在缓存特征之上加一个 bottleneck 残差 Adapter（可学习变换），并用 Focal Loss 做难例挖掘，
对照基线 MLP，报告真实 operating point（用负类分位数扫阈值，覆盖完整 ROC）。

Adapter 结构:  x -> Linear(d,r) -> GELU -> Linear(r,d) -> 残差相加 -> 头
  r = d//ratio (默认 ratio=8)

用法:
  $PY code/experiments_tvc/adapter_hardmining.py --base <SRC> \
      --features clip spn dct --extra-clean --focal --ratio 8 --tag adapter_focal
输出: results/tvc/adapter_<tag>.json
"""
import argparse, json, os, sys
import numpy as np

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)


def build(base, names, dom, sp):
    import torch
    fd = os.path.join(base, "results", "features")
    mats = []
    for nm in names:
        if nm.startswith("dinov2"):
            p = os.path.join(fd, f"dinov2_{dom}_{sp}.pt")
            d = torch.load(p, map_location="cpu", weights_only=False)
            mats.append(d[{"dinov2_cls": "cls", "dinov2_pmean": "patch_mean",
                           "dinov2_pstd": "patch_std"}[nm]].float().numpy())
        else:
            p = os.path.join(fd, f"{dom}_{sp}_features.pt")
            d = torch.load(p, map_location="cpu", weights_only=False)
            src = {"spn": "spn_feats", "dct": "dct_feats", "clip": "clip_feats"}[nm]
            mats.append(np.stack([d[src][i].mean(0).numpy().astype(np.float32)
                                  for i in range(len(d["labels"]))]))
    # 标签来源：任一文件即可（同域同 split 标签一致）
    if names[0].startswith("dinov2"):
        lp = os.path.join(fd, f"dinov2_{dom}_{sp}.pt")
    else:
        lp = os.path.join(fd, f"{dom}_{sp}_features.pt")
    labels = torch.load(lp, map_location="cpu", weights_only=False)["labels"]
    return np.concatenate(mats, 1).astype(np.float32), np.array(labels)


def build_extra_clean(base, names):
    import torch
    if any(n.startswith("dinov2") for n in names):
        return None, None
    p = os.path.join(base, "results", "features", "clean_extra_train_features.pt")
    if not os.path.exists(p):
        return None, None
    d = torch.load(p, map_location="cpu", weights_only=False)
    mats = []
    for nm in names:
        src = {"spn": "spn_feats", "dct": "dct_feats", "clip": "clip_feats"}[nm]
        mats.append(np.stack([d[src][i].mean(0).numpy().astype(np.float32)
                              for i in range(len(d["labels"]))]))
    return np.concatenate(mats, 1).astype(np.float32), np.array(d["labels"])


class AdapterHead:
    def __init__(self, d, ratio=8, device="cuda"):
        import torch.nn as nn
        self.device = device
        r = max(8, d // ratio)
        self.net = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, r), nn.GELU(), nn.Linear(r, d), nn.Tanh(),
        ).to(device)
        self.head = nn.Sequential(
            nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 1)).to(device)
        self.params = list(self.net.parameters()) + list(self.head.parameters())

    def __call__(self, x):
        return self.head(x + self.net(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--features", nargs="+", required=True)
    ap.add_argument("--extra-clean", action="store_true")
    ap.add_argument("--focal", action="store_true")
    ap.add_argument("--ratio", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--tag", default="adapter")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    device = "cuda" if torch.cuda.is_available() else "cpu"

    trX, trY = [], []
    for dom in ["clean", "digital_patch"]:
        X, y = build(args.base, args.features, dom, "train")
        trX.append(X); trY.append(y)
    if args.extra_clean:
        Xe, ye = build_extra_clean(args.base, args.features)
        if Xe is not None:
            trX.append(Xe); trY.append(ye)
            print(f"[{args.tag}] + extra clean {len(ye)}")
    Xtr = np.concatenate(trX); ytr = np.concatenate(trY)

    Xva_c, yva_c = build(args.base, args.features, "clean", "val")
    Xva_p, yva_p = build(args.base, args.features, "physical_patch", "val")
    Xte_c, yte_c = build(args.base, args.features, "clean", "test")
    Xte_p, yte_p = build(args.base, args.features, "physical_patch", "test")
    Xva = np.concatenate([Xva_c, Xva_p]); yva = np.concatenate([yva_c, yva_p])
    Xte = np.concatenate([Xte_c, Xte_p]); yte = np.concatenate([yte_c, yte_p])

    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd; Xva = (Xva - mu) / sd; Xte = (Xte - mu) / sd

    to = lambda a: torch.tensor(a).to(device)
    Xtr_t, ytr_t = to(Xtr), to(ytr.astype(np.float32))
    Xva_t, Xte_t = to(Xva), to(Xte)
    yva_t = to(yva.astype(np.float32))

    model = AdapterHead(Xtr.shape[1], args.ratio, device)
    opt = torch.optim.AdamW(model.params, lr=args.lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    def loss_fn(lg, y):
        if not args.focal:
            return bce(lg, y).mean()
        p = torch.sigmoid(lg)
        pt = p * y + (1 - p) * (1 - y)
        return (bce(lg, y) * (1 - pt) ** args.gamma).mean()

    best_va, best_state = float("inf"), None
    for ep in range(args.epochs):
        model.net.train(); model.head.train(); opt.zero_grad()
        loss_fn(model(Xtr_t).squeeze(1), ytr_t).backward(); opt.step()
        if (ep + 1) % 50 == 0:
            model.net.eval(); model.head.eval()
            with torch.no_grad():
                lv = loss_fn(model(Xva_t).squeeze(1), yva_t).item()
            if lv < best_va:
                best_va = lv
                best_state = ({k: v.detach().clone() for k, v in model.net.state_dict().items()},
                              {k: v.detach().clone() for k, v in model.head.state_dict().items()})

    if best_state is not None:
        model.net.load_state_dict(best_state[0])
        model.head.load_state_dict(best_state[1])

    # 温度校准 (val, 不改变排序)
    model.net.eval(); model.head.eval()
    with torch.no_grad():
        logit_va = model(Xva_t).squeeze(1)
        logit_te = model(Xte_t).squeeze(1)
    T = torch.nn.Parameter(torch.ones(1, device=device))
    optT = torch.optim.LBFGS([T], lr=0.1, max_iter=50)
    bce_s = nn.BCEWithLogitsLoss()

    def _c():
        optT.zero_grad()
        l = bce_s(logit_va / T, yva_t); l.backward(); return l
    optT.step(_c)
    with torch.no_grad():
        p = torch.sigmoid(logit_te / float(T.detach())).cpu().numpy()

    pos, neg = yte == 1, yte == 0
    order = np.argsort(np.concatenate([p[pos], p[neg]]))
    r = np.empty(len(order)); r[order] = np.arange(1, len(order) + 1)
    aucv = (r[:pos.sum()].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum())

    th = np.unique(np.percentile(p[neg], np.linspace(0, 100, 201)))
    sweep = [{"tau": round(float(t), 4),
              "tpr": round(float(((p >= t) & pos).sum() / max(pos.sum(), 1)), 4),
              "fpr": round(float(((p >= t) & neg).sum() / max(neg.sum(), 1)), 4)}
             for t in th]
    rep = {"tag": args.tag, "features": args.features, "focal": args.focal,
           "ratio": args.ratio, "extra_clean": args.extra_clean,
           "n_train": int(len(ytr)), "n_test": int(len(yte)), "dim": int(Xtr.shape[1]),
           "auc": round(float(aucv), 4), "temperature": round(float(T.detach()), 4),
           "sweep": sweep}
    for b in (0.01, 0.05, 0.10, 0.20):
        cand = [s for s in sweep if s["fpr"] <= b]
        rep[f"best_tpr_at_fpr<={int(b*100)}pct"] = max(cand, key=lambda s: s["tpr"]) if cand else None

    out = args.out or os.path.join(os.path.dirname(CODE_DIR), "results", "tvc",
                                   f"adapter_{args.tag}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print(f"[{args.tag}] AUC={rep['auc']} T={rep['temperature']}")
    for k in ("best_tpr_at_fpr<=1pct", "best_tpr_at_fpr<=5pct",
              "best_tpr_at_fpr<=10pct", "best_tpr_at_fpr<=20pct"):
        print(f"   {k}: {rep[k]}")
    print(f"   wrote {out}")


if __name__ == "__main__":
    main()
