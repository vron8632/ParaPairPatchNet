#!/usr/bin/env python3
"""
直接检验 cross-instance consistency 是否真的有用（回应 R#6/R#1）
================================================================
在 K>=2 子集（consistency 统计量唯一生效的场景）上，对比：
  (a) global-only      : [g]  661 维（不含一致性统计量）
  (b) global+cons      : [g;c;s] 698 维（论文完整模型）
若 (a) >= (b)，说明 consistency 统计量不但没帮助、反而有害——机制失效；反之则机制成立。
同时在 K<=1 上做同样对比作为对照（那里 consistency 本应无效）。

用法:
  $PY code/experiments_tvc/test_consistency_value.py --base <SRC> --dataset casia2 --model casia2
输出: results/tvc/consistency_value_<dataset>.json
"""
import argparse, json, os, sys
import numpy as np

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--dataset", default="casia2")
    ap.add_argument("--model", default="casia2")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    from models.consistency import CrossInstanceConsistencyV2
    from models.fusion_classifier import FusionClassifierV2
    from scripts.train import compute_fused_features

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fd = os.path.join(args.base, "results", "features")
    data = torch.load(os.path.join(fd, f"{args.dataset}_{args.split}_features.pt"),
                      map_location="cpu", weights_only=False)
    s, d, c, y = data["spn_feats"], data["dct_feats"], data["clip_feats"], np.array(data["labels"])
    ic = np.array(data["instance_counts"]) if data.get("instance_counts") else None
    n = len(y)

    # 计算两种特征: global-only(661) 与 full(698)
    cic = CrossInstanceConsistencyV2()
    from scripts.train import compute_fused_features
    Xfull, Xglob = [], []
    with torch.no_grad():
        for i in range(n):
            f = compute_fused_features([s[i]], [d[i]], [c[i]], cic, device)  # (1,698)
            Xfull.append(f.cpu())
            Xglob.append(f[:, 37:].cpu())   # 列切片: 698-37=661 -> [g(661)]
    Xfull = torch.cat(Xfull); Xglob = torch.cat(Xglob)

    # 两组掩码
    masks = {"K>=2": ic >= 2, "K<=1": ic <= 1, "all": np.ones(n, bool)}
    if ic is None:
        masks = {"all": np.ones(n, bool)}

    def train_eval(X, mask, seed=0):
        idx = np.where(mask)[0]
        if len(idx) < 20:
            return None
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(idx))
        ntr = int(len(perm) * 0.7)
        tr, te = idx[perm[:ntr]], idx[perm[ntr:]]
        Xtr = X[tr].to(device); ytr = torch.tensor(y[tr], dtype=torch.float32).to(device)
        Xte = X[te].to(device); yte = y[te]
        model = nn.Sequential(nn.Linear(X.shape[1], 128), nn.ReLU(),
                              nn.Linear(128, 1)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        lf = nn.BCEWithLogitsLoss()
        for _ in range(400):
            model.train(); opt.zero_grad()
            lf(model(Xtr).squeeze(1), ytr).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(Xte).squeeze(1)).cpu().numpy()
        pos, neg = yte == 1, yte == 0
        if pos.sum() == 0 or neg.sum() == 0:
            return {"n": int(len(yte)), "auc": None}
        order = np.argsort(np.concatenate([p[pos], p[neg]]))
        r = np.empty(len(order)); r[order] = np.arange(1, len(order) + 1)
        auc = (r[:pos.sum()].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum())
        pred = p >= 0.5
        return {"n": int(len(yte)), "n_pos": int(pos.sum()), "n_neg": int(neg.sum()),
                "auc": round(float(auc), 4),
                "acc@0.5": round(float((pred == yte).mean()), 4),
                "tpr@0.5": round(float((pred & pos).sum() / pos.sum()), 4),
                "fpr@0.5": round(float((pred & neg).sum() / neg.sum()), 4)}

    rep = {"dataset": args.dataset, "model": args.model,
           "n_total": int(n),
           "k_dist": ({str(k): int(v) for k, v in
                       zip(*np.unique(ic, return_counts=True))} if ic is not None else None),
           "per_group": {}}
    for gname, m in masks.items():
        a = train_eval(Xglob, m)
        b = train_eval(Xfull, m)
        if a and b:
            rep["per_group"][gname] = {"global_only": a, "global+consistency": b,
                                       "delta_auc": round(b["auc"] - a["auc"], 4),
                                       "delta_acc": round(b["acc@0.5"] - a["acc@0.5"], 4)}
            print(f"[{gname}] n={a['n']} (pos={a['n_pos']},neg={a['n_neg']})")
            print(f"    global-only    AUC={a['auc']}  acc={a['acc@0.5']}  TPR={a['tpr@0.5']} FPR={a['fpr@0.5']}")
            print(f"    +consistency   AUC={b['auc']}  acc={b['acc@0.5']}  TPR={b['tpr@0.5']} FPR={b['fpr@0.5']}")
            print(f"    delta AUC={rep['per_group'][gname]['delta_auc']}  delta acc={rep['per_group'][gname]['delta_acc']}")

    out = args.out or os.path.join(os.path.dirname(CODE_DIR), "results", "tvc",
                                   f"consistency_value_{args.dataset}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print(f"[cons-value] wrote {out}")


if __name__ == "__main__":
    main()
