#!/usr/bin/env python3
"""
跨数据集泛化实验（回应"只在单数据集上验证"的致命质疑）
========================================================
三个方向:
  (a) in-domain      : COCO 训练 -> COCO 测试（同域上界参考）
  (b) CASIA -> COCO  : CASIA 衍生训练 -> COCO 补丁测试（跨数据集，重要对照）
  (c) COCO -> CASIA  : 反向

特征: clip/spn/dct（patch_dataset 与 coco_patch 均已提取，格式一致）
训练: clean+digital_patch 的 train split；测试: clean vs physical_patch 的 test split
评估: 负类分位数扫 ROC，报 TPR@FPR∈{1,5,10,20}%

用法:
  $PY code/experiments_tvc/cross_dataset.py --mode both
输出: results/tvc/cross_dataset.json
"""
import argparse, json, os, sys
import numpy as np

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)
SRC = "/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard"
FEAT = os.path.join(SRC, "results", "features")
OUT = os.path.join(os.path.dirname(CODE_DIR), "results", "tvc")
BRANCHES = ("clip", "spn", "dct")


def load(feat_dir, dom, split, branches=BRANCHES):
    import torch
    p = os.path.join(feat_dir, f"{dom}_{split}_features.pt")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    d = torch.load(p, map_location="cpu", weights_only=False)
    mats = []
    for b in branches:
        src = {"clip": "clip_feats", "spn": "spn_feats", "dct": "dct_feats"}[b]
        mats.append(np.stack([d[src][i].mean(0).numpy().astype(np.float32)
                              for i in range(len(d["labels"]))]))
    return np.concatenate(mats, 1), np.array(d["labels"])


def train_and_eval(train_dir, test_dir, tag, device, epochs=400, seed=0):
    import torch
    import torch.nn as nn

    trX, trY = [], []
    for dom in ("clean", "digital_patch"):
        X, y = load(train_dir, dom, "train")
        trX.append(X); trY.append(y)
    Xtr = np.concatenate(trX); ytr = np.concatenate(trY)
    teX, teY = [], []
    for dom in ("clean", "physical_patch"):
        X, y = load(test_dir, dom, "test")
        teX.append(X); teY.append(y)
    Xte = np.concatenate(teX); yte = np.concatenate(teY)

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd

    torch.manual_seed(seed)
    m = nn.Sequential(nn.Linear(Xtr.shape[1], 128), nn.ReLU(),
                      nn.Dropout(0.3), nn.Linear(128, 1)).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    lf = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(Xtr).to(device); yt = torch.tensor(ytr, dtype=torch.float32).to(device)
    for _ in range(epochs):
        m.train(); opt.zero_grad()
        lf(m(Xt).squeeze(1), yt).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        p = torch.sigmoid(m(torch.tensor(Xte).to(device)).squeeze(1)).cpu().numpy()

    pos, neg = yte == 1, yte == 0
    order = np.argsort(np.concatenate([p[pos], p[neg]]))
    r = np.empty(len(order)); r[order] = np.arange(1, len(order) + 1)
    auc = (r[:pos.sum()].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum())
    th = np.unique(np.percentile(p[neg], np.linspace(0, 100, 201)))
    sweep = [{"tau": round(float(t), 4),
              "tpr": round(float(((p >= t) & pos).sum() / pos.sum()), 4),
              "fpr": round(float(((p >= t) & neg).sum() / neg.sum()), 4)} for t in th]
    rep = {"tag": tag, "n_train": int(len(ytr)), "n_test": int(len(yte)),
           "n_pos": int(pos.sum()), "n_neg": int(neg.sum()),
           "auc": round(float(auc), 4), "sweep": sweep}
    for b in (0.01, 0.05, 0.10, 0.20):
        cand = [s for s in sweep if s["fpr"] <= b]
        rep[f"best_tpr_at_fpr<={int(b*100)}pct"] = max(cand, key=lambda s: s["tpr"]) if cand else None
    print(f"[{tag}] AUC={rep['auc']} n_test={rep['n_test']} "
          + "  ".join(f"TPR@{k.split('<=')[1]}="
                      f"{rep[k]['tpr'] if rep[k] else '-'}"
                      for k in ("best_tpr_at_fpr<=1pct", "best_tpr_at_fpr<=5pct",
                                "best_tpr_at_fpr<=10pct", "best_tpr_at_fpr<=20pct")))
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["in_coco", "casia2coco", "coco2casia",
                                       "both", "loo"], default="both")
    args = ap.parse_args()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    doms = {"CASIA": FEAT,
            "COCO": os.path.join(FEAT, "coco_patch"),
            "INRIA": os.path.join(FEAT, "inria_patch")}

    if args.mode == "loo":
        # 三数据集 leave-one-out: 单源 -> 各目标域，并含多源联合
        res = {}
        for src in doms:
            for tgt in doms:
                if src == tgt:
                    continue
                tag = f"cross:{src}->{tgt}"
                res[tag.replace(":", "_").replace("->", "_to_")] = train_and_eval(
                    doms[src], doms[tgt], tag, device)
        # 多源(其余两个) -> 第三个
        for tgt in doms:
            srcs = [doms[s] for s in doms if s != tgt]

            def train_multi(train_dirs, test_dir, tag, device, epochs=400, seed=0):
                import torch.nn as nn
                trX, trY = [], []
                for td in train_dirs:
                    for dom in ("clean", "digital_patch"):
                        X, y = load(td, dom, "train")
                        trX.append(X); trY.append(y)
                Xtr = np.concatenate(trX); ytr = np.concatenate(trY)
                teX, teY = [], []
                for dom in ("clean", "physical_patch"):
                    X, y = load(test_dir, dom, "test")
                    teX.append(X); teY.append(y)
                Xte = np.concatenate(teX); yte = np.concatenate(teY)
                mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
                Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
                torch.manual_seed(seed)
                m = nn.Sequential(nn.Linear(Xtr.shape[1], 128), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(128, 1)).to(device)
                opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
                lf = nn.BCEWithLogitsLoss()
                Xt = torch.tensor(Xtr).to(device)
                yt = torch.tensor(ytr, dtype=torch.float32).to(device)
                for _ in range(epochs):
                    m.train(); opt.zero_grad()
                    lf(m(Xt).squeeze(1), yt).backward(); opt.step()
                m.eval()
                with torch.no_grad():
                    p = torch.sigmoid(m(torch.tensor(Xte).to(device)).squeeze(1)).cpu().numpy()
                pos, neg = yte == 1, yte == 0
                order = np.argsort(np.concatenate([p[pos], p[neg]]))
                rk = np.empty(len(order)); rk[order] = np.arange(1, len(order) + 1)
                a = (rk[:pos.sum()].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum())
                th = np.unique(np.percentile(p[neg], np.linspace(0, 100, 201)))
                sweep = [{"tau": round(float(t), 4),
                          "tpr": round(float(((p >= t) & pos).sum() / pos.sum()), 4),
                          "fpr": round(float(((p >= t) & neg).sum() / neg.sum()), 4)} for t in th]
                rep = {"tag": tag, "auc": round(float(a), 4), "n_test": int(len(yte))}
                for b in (0.01, 0.05, 0.10, 0.20):
                    c = [s for s in sweep if s["fpr"] <= b]
                    rep[f"best_tpr_at_fpr<={int(b*100)}pct"] = max(c, key=lambda s: s["tpr"]) if c else None
                print(f"[{tag}] AUC={rep['auc']} " + "  ".join(
                    f"TPR@{k.split('<=')[1]}={rep[k]['tpr'] if rep[k] else '-'}"
                    for k in ("best_tpr_at_fpr<=5pct", "best_tpr_at_fpr<=20pct")))
                return rep

            res[f"multi_to_{tgt}"] = train_multi(
                srcs, doms[tgt], f"multi->{tgt}", device)
        out = os.path.join(OUT, "cross_dataset_loo.json")
        json.dump(res, open(out, "w"), indent=2, ensure_ascii=False)
        print(f"[cross] wrote {out}")
        return

    casia = os.path.join(FEAT)                       # patch_dataset_v2 的 features 平铺在这里
    coco = os.path.join(FEAT, "coco_patch")
    res = {}
    if args.mode in ("in_coco", "both"):
        res["in_domain_coco"] = train_and_eval(coco, coco, "in_domain:COCO->COCO", device)
    if args.mode in ("casia2coco", "both"):
        res["casia_to_coco"] = train_and_eval(casia, coco, "cross:CASIA->COCO", device)
    if args.mode in ("coco2casia", "both"):
        res["coco_to_casia"] = train_and_eval(coco, casia, "cross:COCO->CASIA", device)

    out = os.path.join(OUT, "cross_dataset.json")
    json.dump(res, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"[cross] wrote {out}")


if __name__ == "__main__":
    main()
