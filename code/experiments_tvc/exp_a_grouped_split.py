#!/usr/bin/env python3
"""
TVC 补实验 A — 用背景指纹恢复 base 分组 + grouped vs random split
==================================================================
动机: 经核实 (run_all.sh stage1):
  * clean 域 = 前 1000 张 base；digital 域 = 同一批 base + patch；physical = 另 1000 张 base。
  * 三域各自 random.shuffle 后 60/20/20 划分 => clean/digital 共享 base 却独立划分。
  * 原始特征不含 base_id，无法事后分组。

由于环境**无网络**（CLIP 权重不可下载，无法重提特征），本脚本改用**背景指纹**恢复 base 分组：
  同一 base 的 clean/digital/physical 共享背景（patch 只占 ~6%），降采样指纹高度相似。
  patch_dataset_v2/{domain}_{split}.pt 与 features/{domain}_{split}_features.pt 顺序一致，
  因此指纹分组可直接映射到已缓存的特征上。

用法:
  PY=/home/oyp/miniconda3/envs/apjf/bin/python
  $PY code/experiments_tvc/exp_a_grouped_split.py --stage finger --base <SRC>
  $PY code/experiments_tvc/exp_a_grouped_split.py --stage eval   --base <SRC>
输出: results/tvc/grouped_split.json
"""
import argparse, json, os, sys
import numpy as np

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)
DOMAINS = ["clean", "digital_patch", "physical_patch"]
SPLITS = ["train", "val", "test"]


def fingerprint(t):
    import torch.nn.functional as F
    g = t.mean(0, keepdim=True).unsqueeze(0)               # (1,1,H,W)
    s = F.interpolate(g, size=(32, 32), mode="area")[0, 0]  # (32,32)
    v = s.flatten().numpy().astype(np.float32)
    v = v - v.mean()
    n = np.linalg.norm(v) + 1e-8
    return v / n                                           # cosine-ready


def do_finger(args):
    import torch
    pd = os.path.join(args.base, "results", "features", "patch_dataset_v2")
    out = {}
    metas = []
    for dom in DOMAINS:
        for sp in SPLITS:
            p = os.path.join(pd, f"{dom}_{sp}.pt")
            if not os.path.exists(p):
                print(f"[finger] 缺少 {p}，跳过"); continue
            d = torch.load(p, map_location="cpu", weights_only=False)
            for i, t in enumerate(d["tensors"]):
                out[f"{dom}|{sp}|{i}"] = fingerprint(t)
                metas.append((dom, sp, i))
            print(f"[finger] {dom}/{sp}: {len(d['tensors'])}")
    fp = np.stack([out[f"{d}|{s}|{i}"] for (d, s, i) in metas])
    np.save(os.path.join(args.base, "results", "tvc", "fingerprints.npy"), fp)
    with open(os.path.join(args.base, "results", "tvc", "fingerprints_meta.json"), "w") as f:
        json.dump(metas, f)
    print(f"[finger] saved {fp.shape}")


def _load_fp(base):
    fp = np.load(os.path.join(base, "results", "tvc", "fingerprints.npy"))
    meta = json.load(open(os.path.join(base, "results", "tvc", "fingerprints_meta.json")))
    return fp, meta


def cluster(fp, thresh):
    """union-find：cosine 相似度 > thresh 视为同一 base。"""
    n = len(fp)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    sim = fp @ fp.T  # cosine（向量已归一化）
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] > thresh:
                union(i, j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return groups


def do_eval(args):
    import torch
    import torch.nn as nn
    from models.consistency import CrossInstanceConsistencyV2
    from models.fusion_classifier import FusionClassifierV2
    from scripts.train import compute_fused_features

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp, meta = _load_fp(args.base)
    print(f"[eval] {len(meta)} 样本, 聚类阈值={args.thresh}")
    groups = cluster(fp, args.thresh)
    print(f"[eval] 得到 {len(groups)} 个 base 组; 组大小分布: "
          f"{dict(sorted({k2: sum(1 for v in groups.values() if len(v)==k2) for k2 in set(len(v) for v in groups.values())}.items()))}")

    # 索引 -> 组 id
    idx2group = {}
    for gid, members in enumerate(groups.values()):
        for m in members:
            idx2group[m] = gid
    # 泄漏统计：组内出现 >1 个 split
    leak = 0
    for members in groups.values():
        splits = {meta[m][1] for m in members}
        if len(splits) > 1:
            leak += 1
    print(f"[eval] 跨 split 的 base 组数 = {leak} / {len(groups)}")

    # 特征加载
    feat = {}
    for dom in DOMAINS:
        for sp in SPLITS:
            p = os.path.join(args.base, "results", "features", f"{dom}_{sp}_features.pt")
            d = torch.load(p, map_location="cpu", weights_only=False)
            feat[(dom, sp)] = d

    cic = CrossInstanceConsistencyV2()

    def make_xy(assign_fn, tag):
        Xs, Xd, Xc, y, info = [], [], [], [], []
        for (dom, sp), d in feat.items():
            for i in range(len(d["labels"])):
                m = meta.index([dom, sp, i]) if False else None
        # 用 meta 反查索引
        key2idx = {tuple(m): k for k, m in enumerate(meta)}
        for k, (dom, sp, i) in enumerate(meta):
            d = feat[(dom, sp)]
            want = assign_fn(k)
            Xs.append(d["spn_feats"][i]); Xd.append(d["dct_feats"][i])
            Xc.append(d["clip_feats"][i]); y.append(d["labels"][i])
            info.append((k, want))
        # 重新按 want 分组
        parts = {"train": [], "val": [], "test": []}
        for k, (dom, sp, i) in enumerate(meta):
            parts[assign_fn(k)].append(k)
        return parts

    # random split：对每个域内部随机划分（复现原设置）
    rng = np.random.RandomState(args.seed)
    key2idx = {tuple(m): k for k, m in enumerate(meta)}
    domain_idx = {dom: [key2idx[(dom, sp, i)] for sp in SPLITS for i in range(len(feat[(dom, sp)]["labels"]))]
                  for dom in DOMAINS}
    assign_random = {}
    for dom, idxs in domain_idx.items():
        perm = rng.permutation(len(idxs))
        ntr, nva = int(len(idxs) * 0.6), int(len(idxs) * 0.2)
        for j, pos in enumerate(perm):
            assign_random[idxs[pos]] = "train" if j < ntr else ("val" if j < ntr + nva else "test")

    # grouped split：base 组级别划分
    gids = list(range(len(groups)))
    rng2 = np.random.RandomState(args.seed)
    perm = rng2.permutation(len(gids))
    ntr, nva = int(len(gids) * 0.6), int(len(gids) * 0.2)
    gid2split = {}
    for j, g in enumerate(perm):
        gid2split[gids[g]] = "train" if j < ntr else ("val" if j < ntr + nva else "test")
    assign_grouped = {k: gid2split[idx2group[k]] for k in range(len(meta))}

    def build_and_train(assign, tag):
        parts = {"train": [], "val": [], "test": []}
        for k, (dom, sp, i) in enumerate(meta):
            parts[assign[k]].append((dom, sp, i))
        def featurize(sel):
            feats, ys = [], []
            for (dom, sp, i) in sel:
                d = feat[(dom, sp)]
                feats.append(compute_fused_features([d["spn_feats"][i]], [d["dct_feats"][i]],
                                                    [d["clip_feats"][i]], cic, device))
                ys.append(d["labels"][i])
            return torch.cat(feats, 0), torch.tensor(ys, dtype=torch.float32)
        Xtr, ytr = featurize(parts["train"]); Xva, yva = featurize(parts["val"])
        Xte, yte = featurize(parts["test"])
        model = FusionClassifierV2(input_dim=698).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        lossf = nn.BCEWithLogitsLoss()
        Xtr, ytr = Xtr.to(device), ytr.to(device)
        for _ in range(200):
            model.train(); opt.zero_grad()
            loss = lossf(model(Xtr).squeeze(1), ytr); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(Xte.to(device)).squeeze(1)).cpu().numpy()
        y = yte.numpy(); pos, neg = y == 1, y == 0
        sweep = []
        for t in np.arange(0.05, 1.0, 0.05):
            pred = p >= t
            sweep.append({"tau": round(float(t), 2),
                          "tpr": round(float((pred & pos).sum() / max(pos.sum(), 1)), 4),
                          "fpr": round(float((pred & neg).sum() / max(neg.sum(), 1)), 4)})
        rep = {"tag": tag, "n_train": len(ytr), "n_test": len(y), "sweep": sweep}
        for b in (0.05, 0.10, 0.20):
            cand = [r for r in sweep if r["fpr"] <= b]
            rep[f"best_tpr_at_fpr<={int(b*100)}pct"] = max(cand, key=lambda r: r["tpr"]) if cand else None
        print(f"[{tag}] train={rep['n_train']} test={rep['n_test']} "
              f"(pos={int(pos.sum())},neg={int(neg.sum())})")
        for kk in ("best_tpr_at_fpr<=5pct", "best_tpr_at_fpr<=10pct", "best_tpr_at_fpr<=20pct"):
            print(f"    {kk}: {rep[kk]}")
        return rep

    rep_r = build_and_train(assign_random, "random(per-domain, 原设置)")
    rep_g = build_and_train(assign_grouped, "grouped(base-level, 无泄漏)")

    report = {"n_samples": len(meta), "n_groups": len(groups), "cross_split_leak_groups": leak,
              "thresh": args.thresh, "random": rep_r, "grouped": rep_g}
    out = os.path.join(args.base, "results", "tvc", "grouped_split.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[eval] wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["finger", "eval"], required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--thresh", type=float, default=0.995)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.base, "results", "tvc"), exist_ok=True)
    if args.stage == "finger":
        do_finger(args)
    else:
        do_eval(args)


if __name__ == "__main__":
    main()
