#!/usr/bin/env python3
"""
TVC 补实验 A — 重建带 base_id 的物理补丁数据集 + grouped vs random split
=========================================================================
动机: 同一 base image 的三版本是否跨 split（泄漏）。

现状（已核实 run_all.sh stage1）:
  * clean 域 = 前 1000 张 base 原图；digital 域 = 同一批 base 的 patch 版本；
    physical 域 = 另 1000 张 base 的 patch 版本。
  * 三个域各自 random.shuffle 后按 60/20/20 划分 —— clean 与 digital 共享 base，
    却独立划分 => 潜在的跨 split 泄漏，论文未声明。
  * 原始特征文件不含 paths/base_id，无法事后分组。

本脚本重建一个**受控**数据集：同一批 base image 生成 clean/digital/physical 三个版本，
全程记录 base_id，然后对比
  (a) random split  —— 复现原设置（可能有泄漏）
  (b) grouped split —— 同一 base 的三版本强制同 split（无泄漏）
在同一 MLP 协议下的性能差异，量化泄漏影响。

子命令:
  gen   : 生成 + 提特征（一步，边生成边提特征，避免存大图像）
  eval  : 两种划分下训练 + 评估，输出 operating point
  all   : gen + eval

用法:
  PY=/home/oyp/miniconda3/envs/apjf/bin/python
  $PY code/experiments_tvc/rebuild_grouped.py --stage gen --n 300
  $PY code/experiments_tvc/rebuild_grouped.py --stage eval
"""
import argparse, json, os, random, sys, time
import numpy as np

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)

DUETGUARD = "/media/oyp/工作/Projects/042_image_forensic/DuetGuard"
AU_DIR = f"{DUETGUARD}/data/casia2/CASIA2.0_revised/Au"
OUT_DIR = os.path.join(os.path.dirname(CODE_DIR), "results", "patch_grouped")
DOMAINS = ["clean", "digital_patch", "physical_patch"]


# ─────────────────────────── 生成 + 提特征 ───────────────────────────
def extract_one(seg, spn, dct, clip, img, device):
    """复现 scripts/extract_features.py 的逐实例特征提取。返回 (spn,dct,clip,K)。"""
    import torch
    masks = seg.segment(img)
    masks = [(m, b) for m, b in masks if m.sum() > 32 * 32]
    K = len(masks)
    if K < 2:
        img_d = img.to(device)
        return (spn.extract(img_d).unsqueeze(0).cpu(),
                dct.extract_profile(img).unsqueeze(0).cpu(),
                clip.encode_image(img).unsqueeze(0).cpu(), K)
    sl, dl, cl = [], [], []
    for mask, _ in masks:
        img_d = img.to(device)
        mask_t = torch.from_numpy(np.asarray(mask)).float().to(device)
        sl.append(spn.extract_instance(img_d, mask_t).cpu())
        dl.append(dct.extract_profile(img, mask).cpu())
        cl.append(clip.encode_image(img, mask).cpu())
    return torch.stack(sl), torch.stack(dl), torch.stack(cl), K


def do_gen(args):
    import torch
    import torchvision.transforms as T
    from PIL import Image
    from features.segmenter import InstanceSegmenter
    from features.spn_extractor import get_spn_extractor
    from features.dct_profile import DCTProfileExtractor
    from features.clip_encoder import get_clip_extractor
    from scripts.patch_attack import AdversarialPatchGeneratorV2
    from scripts.pc_simulation import PCSimulation

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) 选 base image（CASIA Au，固定 seed）
    imgs = sorted(f for f in os.listdir(AU_DIR)
                  if f.lower().endswith((".jpg", ".jpeg", ".png")))
    rng = random.Random(args.seed)
    rng.shuffle(imgs)
    imgs = imgs[:args.n]
    print(f"[gen] 选定 {len(imgs)} 张 base image (seed={args.seed})")

    gen = AdversarialPatchGeneratorV2(patch_size=128, device=device)
    pc = PCSimulation(severity="strong")
    seg = InstanceSegmenter(device=device)
    spn = get_spn_extractor(mode="learned", device=device)
    dct = DCTProfileExtractor()
    clip = get_clip_extractor(device=device)
    tf = T.Compose([T.Resize((512, 512)), T.ToTensor()])

    acc = {d: {"spn": [], "dct": [], "clip": [], "labels": [], "base_ids": [], "K": []}
           for d in DOMAINS}
    t0 = time.time()
    for i, fn in enumerate(imgs):
        try:
            img = tf(Image.open(os.path.join(AU_DIR, fn)).convert("RGB"))
        except Exception as e:
            print(f"  跳过 {fn}: {e}")
            continue
        mask, patch_content = gen.generate_strong_pgd_patch(
            img, loss_type="color", strategy="random")
        img_digital = gen.apply_patch(img, mask, patch_content)
        patch_phys = pc.simulate(patch_content)
        img_physical = gen.apply_patch(img, mask, patch_phys)
        variants = {"clean": img, "digital_patch": img_digital,
                    "physical_patch": img_physical}
        for dom in DOMAINS:
            s, d, c, K = extract_one(seg, spn, dct, clip, variants[dom], device)
            acc[dom]["spn"].append(s); acc[dom]["dct"].append(d)
            acc[dom]["clip"].append(c); acc[dom]["K"].append(K)
            acc[dom]["labels"].append(0 if dom == "clean" else 1)
            acc[dom]["base_ids"].append(fn)
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  [{i+1}/{len(imgs)}] {el:.0f}s, ETA {el/(i+1)*(len(imgs)-i-1):.0f}s")

    for dom in DOMAINS:
        p = os.path.join(OUT_DIR, f"{dom}_features.pt")
        torch.save({"spn_feats": acc[dom]["spn"], "dct_feats": acc[dom]["dct"],
                    "clip_feats": acc[dom]["clip"], "labels": acc[dom]["labels"],
                    "base_ids": acc[dom]["base_ids"],
                    "instance_counts": acc[dom]["K"]}, p)
        kd = {str(k): acc[dom]["K"].count(k) for k in sorted(set(acc[dom]["K"]))}
        print(f"[gen] saved {p}  n={len(acc[dom]['labels'])}  K={kd}")


# ─────────────────────────── 训练 + 评估 ───────────────────────────
def _fused(model_in, cic, fuse, spn_i, dct_i, clip_i, device):
    import torch
    fused = fuse([spn_i], [dct_i], [clip_i], cic, device)
    return float(torch.sigmoid(model_in(fused)).item())


def _load(dom):
    import torch
    return torch.load(os.path.join(OUT_DIR, f"{dom}_features.pt"),
                      map_location="cpu", weights_only=False)


def _train_eval(assign, tag, device, epochs=200):
    """assign: {base_id: 'train'|'val'|'test'}. 训练 MLP 并在 test 上算 operating point。"""
    import torch
    import torch.nn as nn
    from models.consistency import CrossInstanceConsistencyV2
    from models.fusion_classifier import FusionClassifierV2
    from scripts.train import compute_fused_features

    cic = CrossInstanceConsistencyV2()
    fuse = compute_fused_features
    data = {d: _load(d) for d in DOMAINS}

    # 组装样本
    def build(want):
        X_spn, X_dct, X_clip, y = [], [], [], []
        for dom in DOMAINS:
            d = data[dom]
            for i, b in enumerate(d["base_ids"]):
                if assign.get(b) != want:
                    continue
                X_spn.append(d["spn_feats"][i]); X_dct.append(d["dct_feats"][i])
                X_clip.append(d["clip_feats"][i]); y.append(d["labels"][i])
        return X_spn, X_dct, X_clip, y

    def featurize(parts):
        Xs, Xd, Xc, y = parts
        feats = []
        for i in range(len(y)):
            feats.append(fuse([Xs[i]], [Xd[i]], [Xc[i]], cic, device))
        return torch.cat(feats, 0), torch.tensor(y, dtype=torch.float32)

    (tr, va, te) = (build("train"), build("val"), build("test"))
    Xtr, ytr = featurize(tr)
    Xva, yva = featurize(va)
    Xte, yte = featurize(te)

    model = FusionClassifierV2(input_dim=698).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss()
    Xtr, ytr = Xtr.to(device), ytr.to(device)
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        loss = lossf(model(Xtr).squeeze(1), ytr)
        loss.backward(); opt.step()

    # test operating point
    model.eval()
    with torch.no_grad():
        p_te = torch.sigmoid(model(Xte.to(device)).squeeze(1)).cpu().numpy()
        p_va = torch.sigmoid(model(Xva.to(device)).squeeze(1)).cpu().numpy()
    yv, yt = yva.numpy(), yte.numpy()

    def sweep(p, y, taus=np.arange(0.05, 1.0, 0.05)):
        pos, neg = y == 1, y == 0
        rows = []
        for t in taus:
            pred = p >= t
            rows.append({"tau": round(float(t), 2),
                         "tpr": round(float((pred & pos).sum() / max(pos.sum(), 1)), 4),
                         "fpr": round(float((pred & neg).sum() / max(neg.sum(), 1)), 4)})
        return rows

    rep = {"tag": tag, "n_train": int(len(ytr)), "n_val": int(len(yv)), "n_test": int(len(yt)),
           "test_pos": int((yt == 1).sum()), "test_neg": int((yt == 0).sum()),
           "sweep": sweep(p_te, yt)}
    for b in (0.05, 0.10, 0.20):
        cand = [r for r in rep["sweep"] if r["fpr"] <= b]
        rep[f"best_tpr_at_fpr<={int(b*100)}pct"] = max(cand, key=lambda r: r["tpr"]) if cand else None
    print(f"[eval:{tag}] n_train={rep['n_train']} n_test={rep['n_test']} "
          f"(pos={rep['test_pos']}, neg={rep['test_neg']})")
    for k in ("best_tpr_at_fpr<=5pct", "best_tpr_at_fpr<=10pct", "best_tpr_at_fpr<=20pct"):
        print(f"   {k}: {rep[k]}")
    return rep


def do_eval(args):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clean = _load("clean")
    base_ids = sorted(set(clean["base_ids"]))
    print(f"[eval] {len(base_ids)} 个 base image")

    # (a) random split：对每个域的**每个样本**随机划分（复现原设置，允许同 base 跨域跨 split）
    rng = np.random.RandomState(args.seed)
    per_sample = {}
    for dom in DOMAINS:
        d = _load(dom)
        n = len(d["base_ids"])
        idx = rng.permutation(n)
        ntr, nva = int(n * 0.6), int(n * 0.2)
        for j, k in enumerate(idx):
            per_sample[(dom, k)] = "train" if j < ntr else ("val" if j < ntr + nva else "test")

    # 把 per-sample 映射成 "以 (dom,base) 为 key" 的 assign：用一个代理 dict
    class Proxy(dict):
        pass
    assign_random = Proxy()
    for dom in DOMAINS:
        d = _load(dom)
        for k, b in enumerate(d["base_ids"]):
            assign_random[(dom, k)] = per_sample[(dom, k)]

    # (b) grouped split：base 级划分，同一 base 的三版本同 split
    rb = np.random.RandomState(args.seed)
    bidx = rb.permutation(len(base_ids))
    ntr, nva = int(len(base_ids) * 0.6), int(len(base_ids) * 0.2)
    base_assign = {}
    for j, k in enumerate(bidx):
        base_assign[base_ids[k]] = "train" if j < ntr else ("val" if j < ntr + nva else "test")

    # 用 grouped 的 assign 但按 base_id 键；random 的需要单独处理 => 重写 build 用 base_id：
    # grouped: 直接 base_id -> split
    rep_grouped = _train_eval_grouped(base_assign, "grouped(base-level)", device)
    rep_random = _train_eval_random(per_sample, "random(per-sample, like original)", device)

    out = os.path.join(os.path.dirname(CODE_DIR), "results", "tvc", "grouped_split.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"n_base": len(base_ids), "grouped": rep_grouped,
                   "random": rep_random}, f, indent=2, ensure_ascii=False)
    print(f"[eval] wrote {out}")
    print("\n=== 泄漏影响 ===")
    print(f"  grouped  TPR@FPR<=20%: {rep_grouped.get('best_tpr_at_fpr<=20pct')}")
    print(f"  random   TPR@FPR<=20%: {rep_random.get('best_tpr_at_fpr<=20pct')}")


def _featurize_indices(dom_index_pairs, data, fuse, cic, device):
    import torch
    feats, y = [], []
    for dom, i in dom_index_pairs:
        d = data[dom]
        feats.append(fuse([d["spn_feats"][i]], [d["dct_feats"][i]], [d["clip_feats"][i]], cic, device))
        y.append(d["labels"][i])
    return torch.cat(feats, 0), torch.tensor(y, dtype=torch.float32)


def _train_and_report(Xtr, ytr, Xva, yva, Xte, yte, tag, device, epochs=200):
    import torch, torch.nn as nn
    from models.fusion_classifier import FusionClassifierV2
    model = FusionClassifierV2(input_dim=698).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss()
    Xtr, ytr = Xtr.to(device), ytr.to(device)
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        loss = lossf(model(Xtr).squeeze(1), ytr); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(Xte.to(device)).squeeze(1)).cpu().numpy()
    y = yte.numpy()
    pos, neg = y == 1, y == 0
    sweep = []
    for t in np.arange(0.05, 1.0, 0.05):
        pred = p >= t
        sweep.append({"tau": round(float(t), 2),
                      "tpr": round(float((pred & pos).sum() / max(pos.sum(), 1)), 4),
                      "fpr": round(float((pred & neg).sum() / max(neg.sum(), 1)), 4)})
    rep = {"tag": tag, "n_train": int(len(ytr)), "n_test": int(len(y)),
           "test_pos": int(pos.sum()), "test_neg": int(neg.sum()), "sweep": sweep}
    for b in (0.05, 0.10, 0.20):
        cand = [r for r in sweep if r["fpr"] <= b]
        rep[f"best_tpr_at_fpr<={int(b*100)}pct"] = max(cand, key=lambda r: r["tpr"]) if cand else None
    print(f"[{tag}] train={rep['n_train']} test={rep['n_test']} (pos={rep['test_pos']},neg={rep['test_neg']})")
    for k in ("best_tpr_at_fpr<=5pct", "best_tpr_at_fpr<=10pct", "best_tpr_at_fpr<=20pct"):
        print(f"    {k}: {rep[k]}")
    return rep


def _prep(device):
    from models.consistency import CrossInstanceConsistencyV2
    from scripts.train import compute_fused_features
    return CrossInstanceConsistencyV2(), compute_fused_features


def _train_eval_grouped(base_assign, tag, device):
    data = {d: _load(d) for d in DOMAINS}
    cic, fuse = _prep(device)
    parts = {"train": [], "val": [], "test": []}
    for dom in DOMAINS:
        d = data[dom]
        for i, b in enumerate(d["base_ids"]):
            parts[base_assign[b]].append((dom, i))
    Xtr, ytr = _featurize_indices(parts["train"], data, fuse, cic, device)
    Xva, yva = _featurize_indices(parts["val"], data, fuse, cic, device)
    Xte, yte = _featurize_indices(parts["test"], data, fuse, cic, device)
    return _train_and_report(Xtr, ytr, Xva, yva, Xte, yte, tag, device)


def _train_eval_random(per_sample, tag, device):
    data = {d: _load(d) for d in DOMAINS}
    cic, fuse = _prep(device)
    parts = {"train": [], "val": [], "test": []}
    for dom in DOMAINS:
        d = data[dom]
        for i in range(len(d["base_ids"])):
            parts[per_sample[(dom, i)]].append((dom, i))
    Xtr, ytr = _featurize_indices(parts["train"], data, fuse, cic, device)
    Xva, yva = _featurize_indices(parts["val"], data, fuse, cic, device)
    Xte, yte = _featurize_indices(parts["test"], data, fuse, cic, device)
    return _train_and_report(Xtr, ytr, Xva, yva, Xte, yte, tag, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["gen", "eval", "all"], default="all")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    if args.stage in ("gen", "all"):
        do_gen(args)
    if args.stage in ("eval", "all"):
        do_eval(args)


if __name__ == "__main__":
    main()
