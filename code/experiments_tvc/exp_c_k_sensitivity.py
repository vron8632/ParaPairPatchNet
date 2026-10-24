#!/usr/bin/env python3
"""
TVC 补实验 C — 实例数 K 敏感性 （可运行版）
==============================================
动机:
"effects of missed instances, segmentation errors, and different values of K are not
sufficiently studied. All APRICOT images have K=0, so the real-world evaluation does not
validate the proposed cross-instance mechanism."

支持两个任务（真实数据布局）:
  --task patch : clean(0) vs digital_patch(1) vs physical_patch(1)
  --task casia : casia2（拼接+copy-move，labels 0/1）

输出每个 K 桶的 (n, acc, TPR, FPR)，以及 K>=2 子集的独立 operating point。
用法:
  python3 code/experiments_tvc/exp_c_k_sensitivity.py --base <SRC> --model digital_baseline_v2 --task patch
输出: results/tvc/k_sensitivity_<model>_<task>.json
"""
import argparse, collections, json, os, sys
import numpy as np

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)

PATCH_DOMAINS = {"clean": 0, "digital_patch": 1, "physical_patch": 1}


def infer(model_dir, model_name, feature_dir, split, device, task):
    import torch
    from models.consistency import CrossInstanceConsistencyV2
    from models.fusion_classifier import FusionClassifierV2
    from scripts.train import compute_fused_features

    model = FusionClassifierV2(input_dim=698).to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, model_name, "fusion_best.pth"),
                                     map_location="cpu", weights_only=False))
    model.eval()
    cic = CrossInstanceConsistencyV2()

    units = PATCH_DOMAINS if task == "patch" else {"casia2": None}
    probs, labels, ks = [], [], []
    with torch.no_grad():
        for domain, forced in units.items():
            d = torch.load(os.path.join(feature_dir, f"{domain}_{split}_features.pt"),
                           map_location="cpu", weights_only=False)
            s, dd, c = d["spn_feats"], d["dct_feats"], d["clip_feats"]
            ic = d.get("instance_counts")
            for i in range(len(s)):
                fused = compute_fused_features([s[i]], [dd[i]], [c[i]], cic, device)
                probs.append(float(torch.sigmoid(model(fused)).item()))
                labels.append(forced if forced is not None else int(d["labels"][i]))
                ks.append(int(ic[i]) if ic else int(len(s[i])))
    return np.array(probs), np.array(labels), np.array(ks)


def bucket(k):
    return str(k) if k < 5 else "5+"


def k_report(probs, labels, ks, tau=0.5):
    pred = probs >= tau
    rep = {}
    for b in ["0", "1", "2", "3", "4", "5+"]:
        m = np.array([bucket(int(k)) == b for k in ks])
        if m.sum() == 0:
            continue
        pos, neg = (labels == 1) & m, (labels == 0) & m
        rep[f"K={b}"] = {
            "n": int(m.sum()),
            "acc_pct": round(float((pred[m] == labels[m]).mean() * 100), 2),
            "tpr": round(float((pred & pos).sum() / max(pos.sum(), 1)), 4),
            "fpr": round(float((pred & neg).sum() / max(neg.sum(), 1)), 4),
        }
    # 聚集 K>=2
    m2 = ks >= 2
    if m2.sum():
        pos, neg = (labels == 1) & m2, (labels == 0) & m2
        rep["K>=2(agg)"] = {
            "n": int(m2.sum()),
            "acc_pct": round(float((pred[m2] == labels[m2]).mean() * 100), 2),
            "tpr": round(float((pred & pos).sum() / max(pos.sum(), 1)), 4),
            "fpr": round(float((pred & neg).sum() / max(neg.sum(), 1)), 4),
        }
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", default="digital_baseline_v2")
    ap.add_argument("--task", choices=["patch", "casia"], default="patch")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_dir = os.path.join(args.base, "results", "features")
    model_dir = os.path.join(args.base, "results", "models")
    out = args.out or os.path.join(os.path.dirname(CODE_DIR), "results", "tvc",
                                   f"k_sensitivity_{args.model}_{args.task}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    probs, labels, ks = infer(model_dir, args.model, feature_dir, args.split, device, args.task)
    report = {
        "model": args.model, "task": args.task, "split": args.split,
        "n": int(len(labels)),
        "k_distribution": {str(k): int(v) for k, v in
                           sorted(collections.Counter(ks.tolist()).items())},
        "overall": {"acc_pct": round(float(((probs >= 0.5) == labels).mean() * 100), 2)},
        "by_k": k_report(probs, labels, ks),
    }
    with open(out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[exp_c] wrote {out}")
    print(f"  K distribution: {report['k_distribution']}")
    for k, v in report["by_k"].items():
        print(f"  {k}: n={v['n']:5d} acc={v['acc_pct']:6.2f} tpr={v['tpr']:.3f} fpr={v['fpr']:.3f}")


if __name__ == "__main__":
    main()
