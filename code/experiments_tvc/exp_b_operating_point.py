#!/usr/bin/env python3
"""
TVC 补实验 B — Operating Point / Calibration （可运行版）
==========================================================
动机: 多项鲁棒性检查
"95.5% TPR comes with a 70.5% FPR ... correctly accepts only 29% of clean images."

数据布局（已核实）:
  features/{domain}_{split}_features.pt,  domain ∈ {clean(0), digital_patch(1), physical_patch(1)}
  模型: results/models/{model}/fusion_best.pth  (FusionClassifierV2, input_dim=698)

用法:
  python3 code/experiments_tvc/exp_b_operating_point.py \
      --base /media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard \
      --model digital_baseline_v2 --split test
输出: results/tvc/operating_point_<model>_<split>.json   （真实数值，不做任何改动）
"""
import argparse, json, os, sys
import numpy as np

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)

# 物理补丁任务：干净=0，两个 patch 域=1
DOMAIN_LABELS = {"clean": 0, "digital_patch": 1, "physical_patch": 1}


def load_domain(feature_dir, domain, split):
    import torch
    p = os.path.join(feature_dir, f"{domain}_{split}_features.pt")
    return torch.load(p, map_location="cpu", weights_only=False)


def infer_probs(model_dir, model_name, feature_dir, split, device):
    import torch
    from models.consistency import CrossInstanceConsistencyV2
    from models.fusion_classifier import FusionClassifierV2
    from scripts.train import compute_fused_features

    model = FusionClassifierV2(input_dim=698).to(device)
    sd = torch.load(os.path.join(model_dir, model_name, "fusion_best.pth"),
                    map_location="cpu", weights_only=False)
    model.load_state_dict(sd)
    model.eval()
    cic = CrossInstanceConsistencyV2()

    out = {}
    with torch.no_grad():
        for domain, lab in DOMAIN_LABELS.items():
            d = load_domain(feature_dir, domain, split)
            s, dd, c = d["spn_feats"], d["dct_feats"], d["clip_feats"]
            ic = d.get("instance_counts")
            probs, ks = [], []
            for i in range(len(s)):
                fused = compute_fused_features([s[i]], [dd[i]], [c[i]], cic, device)
                probs.append(float(torch.sigmoid(model(fused)).item()))
                ks.append(int(ic[i]) if ic else int(len(s[i])))
            out[domain] = {"label": lab, "probs": probs, "K": ks}
    return out


def sweep(probs, labels, taus):
    probs, labels = np.asarray(probs), np.asarray(labels)
    pos, neg = labels == 1, labels == 0
    rows = []
    for t in taus:
        pred = probs >= t
        tpr = (pred & pos).sum() / max(pos.sum(), 1)
        fpr = (pred & neg).sum() / max(neg.sum(), 1)
        rows.append({"tau": round(float(t), 3), "tpr": round(float(tpr), 4),
                     "fpr": round(float(fpr), 4),
                     "clean_acc": round(float(1 - fpr), 4),
                     "acc": round(float((pred == labels).mean()), 4)})
    return rows


def auc(probs, labels):
    import numpy as np
    probs, labels = np.asarray(probs), np.asarray(labels)
    pos, neg = probs[labels == 1], probs[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    # rank-based AUC
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order), float)
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) /
                 (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", default="digital_baseline_v2")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_dir = os.path.join(args.base, "results", "features")
    model_dir = os.path.join(args.base, "results", "models")
    out = args.out or os.path.join(os.path.dirname(CODE_DIR), "results", "tvc",
                                   f"operating_point_{args.model}_{args.split}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    data = infer_probs(model_dir, args.model, feature_dir, args.split, device)
    taus = [round(x, 2) for x in np.arange(0.05, 1.0, 0.05)]

    # 物理补丁（正类）vs 干净（负类）二分类
    phys_p = data["physical_patch"]["probs"]
    dig_p = data["digital_patch"]["probs"]
    clean_p = data["clean"]["probs"]
    labels = [1] * len(phys_p) + [0] * len(clean_p)
    probs = phys_p + clean_p

    report = {
        "base": args.base, "model": args.model, "split": args.split, "device": device,
        "n_clean": len(clean_p), "n_digital": len(dig_p), "n_physical": len(phys_p),
        "prob_stats": {
            "clean": {"mean": round(float(np.mean(clean_p)), 4),
                      "median": round(float(np.median(clean_p)), 4)},
            "digital_patch": {"mean": round(float(np.mean(dig_p)), 4)},
            "physical_patch": {"mean": round(float(np.mean(phys_p)), 4)},
        },
        "auc_physical_vs_clean": round(auc(probs, labels), 4),
        "sweep": sweep(probs, labels, taus),
    }
    for budget in (0.01, 0.05, 0.10, 0.20):
        cand = [r for r in report["sweep"] if r["fpr"] <= budget]
        report[f"best_tpr_at_fpr<={int(budget*100)}pct"] = (
            max(cand, key=lambda r: r["tpr"]) if cand else None)

    with open(out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[exp_b] wrote {out}")
    print(f"  AUC(physical vs clean) = {report['auc_physical_vs_clean']}")
    print(f"  clean  prob mean = {report['prob_stats']['clean']['mean']}")
    print(f"  phys   prob mean = {report['prob_stats']['physical_patch']['mean']}")
    for k in ("best_tpr_at_fpr<=1pct", "best_tpr_at_fpr<=5pct",
              "best_tpr_at_fpr<=10pct", "best_tpr_at_fpr<=20pct"):
        print(f"  {k}: {report[k]}")


if __name__ == "__main__":
    main()
