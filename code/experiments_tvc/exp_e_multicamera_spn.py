#!/usr/bin/env python3
"""
TVC 补实验 E — 多相机下 SPN / DCT 分支的真实贡献
==================================================
动机: 多相机下的 SPN 稳定性检查
"The ablation study shows that CLIP is overwhelmingly the dominant feature: removing CLIP
reduces accuracy by 17.7 points, whereas removing SPN or DCT does not hurt performance.
This makes it unclear how much the proposed forensic cues actually contribute."

论文当前的解释是"SPN/DCT 在单相机仿真下中性"。这是**假说**，需要用多相机真实数据检验：
若在同一相机（单源）下 SPN/DCT 确实中性、而在**跨相机**场景下变得有判别力，则解释成立、
消融的"缺点"转为"发现"；若不成立，则应诚实下调对取证线索的 claim。

用法:
    python3 code/experiments_tvc/exp_e_multicamera_spn.py --dataset multicam
输出: results/tvc/multicamera_spn.json
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_BASE = "/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard"


def branch_discriminability(feature_dir, dataset, branch, model_dir, device="cpu"):
    """训练/评估仅用单一分支特征的线性探针，报告 test AUC。branch ∈ {spn,dct,clip}。"""
    import torch
    from sklearn.metrics import roc_auc_score
    from models.consistency import CrossInstanceConsistencyV2
    from scripts.train import compute_fused_features

    def load(split):
        return torch.load(os.path.join(feature_dir, f"{dataset}_{split}_features.pt"),
                          map_location="cpu", weights_only=False)

    tr, te = load("train"), load("test")
    cic = CrossInstanceConsistencyV2()

    def feats(d):
        out = []
        for i in range(len(d["labels"])):
            s = d["spn_feats"][i] if branch != "spn" else d["spn_feats"][i]
            dd = d["dct_feats"][i]
            c = d["clip_feats"][i]
            # 只保留目标分支的逐实例平均（其余置零），再走同一 fusion
            if branch == "spn":
                fused = compute_fused_features([s], [torch.zeros_like(dd)],
                                               [torch.zeros_like(c)], cic, device)
            elif branch == "dct":
                fused = compute_fused_features([torch.zeros_like(s)], [dd],
                                               [torch.zeros_like(c)], cic, device)
            else:
                fused = compute_fused_features([torch.zeros_like(s)],
                                               [torch.zeros_like(dd)], [c], cic, device)
            out.append(fused)
        return torch.cat(out, 0)

    Xtr, ytr = feats(tr), torch.tensor(tr["labels"])
    Xte, yte = feats(te), torch.tensor(te["labels"])
    probe = torch.nn.Linear(Xtr.shape[1], 1)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3)
    lossf = torch.nn.BCEWithLogitsLoss()
    for _ in range(200):
        opt.zero_grad()
        loss = lossf(probe(Xtr).squeeze(1), ytr.float())
        loss.backward(); opt.step()
    with torch.no_grad():
        scores = probe(Xte).squeeze(1).numpy()
    return float(roc_auc_score(yte.numpy(), scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--dataset", default="multicam")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    feature_dir = os.path.join(args.base, "results", "features")
    model_dir = os.path.join(args.base, "results", "models")
    out = args.out or os.path.join(args.base, "results", "tvc", "multicamera_spn.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    report = {"dataset": args.dataset, "single_branch_auc": {}}
    try:
        for branch in ("spn", "dct", "clip"):
            report["single_branch_auc"][branch] = round(
                branch_discriminability(feature_dir, args.dataset, branch, model_dir), 4)
    except FileNotFoundError as e:
        report["status"] = "need_multicam_features"
        report["error"] = str(e)

    report["note"] = ("跨相机(feature_dir 含多相机来源)时若 SPN/DCT 的 single-branch AUC 显著"
                      "高于单相机情形，则'取证线索在单相机仿真下冗余、跨设备下有用'的解释成立。")

    with open(out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[multicamera-spn] 已写 {out}")


if __name__ == "__main__":
    main()
