#!/usr/bin/env python3
"""
TVC 补实验 D — 真实 Print-and-Capture 对照评估
================================================
动机: 最致命的一条
"The print-camera simulation is not physically convincing ... Applying different processing
selectively to the pasted patch can introduce an artificial domain boundary that makes
detection easier. The model may recognize the simulator rather than physical printing
artifacts. The simulation requires validation against actual print-and-capture measurements,
ideally across printers, papers, cameras, distances, illumination conditions, and viewing
angles."

本脚本做两件事：
  1) 对**真实打印-拍摄**图像集（由 collect_protocol.md 采集）跑同一 pipeline，报告
     与 PC 模拟域的性能差（sim-to-real gap）。
  2) 关键对照：在真实拍摄图上，把补丁与背景**统一经过同一相机链路**（真实情况）
     vs. 复刻论文当前的"补丁先降质、背景后处理"（人为域边界）——若两者差距很大，
     则坐实"模型可能在识别模拟器"。这一步用真实图 + 受控重渲染实现。

用法:
    python3 code/experiments_tvc/exp_d_print_capture_eval.py \
        --real-dir data/printcapture --labels data/printcapture/labels.json
输出: results/tvc/print_capture.json
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_BASE = "/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard"


def evaluate_dir(real_dir, labels_path, base, device="cpu"):
    """把已训练好的 ForensicGuard 直接应用到真实拍摄目录（零样本）。"""
    import torch
    from dataset.instance_dataset import InstanceForensicDataset  # noqa: F401 (接口占位)
    from features.segmenter import InstanceSegmenter
    from features.spn_extractor import get_spn_extractor
    from features.dct_profile import DCTProfileExtractor
    from features.clip_encoder import get_clip_extractor
    from models.consistency import CrossInstanceConsistencyV2
    from models.fusion_classifier import FusionClassifierV2
    from scripts.train import compute_fused_features

    labels = json.load(open(labels_path))
    seg = InstanceSegmenter(device=device)                     # noqa: F841
    spn = get_spn_extractor(mode="learned", device=device)     # noqa: F841
    dct = DCTProfileExtractor()                                # noqa: F841
    clip = get_clip_extractor(device=device)                   # noqa: F841
    model = FusionClassifierV2(input_dim=698).to(device)
    model.load_state_dict(torch.load(os.path.join(base, "results", "models",
                                                  "casia2", "fusion_best.pth"),
                                     map_location="cpu", weights_only=False))
    model.eval()                                               # noqa: F841
    cic = CrossInstanceConsistencyV2()                         # noqa: F841

    # 说明：真实采集图需要先经 extract_features 落盘，再在此加载评估，保持与论文一致。
    rows = [{"image": k, "label": v} for k, v in labels.items()]
    return {"n": len(rows), "rows": rows,
            "status": "need_extracted_features",
            "note": ("请先对 real-dir 跑 code/scripts/extract_features.py 得到 *_features.pt，"
                     "再在此加载评估；本函数给出接入点。")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--real-dir", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = args.out or os.path.join(args.base, "results", "tvc", "print_capture.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    report = {
        "protocol": "print-and-capture (见 collect_protocol.md)",
        "sim_to_real": evaluate_dir(args.real_dir, args.labels, args.base),
        "boundary_ablation": {
            "todo": ("统一相机链路(真实) vs 论文当前补丁单独降质(人为边界)，"
                     "用真实图 + 受控重渲染对比两者的检测率差异。"),
            "conclusion_placeholder": None,
        },
    }
    with open(out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[print-capture] 已写 {out}（需采集数据后重跑）")


if __name__ == "__main__":
    main()
