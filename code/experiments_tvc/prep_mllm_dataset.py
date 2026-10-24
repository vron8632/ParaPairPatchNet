#!/usr/bin/env python3
"""
MLLM 多数据集验证 — 数据准备
==============================
把 coco_patch / inria_patch 的 tensors 导出为 jpg + index.json（格式同 lora_data/index.json），
用于在 COCO / INRIA 上独立复现 LoRA + FPR 约束损失的效果（补"单数据集验证"这一关）。

用法:
  $PY code/experiments_tvc/prep_mllm_dataset.py --dataset coco_patch --tag lora_coco
  $PY code/experiments_tvc/prep_mllm_dataset.py --dataset inria_patch --tag lora_inria
输出: results/tvc/<tag>/index.json 与 jpg 文件
"""
import argparse, json, os, sys
import numpy as np

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS = os.path.dirname(CODE_DIR)
OUT = os.path.join(WS, "results", "tvc")
FD = "/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard/results/features"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="features 下的子目录名，如 coco_patch")
    ap.add_argument("--tag", required=True, help="输出目录名，如 lora_coco")
    ap.add_argument("--per-split", type=int, default=0, help="每 split 最多导出张数(0=全部)")
    args = ap.parse_args()

    import torch
    from PIL import Image

    src = os.path.join(FD, args.dataset)
    dst = os.path.join(OUT, args.tag)
    if not os.path.isdir(src):
        raise SystemExit(f"缺少 {src}")
    os.makedirs(dst, exist_ok=True)

    rows = []
    for dom, lab in (("clean", 0), ("digital_patch", 1), ("physical_patch", 1)):
        for split in ("train", "val"):
            p = os.path.join(src, f"{dom}_{split}.pt")
            if not os.path.exists(p):
                print(f"  跳过（无 {os.path.basename(p)}）"); continue
            d = torch.load(p, map_location="cpu", weights_only=False)
            tensors, labels = d["tensors"], d["labels"]
            n = len(tensors)
            idx = range(n) if args.per_split <= 0 else np.linspace(
                0, n - 1, min(args.per_split, n)).astype(int)
            k = 0
            for i in idx:
                t = tensors[int(i)]
                arr = np.clip(t.permute(1, 2, 0).numpy() * 255, 0, 255).astype(np.uint8)
                fp = os.path.join(dst, f"{dom}_{split}_{int(i)}.jpg")
                if not os.path.exists(fp):
                    Image.fromarray(arr).save(fp, quality=92)
                rows.append({"path": fp, "label": lab, "domain": dom, "split": split})
                k += 1
            print(f"  {args.dataset}/{dom}/{split}: 导出 {k} 张")

    idxp = os.path.join(dst, "index.json")
    json.dump(rows, open(idxp, "w"), indent=2)
    from collections import Counter
    print(f"[prep] 共 {len(rows)} 条 -> {idxp}")
    print("      域xsplit:", dict(Counter((r["domain"], r["split"]) for r in rows)))


if __name__ == "__main__":
    main()
