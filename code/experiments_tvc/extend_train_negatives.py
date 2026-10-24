#!/usr/bin/env python3
"""
任务2 数据准备 — 多源负样本训练集（根治 confound）
====================================================
问题: 现有 lora_data 只含 CASIA Au 衍生数据，模型对其他来源（COCO/INRIA）的干净图
      打分明显偏高（coco 0.085 vs casia 0.033），存在数据源 confound。
对策: 把 COCO / INRIA 的**干净图**也作为额外负样本加入训练集，逼模型在所有来源上都判
      "无补丁" => 跨源 FPR 校准一致，消除 confound。

输出: results/tvc/lora_data/index_multi.json
      （= 原 index.json 的 train/val 条目 + COCO/INRIA clean 的 train/val 条目）

用法:
  $PY code/experiments_tvc/extend_train_negatives.py --per-source 400
"""
import argparse, json, os, sys
import numpy as np

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS = os.path.dirname(CODE_DIR)
OUT = os.path.join(WS, "results", "tvc")
SRC = "/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard"
FD = os.path.join(SRC, "results", "features")
IMG_DIR = os.path.join(OUT, "lora_data")


def export_clean(feat_dir, dom, split, tag, per_split, seed=0):
    """把某数据集的 clean 域导出为 jpg（负样本），返回条目列表。"""
    import torch
    from PIL import Image
    p = os.path.join(feat_dir, f"{dom}_{split}.pt")
    if not os.path.exists(p):
        print(f"  跳过（无 {p}）"); return []
    d = torch.load(p, map_location="cpu", weights_only=False)
    n = len(d["tensors"])
    k = min(per_split, n)
    idx = np.linspace(0, n - 1, k).astype(int)
    rows = []
    for i in idx:
        t = d["tensors"][i]
        arr = np.clip(t.permute(1, 2, 0).numpy() * 255, 0, 255).astype(np.uint8)
        fp = os.path.join(IMG_DIR, f"{tag}_{split}_{i}.jpg")
        if not os.path.exists(fp):
            Image.fromarray(arr).save(fp, quality=92)
        rows.append({"path": fp, "label": 0, "domain": tag, "split": split})
    print(f"  {tag}/{split}: 导出 {len(rows)} 张")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-source", type=int, default=400, help="每个额外来源每 split 的张数")
    ap.add_argument("--out", default=os.path.join(OUT, "lora_data", "index_multi.json"))
    args = ap.parse_args()

    base = json.load(open(os.path.join(IMG_DIR, "index.json")))
    print(f"[ext] 原训练集 {len(base)} 条")

    extra = []
    for tag, fd in (("coco_clean", os.path.join(FD, "coco_patch")),
                    ("inria_clean", os.path.join(FD, "inria_patch"))):
        for split, tag_split in (("train", "train"), ("val", "val")):
            extra += export_clean(fd, "clean", split, tag, args.per_source)

    rows = base + extra
    json.dump(rows, open(args.out, "w"), indent=2)
    from collections import Counter
    print(f"[ext] 共 {len(rows)} 条 -> {args.out}")
    print("     域 x split:", dict(Counter((r["domain"], r["split"]) for r in rows)))


if __name__ == "__main__":
    main()
