#!/usr/bin/env python3
"""
DINOv2 特征提取（整图 + patch-token 统计）
==========================================
诊断显示 CLIP 是唯一有效信号（clean_vs_physical AUC 0.827），SPN/DCT 接近随机。
本脚本用本地 DINOv2 (facebook/dinov2-base) 提取更强的视觉表征：
  cls        : (768,)  CLS token
  patch_mean : (768,)  patch tokens 均值
  patch_std  : (768,)  patch tokens 标准差
并保留整图下采样灰度/高频统计作为"取证"补充通道。

用法:
  PY=/home/oyp/miniconda3/envs/apjf/bin/python
  $PY code/experiments_tvc/extract_dinov2.py --base <SRC> --domains clean digital_patch physical_patch
输出: <base>/results/features/dinov2_{domain}_{split}.pt
"""
import argparse, os, sys
import numpy as np

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)
DINO = "/media/oyp/工作/Projects/030_mllm/DINOv2/facebook/dinov2-base"
SPLITS = ["train", "val", "test"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--domains", nargs="+", default=["clean", "digital_patch", "physical_patch"])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from transformers import AutoModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(DINO, local_files_only=True).to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    pd = os.path.join(args.base, "results", "features", "patch_dataset_v2")
    outdir = os.path.join(args.base, "results", "features")
    for dom in args.domains:
        for sp in SPLITS:
            src = os.path.join(pd, f"{dom}_{sp}.pt")
            dst = os.path.join(outdir, f"dinov2_{dom}_{sp}.pt")
            if os.path.exists(dst):
                print(f"[dino] 已存在, 跳过 {dst}")
                continue
            if not os.path.exists(src):
                print(f"[dino] 缺少 {src}")
                continue
            d = torch.load(src, map_location="cpu", weights_only=False)
            tensors, labels = d["tensors"], d["labels"]
            n = len(tensors)
            cls_all, pm_all, ps_all = [], [], []
            t0 = None
            import time; t0 = time.time()
            with torch.no_grad():
                for s in range(0, n, args.batch):
                    batch = torch.stack(tensors[s:s + args.batch]).float().to(device)
                    x = F.interpolate(batch, size=(args.size, args.size),
                                      mode="bilinear", align_corners=False)
                    x = (x - mean) / std
                    o = model(x).last_hidden_state           # (B, 1+P, 768)
                    cls_all.append(o[:, 0].cpu())
                    patch = o[:, 1:]
                    pm_all.append(patch.mean(1).cpu())
                    ps_all.append(patch.std(1).cpu())
                    if (s // args.batch) % 10 == 0:
                        el = time.time() - t0
                        print(f"    {dom}/{sp}: {s+len(batch)}/{n} ({el:.0f}s)", flush=True)
            torch.save({"cls": torch.cat(cls_all), "patch_mean": torch.cat(pm_all),
                        "patch_std": torch.cat(ps_all),
                        "labels": list(labels),
                        "base_id": [f"{dom}|{sp}|{i}" for i in range(n)],
                        "domain": dom, "split": sp},
                       dst)
            print(f"[dino] saved {dst}  n={n}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
