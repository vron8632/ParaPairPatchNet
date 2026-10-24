#!/usr/bin/env python3
"""
通用特征提取（针对 patch_dataset_v2 / coco_patch 这类 .pt 张量数据集）
======================================================================
原 code/scripts/extract_features.py 依赖 InstanceForensicDataset（不支持自定义 .pt），
且 CLIP 权重此前无法离线加载。本脚本：
  * 直接读 {domain}_{split}.pt 的 tensors/labels/base_ids
  * CLIP 通过 HF_ENDPOINT=https://hf-mirror.com 在线加载（已验证可用）
  * 输出 {dom}_{sp}_features.pt，与 patch_dataset_v2 的 features 格式完全一致
    （含 paths/base_ids，便于 grouped split 与对齐）

用法:
  export HF_ENDPOINT=https://hf-mirror.com
  $PY code/experiments_tvc/extract_patch_features.py --src <dir>/coco_patch --out <dir>
  $PY code/experiments_tvc/extract_patch_features.py --src <dir>/patch_dataset_v2 --out <dir>
输出: <out>/{domain}_{split}_features.pt
"""
import argparse, os, sys, time

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="含 {dom}_{sp}.pt 的目录")
    ap.add_argument("--out", default=None, help="默认与 --src 同目录")
    ap.add_argument("--batch-checkpoint", type=int, default=100)
    args = ap.parse_args()

    import torch
    import numpy as np
    from features.segmenter import InstanceSegmenter
    from features.spn_extractor import get_spn_extractor
    from features.dct_profile import DCTProfileExtractor
    from features.clip_encoder import get_clip_extractor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = args.out or args.src
    os.makedirs(out, exist_ok=True)
    print(f"[extract] src={args.src}  out={out}  device={device}")

    seg = InstanceSegmenter(device=device)
    spn = get_spn_extractor(mode="learned", device=device)
    dct = DCTProfileExtractor()
    clip = get_clip_extractor(device=device)

    import glob
    files = sorted(glob.glob(os.path.join(args.src, "*_*.pt")))
    files = [f for f in files if "features" not in os.path.basename(f)
             and "meta" not in os.path.basename(f)]
    print(f"[extract] 待处理 {len(files)} 个文件: {[os.path.basename(f) for f in files]}")

    for fp in files:
        base = os.path.basename(fp).replace(".pt", "")
        dst = os.path.join(out, f"{base}_features.pt")
        if os.path.exists(dst):
            print(f"  skip {base} (已存在)"); continue
        d = torch.load(fp, map_location="cpu", weights_only=False)
        tensors, labels = d["tensors"], d["labels"]
        base_ids = d.get("base_ids") or [f"{base}_{i}" for i in range(len(tensors))]
        n = len(tensors)
        all_spn, all_dct, all_clip, all_K = [], [], [], []
        t0 = time.time()
        for i, t in enumerate(tensors):
            img = t.float()
            masks = seg.segment(img)
            masks = [(m, b) for m, b in masks if np.asarray(m).sum() > 32 * 32]
            K = len(masks)
            all_K.append(K)
            if K < 2:
                # K<2 时退化为整图特征（与原 extract_features.py 一致）
                all_spn.append(spn.extract(img.to(device)).unsqueeze(0).cpu())
                all_dct.append(dct.extract_profile(img).unsqueeze(0).cpu())
                all_clip.append(clip.encode_image(img).unsqueeze(0).cpu())
            else:
                sl, dl, cl = [], [], []
                for mask, _ in masks:
                    mask_t = torch.from_numpy(np.asarray(mask)).float().to(device)
                    sl.append(spn.extract_instance(img.to(device), mask_t).cpu())
                    dl.append(dct.extract_profile(img, mask).cpu())
                    cl.append(clip.encode_image(img, mask).cpu())
                all_spn.append(torch.stack(sl)); all_dct.append(torch.stack(dl))
                all_clip.append(torch.stack(cl))
            if (i + 1) % 50 == 0:
                el = time.time() - t0
                print(f"    {base}: {i+1}/{n} ({el:.0f}s, ETA {el/(i+1)*(n-i-1):.0f}s)",
                      flush=True)

        torch.save({"spn_feats": all_spn, "dct_feats": all_dct, "clip_feats": all_clip,
                    "labels": labels, "paths": base_ids, "base_ids": base_ids,
                    "instance_counts": all_K,
                    "dataset": base, "split": "all", "num_samples": n},
                   dst)
        kd = {str(k): all_K.count(k) for k in sorted(set(all_K))}
        print(f"  saved {dst}  n={n}  K={kd}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
