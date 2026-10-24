#!/usr/bin/env python3
"""导出 APRICOT 本地全部（138 张）带掩码正样本，供主表扩容评测。"""
import os, numpy as np, torch
from PIL import Image
AP = "/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard/data/apricot/dev/data_mask_v2"
OUT = "/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard/results/tvc/baseline_imgs/pos_all"
os.makedirs(OUT, exist_ok=True)
files = sorted(os.listdir(AP), key=lambda f: int(f.split("_")[1].split(".")[0]))
n = 0
for f in files:
    dst = f"{OUT}/{f.replace('.pt', '.png')}"
    if os.path.exists(dst): n += 1; continue
    x = torch.load(os.path.join(AP, f), map_location="cpu", weights_only=False)
    if x["Mask"].sum() <= 0: continue
    arr = np.clip(x["Image"][0].astype(np.float32) * 255.0, 0, 255).astype(np.uint8)
    im = Image.fromarray(arr).convert("RGB"); im.thumbnail((1024, 1024))
    im.save(dst); n += 1
    if n % 20 == 0: print(f"{n} exported", flush=True)
print(f"EXPORT DONE n={n}")
