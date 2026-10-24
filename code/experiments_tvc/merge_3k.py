#!/usr/bin/env python3
"""合并三个构建集训练数据 → index_3k.json（数据规模提升到 3300）"""
import json, os
TVC = "/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard/results/tvc"
merged = []
for d in ("lora_data", "lora_coco", "lora_inria"):
    p = f"{TVC}/{d}/index.json"
    if not os.path.exists(p): print("skip", d); continue
    for r in json.load(open(p)):
        if r["split"] == "train":
            r = dict(r); r["src"] = d; merged.append(r)
out = f"{TVC}/lora_3k"
os.makedirs(out, exist_ok=True)
json.dump(merged, open(f"{out}/index.json", "w"))
from collections import Counter
c = Counter(r["label"] for r in merged)
print(f"merged train: {len(merged)} (pos={c[1]}, neg={c[0]}) → {out}/index.json")
