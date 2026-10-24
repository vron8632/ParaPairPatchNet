#!/usr/bin/env python3
"""APDE train parquet → jpg（按 image_sha256 去重），供 P1 族级泛化训练"""
import os, json, glob, io, sys
from collections import Counter, defaultdict
import pandas as pd
from PIL import Image

SRC = "/home/oyp/fastdata/apde_full/data/train"
CLN = "/home/oyp/fastdata/apde_full/clean"
OUT = "/home/oyp/fastdata/apde_train_jpg"

def dump(df, tag, out_dir, seen, idx):
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for _, r in df.iterrows():
        b = r["image"]
        if not isinstance(b, dict) or "bytes" not in b: continue
        h = r["image_sha256"]
        if h in seen: continue
        seen.add(h)
        try:
            im = Image.open(io.BytesIO(b["bytes"])).convert("RGB")
        except Exception:
            continue
        name = f"{tag}_{h[:16]}.jpg"
        im.save(os.path.join(out_dir, name), quality=92)
        idx.append({"path": os.path.join(out_dir, name), "label": 1,
                    "method": str(r["method"]), "detector": str(r["detector"]),
                    "source_dataset": str(r["source_dataset"]), "domain": tag})
        n += 1
    return n

def main():
    seen, idx = set(), []
    fs = sorted(glob.glob(f"{SRC}/*.parquet"))
    print(f"[export] {len(fs)} 个 train parquet", flush=True)
    per_method = Counter()
    for i, f in enumerate(fs):
        df = pd.read_parquet(f, columns=["image", "image_sha256", "method", "detector", "source_dataset"])
        tag = os.path.basename(f).replace(".parquet", "")
        n = dump(df, tag, f"{OUT}/patched", seen, idx)
        per_method[str(df["method"].iloc[0])] += n
        if (i+1) % 8 == 0:
            print(f"  {i+1}/{len(fs)} 去重后累计 {len(idx)}", flush=True)
    # clean 负样本
    for name in ("negative", "positive"):
        p = f"{CLN}/{name}.parquet"
        if not os.path.exists(p): continue
        df = pd.read_parquet(p, columns=["image", "image_sha256"])
        n = dump(df, f"clean_{name}", f"{OUT}/clean", seen, idx)
        # clean 的 label 改 0
        for r in idx[-n:]: r["label"] = 0; r["method"] = "clean"
        print(f"  clean/{name}: {n}", flush=True)
    json.dump(idx, open(f"{OUT}/index.json", "w"), ensure_ascii=False)
    print(f"\n[export] 总计 {len(idx)} 张 → {OUT}")
    print(f"  按 method: {dict(per_method)}")
    print(f"  label: {dict(Counter(r['label'] for r in idx))}")

if __name__ == "__main__":
    main()
