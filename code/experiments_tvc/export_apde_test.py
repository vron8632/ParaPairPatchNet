#!/usr/bin/env python3
"""APDE test parquet(13 族) → jpg，按 image_sha256 去重"""
import os, json, glob, io
import pandas as pd
from PIL import Image
SRC="/home/oyp/fastdata/apde_test"; OUT="/home/oyp/fastdata/apde_test_jpg"
os.makedirs(f"{OUT}/patched", exist_ok=True)
seen, idx = set(), []
for f in sorted(glob.glob(f"{SRC}/*.parquet")):
    df = pd.read_parquet(f, columns=["image","image_sha256","method","detector"])
    tag = os.path.basename(f).replace(".parquet","")
    for _, r in df.iterrows():
        b=r["image"]
        if not isinstance(b,dict) or "bytes" not in b: continue
        h=r["image_sha256"]
        if h in seen: continue
        seen.add(h)
        try: im=Image.open(io.BytesIO(b["bytes"])).convert("RGB")
        except Exception: continue
        n=f"{tag}_{h[:16]}.jpg"; im.save(f"{OUT}/patched/{n}", quality=92)
        idx.append({"path":f"{OUT}/patched/{n}","label":1,"method":str(r["method"]),"domain":tag})
    print(f"  {tag}: 累计 {len(idx)}", flush=True)
# 负样本：APDE clean/negative
CLN="/home/oyp/fastdata/apde_full/clean/negative.parquet"
if os.path.exists(CLN):
    os.makedirs(f"{OUT}/clean", exist_ok=True)
    df=pd.read_parquet(CLN)
    hcol = "image_sha256" if "image_sha256" in df.columns else "original_sha256"
    n0=0
    for _, r in df.iterrows():
        b=r["image"]
        if not isinstance(b,dict) or "bytes" not in b: continue
        h=r[hcol]
        if h in seen: continue
        seen.add(h)
        try: im=Image.open(io.BytesIO(b["bytes"])).convert("RGB")
        except Exception: continue
        nm=f"clean_{h[:16]}.jpg"; im.save(f"{OUT}/clean/{nm}", quality=92)
        idx.append({"path":f"{OUT}/clean/{nm}","label":0,"method":"clean","domain":"clean"}); n0+=1
    print(f"  clean: {n0}", flush=True)
json.dump(idx, open(f"{OUT}/index.json","w"), ensure_ascii=False)
from collections import Counter
print(f"[test export] {len(idx)} 张 | label={dict(Counter(r['label'] for r in idx))}")
print(f"  族: {dict(Counter(r['method'] for r in idx if r['label']==1))}")
