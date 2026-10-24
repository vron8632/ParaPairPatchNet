#!/usr/bin/env python3
"""导出额外负样本（COCO-train2017，与 APDE 的 val2017 无重叠）"""
import os, json, glob, random, argparse
ap=argparse.ArgumentParser(); ap.add_argument("--n",type=int,default=20000); ap.add_argument("--seed",type=int,default=0)
a=ap.parse_args()
SRC="/media/oyp/工作/Projects/042_image_forensic/DuetGuard/data/coco/train2017"
OUT="/home/oyp/fastdata/apde_negs"; os.makedirs(OUT,exist_ok=True)
fs=sorted(glob.glob(f"{SRC}/*.jpg"))
# 排除 APDE clean 里出现过的（它们是 val2017，理论上不重叠，双保险）
import pandas as pd
used=set()
for p in ["/home/oyp/fastdata/apde_full/clean/negative.parquet","/home/oyp/fastdata/apde_full/clean/positive.parquet"]:
    d=pd.read_parquet(p,columns=["source_id"])
    used |= {s.replace("coco_","") for s in d.source_id if str(s).startswith("coco_")}
fs=[f for f in fs if os.path.basename(f).replace(".jpg","") not in used]
print(f"  候选 {len(fs)} 张（已排除 APDE 用过的 {len(used)} 个 source）")
rng=random.Random(a.seed); rng.shuffle(fs); sel=fs[:a.n]
idx=[{"path":p,"label":0,"method":"clean","domain":"coco_train_clean"} for p in sel]
json.dump(idx,open(f"{OUT}/index.json","w"),ensure_ascii=False)
print(f"  ✅ 导出 {len(idx)} 张负样本索引 → {OUT}/index.json（不复制文件，直接引用原路径）")
