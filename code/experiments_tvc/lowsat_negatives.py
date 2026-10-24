#!/usr/bin/env python3
"""
低饱和/低纹理负样本增强（扩展实验）
===============================================
机制分析发现：高分误报样本显著低饱和（ρ=-0.26）、低纹理。
本脚本从 COCO/INRIA 池中筛出饱和度与纹理最低的负样本，加入训练集，
重训后重测 columbia 源尾部 FPR——把"相关性归因"升级为"干预性证据"。
输出: results/tvc/lora_lowsat/index.json
"""
import os, json
import numpy as np

WS = "/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard"
TVC = f"{WS}/results/tvc"
BASE = f"{TVC}/baseline_imgs"
OUT = f"{TVC}/lora_lowsat"


def img_stats(p):
    from PIL import Image
    im = Image.open(p).convert("RGB")
    im.thumbnail((256, 256))
    a = np.asarray(im).astype(np.float32) / 255.0
    mx, mn = a.max(2), a.min(2)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0).mean()
    g = a.mean(2)
    lap = np.abs(4*g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]).mean()
    return float(sat), float(lap)


def main():
    idx = json.load(open(f"{TVC}/lora_data/index.json"))
    base = [r for r in idx if r["split"] == "train"]
    # 候选池：本地点样本之外的自然图（COCO val + INRIA 池）
    cand = []
    for src in ("coco_val", "inria_noped"):
        d = f"{BASE}/{src}"
        if os.path.isdir(d):
            cand += [(f"{d}/{f}", src) for f in sorted(os.listdir(d))]
    # 扩展到更多本地可得图（COCO 训练池目录若存在）
    for extra in ("/media/oyp/工作/Projects/042_image_forensic/DuetGuard/data/coco/val2017",
                  "/media/oyp/工作/Projects/042_image_forensic/DuetGuard/data/inria"):
        if os.path.isdir(extra):
            fs = [f for f in sorted(os.listdir(extra)) if f.lower().endswith((".jpg", ".png", ".jpeg"))][:600]
            cand += [(os.path.join(extra, f), os.path.basename(extra)) for f in fs]
    print(f"候选负样本池: {len(cand)}")
    sc = []
    for i, (p, src) in enumerate(cand):
        try:
            s, l = img_stats(p); sc.append((s, l, p, src))
        except Exception: pass
        if (i+1) % 200 == 0: print(f"  统计 {i+1}/{len(cand)}", flush=True)
    # 低饱和 & 低纹理各取前 15%，去重合并
    sc.sort(key=lambda x: x[0]); low_sat = sc[:max(1, len(sc)//7)]
    sc.sort(key=lambda x: x[1]); low_tex = sc[:max(1, len(sc)//7)]
    picked = {}
    for s, l, p, src in low_sat + low_tex: picked[p] = (s, l, src, p)
    print(f"入选低饱和/低纹理负样本: {len(picked)}")
    add = [{"path": p, "label": 0, "domain": f"lowsat_{v[2]}", "split": "train"} for p, v in picked.items()]
    # 训练集 = 原 train（1000 pos + 500 neg）+ 新增低饱和 neg
    out = [r for r in base] + add
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(f"{OUT}/index.json", "w"))
    npos = sum(1 for r in out if r["label"] == 1)
    print(f"→ {OUT}/index.json: {len(out)} 条 (pos={npos}, neg={len(out)-npos})")


if __name__ == "__main__":
    main()
