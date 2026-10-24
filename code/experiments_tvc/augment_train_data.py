#!/usr/bin/env python3
"""
P1：训练数据扩充（EoT 风格增广）
================================
现实约束（如实）：
  - 本地无 APRICOT 原始补丁素材（MITRE 需申请，网络不通）→ 无法从头合成 10k 真实补丁
  - 现有可用：lora_data 1500 张 + advpatchxai_data/train/patched 1000 张 = 2500 张
做法：对每张 positive 样本施加 N 种 EoT 变换（缩放/亮度/对比度/噪声/模糊/JPEG/旋转），
      生成 2500 → 2500+10k 的增广集（同分布变体，用于验证"数据量"效应）。
      负样本同样增广（保持类别平衡）。
输出: /home/oyp/fastdata/lora_data_aug/{images,index.json}
"""
import os, sys, json, argparse, random
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

OUT = "/home/oyp/fastdata/lora_data_aug"


def eot_variant(im, rng):
    """随机 EoT：缩放 → 亮度/对比度 → 模糊 → 噪声 → JPEG"""
    w, h = im.size
    # 1) 随机缩放（0.5–1.0），再贴回原尺寸背景（模拟距离变化）
    s = rng.uniform(0.5, 1.0)
    small = im.resize((max(32, int(w*s)), max(32, int(h*s))), Image.BILINEAR)
    if s < 0.999:
        canvas = Image.new("RGB", (w, h), (114, 114, 114))
        ox = rng.randint(0, max(0, w - small.width)); oy = rng.randint(0, max(0, h - small.height))
        canvas.paste(small, (ox, oy)); im = canvas
    # 2) 亮度/对比度
    im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.7, 1.3))
    im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.8, 1.2))
    # 3) 模糊（模拟失焦/运动）
    if rng.random() < 0.5:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.5)))
    # 4) 噪声
    if rng.random() < 0.5:
        a = np.asarray(im).astype(np.float32)
        a += rng.uniform(0, 12) * np.random.RandomState(rng.randint(0, 2**31-1)).randn(*a.shape)
        im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    return im


def jpeg_roundtrip(im, q):
    import io
    b = io.BytesIO(); im.save(b, "JPEG", quality=q); b.seek(0)
    return Image.open(b).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-image", type=int, default=6, help="每张生成多少变体")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jpeg-prob", type=float, default=0.6)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    os.makedirs(f"{OUT}/images", exist_ok=True)

    # 汇总源：lora_data 训练集 + advpatchxai train/patched（作为额外正样本）
    src = []
    idx = json.load(open("/home/oyp/fastdata/lora_data/index.json"))
    for r in idx:
        if r["split"] == "train": src.append((r["path"], r["label"], r.get("domain", "?")))
    extra = "/home/oyp/fastdata/advpatchxai_data/train/patched"
    if os.path.isdir(extra):
        for i, f in enumerate(sorted(os.listdir(extra))):
            src.append((os.path.join(extra, f), 1, "apx_patched"))
    print(f"[P1] 源样本 {len(src)} 张 → 每张 {args.per_image} 变体", flush=True)

    out_idx = []
    for base_id in range(args.per_image):
        for i, (p, y, dom) in enumerate(src):
            try:
                im = Image.open(p).convert("RGB")
            except Exception:
                continue
            v = eot_variant(im, rng)
            if rng.random() < args.jpeg_prob:
                v = jpeg_roundtrip(v, rng.choice([50, 65, 75, 85]))
            name = f"{dom}_{base_id}_{i}.jpg"
            v.save(f"{OUT}/images/{name}", quality=92)
            out_idx.append({"path": f"{OUT}/images/{name}", "label": y,
                            "domain": dom, "split": "train", "aug_id": base_id})
            if (i+1) % 500 == 0:
                print(f"  aug{base_id}: {i+1}/{len(src)}", flush=True)

    # 原始集也纳入（split=train）
    for p, y, dom in src:
        out_idx.append({"path": p, "label": y, "domain": dom, "split": "train", "aug_id": -1})
    json.dump(out_idx, open(f"{OUT}/index.json", "w"), ensure_ascii=False)
    from collections import Counter
    print(f"[P1] 完成：{len(out_idx)} 条 → {OUT}/index.json")
    print(f"     label={dict(Counter(r['label'] for r in out_idx))}")
    print(f"     domain={dict(Counter(r['domain'] for r in out_idx))}")


if __name__ == "__main__":
    main()
