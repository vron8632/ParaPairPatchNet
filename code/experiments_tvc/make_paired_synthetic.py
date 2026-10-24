#!/usr/bin/env python3
"""
构造「同背景合成对」：背景照片 C 与 C+patch（真实补丁纹理），用于背景对照损失。

数据来源（均与评测集无泄漏）：
  补丁纹理：APDE train parquet 的 image+mask（8 个 seen 攻击族，与 APRICOT 无关）
  背景照片：lora clean 500 张（CASIA）+ coco_train[20000:22000]（**避开** 21k 评测负样本用的前 20000 张）

输出：/home/oyp/fastdata/paired_syn/
  pos/xxx.jpg  neg/xxx.jpg  index.json [{pair_id, path_pos, path_neg, area, patch_src}]
"""
import os, json, glob, io, argparse, random
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--n-pairs", type=int, default=2000)
ap.add_argument("--n-texture-files", type=int, default=20, help="读多少个 APDE train parquet 裁纹理")
ap.add_argument("--per-file", type=int, default=25)
ap.add_argument("--n-coco-bg", type=int, default=1500)
ap.add_argument("--out", default="/home/oyp/fastdata/paired_syn")
ap.add_argument("--long-side", type=int, default=768)
ap.add_argument("--area-lo", type=float, default=0.01)
ap.add_argument("--area-hi", type=float, default=0.05)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rng = random.Random(a.seed)
OUT = a.out
os.makedirs(f"{OUT}/pos", exist_ok=True); os.makedirs(f"{OUT}/neg", exist_ok=True)

D = "/media/oyp/工作/Projects/042_image_forensic"
AXAI_TRAIN = "/home/oyp/fastdata/apde_full/data/train"

# ── 1) 补丁纹理库 ──
def crop_patch_textures():
    fs = sorted(glob.glob(f"{AXAI_TRAIN}/*.parquet"))[: a.n_texture_files]
    lib = []
    for f in fs:
        try:
            t = pq.read_table(f, columns=["image", "mask"]).slice(0, a.per_file).to_pydict()
        except Exception as e:
            print("  skip", f, e); continue
        tag = os.path.basename(f).replace(".parquet", "")
        for i in range(len(t["image"])):
            im, mk = t["image"][i], t["mask"][i]
            if not (isinstance(im, dict) and im.get("bytes") and isinstance(mk, dict) and mk.get("bytes")):
                continue
            try:
                img = Image.open(io.BytesIO(im["bytes"])).convert("RGB")
                m = np.asarray(Image.open(io.BytesIO(mk["bytes"])).convert("L"))
            except Exception:
                continue
            ys, xs = np.where(m > 0)
            if len(xs) < 100: continue
            x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
            pw, ph = x1 - x0 + 1, y1 - y0 + 1
            pad = int(0.15 * max(pw, ph))
            x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
            x1, y1 = min(img.width - 1, x1 + pad), min(img.height - 1, y1 + pad)
            if x1 - x0 < 24 or y1 - y0 < 24: continue
            patch = img.crop((x0, y0, x1 + 1, y1 + 1))
            buf = io.BytesIO(); patch.save(buf, format="PNG")
            lib.append({"png": buf.getvalue(), "src": tag, "ar": patch.width / patch.height})
        print(f"  纹理 {tag}: 累计 {len(lib)}", flush=True)
        if len(lib) >= 600: break
    return lib


def paste(bg_im, patch_im, area_frac):
    """把补丁纹理按目标面积占比贴到背景上，返回贴好的图与实际面积占比"""
    W, H = bg_im.size
    target = area_frac * W * H
    pw, ph = patch_im.size
    s = (target / (pw * ph)) ** 0.5
    tw, th = max(12, int(pw * s)), max(12, int(ph * s))
    p = patch_im.resize((tw, th), Image.BILINEAR)
    x = rng.randint(0, max(0, W - tw)); y = rng.randint(0, max(0, H - th))
    out = bg_im.copy(); out.paste(p, (x, y))
    return out, (tw * th) / (W * H)


def main():
    print("裁补丁纹理 ...", flush=True)
    lib = crop_patch_textures()
    print(f"补丁纹理库: {len(lib)} 块", flush=True)

    lora = json.load(open(f"{WS_DIR}/results/tvc/lora_data/index.json"))
    bg = [r["path"] for r in lora if r["split"] == "train" and r["label"] == 0]
    coco = [r["path"] for r in json.load(open("/home/oyp/fastdata/apde_negs/index.json"))]
    bg += coco[20000: 20000 + a.n_coco_bg]          # 避开评测负样本的前 20000 张
    print(f"背景库: {len(bg)} 张", flush=True)

    idx = []
    bg_idx = list(range(len(bg))); rng.shuffle(bg_idx)
    for k in range(a.n_pairs):
        bpath = bg[bg_idx[k % len(bg_idx)]]
        tex = lib[rng.randrange(len(lib))]
        try:
            bim = Image.open(bpath).convert("RGB")
        except Exception:
            continue
        w, h = bim.size
        s = a.long_side / max(w, h)
        if s < 1: bim = bim.resize((int(w * s), int(h * s)), Image.BILINEAR)
        try: pim = Image.open(io.BytesIO(tex["png"]))
        except Exception: continue
        af = rng.uniform(a.area_lo, a.area_hi)
        pos_im, real_af = paste(bim, pim, af)
        pn = f"pos/p{k:05d}.jpg"; nn = f"neg/n{k:05d}.jpg"
        pos_im.save(f"{OUT}/{pn}", quality=92)
        bim.save(f"{OUT}/{nn}", quality=92)
        idx.append({"pair_id": k, "path_pos": f"{OUT}/{pn}", "path_neg": f"{OUT}/{nn}",
                    "area": round(real_af, 4), "patch_src": tex["src"]})
        if (k + 1) % 400 == 0: print(f"  合成 {k+1}/{a.n_pairs}", flush=True)
    json.dump(idx, open(f"{OUT}/index.json", "w"), indent=1, ensure_ascii=False)
    print(f"✅ 合成对 {len(idx)} → {OUT}/index.json", flush=True)


WS_DIR = "/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard"
if __name__ == "__main__":
    main()
