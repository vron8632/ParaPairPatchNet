#!/usr/bin/env python3
"""
主表（n=1000 负样本口径）
================================================================================
对 3 个训练种子，在 138 正样本（APRICOT 本地全部）+ 1000 负样本（4 源各 250 张，
与 DKW 认证口径一致）上评测，输出三源/四源双口径 TPR@FPR + Wilson CI。
负样本直接从原始数据集按固定种子抽取（复用 eval_neg_scale 的采集逻辑）。
输出: results/tvc/main_table_1000neg.json
"""
import os, sys, json, argparse, math, time
import numpy as np

WS = "/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard"
TVC = f"{WS}/results/tvc"
D = "/media/oyp/工作/Projects/042_image_forensic/DuetGuard/data"
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
S3 = ["casia_au", "coco_val", "inria_noped"]
S4 = S3 + ["columbia_au"]
sys.path.insert(0, f"{WS}/code/experiments_tvc")


def collect_neg(n_per, sources, seed=42):
    from PIL import Image
    rng = np.random.RandomState(seed)
    out = {}
    def _pick(dirpath, tag, filt=None):
        if not os.path.isdir(dirpath): print(f"  [skip] {tag}"); return
        fs = sorted(f for f in os.listdir(dirpath)
                    if f.lower().endswith(IMG_EXT) and (filt is None or filt(f)))
        if not fs: print(f"  [skip] {tag}: 无图像"); return
        pick = [fs[i] for i in rng.permutation(len(fs))[:n_per]]
        items = []
        for f in pick:
            try:
                im = Image.open(os.path.join(dirpath, f)).convert("RGB")
                im.thumbnail((1024, 1024)); items.append((f, im))
            except Exception: pass
        out[tag] = items
        print(f"  {tag}: {len(items)}/{len(fs)}", flush=True)
    if "casia_au" in sources: _pick(f"{D}/casia2/CASIA2.0_revised/Au", "casia_au")
    if "coco_val" in sources: _pick(f"{D}/coco/val2017", "coco_val")
    if "inria_noped" in sources: _pick(f"{D}/inria/data_ped/no_pedestrians", "inria_noped")
    if "columbia_au" in sources:
        # ⚠️ Columbia 必须只取 Au-*（authentic）；Sp-* 是拼接篡改图（见 number_imgs.txt）
        col = f"{D}/columbia/ImSpliceDataset"
        if os.path.isdir(col):
            from PIL import Image
            items = []
            for sub in sorted(os.listdir(col)):
                d2 = os.path.join(col, sub)
                if not os.path.isdir(d2) or not sub.startswith("Au"):
                    continue
                for fn in sorted(os.listdir(d2)):
                    if fn.lower().endswith(IMG_EXT):
                        items.append((f"{sub}/{fn}", os.path.join(d2, fn)))
            rng.shuffle(items)
            loaded = []
            for nm, fp in items[:n_per]:
                try:
                    im = Image.open(fp).convert("RGB"); im.thumbnail((1024, 1024))
                    loaded.append((nm, im))
                except Exception: pass
            out["columbia_au"] = loaded
            print(f"  columbia_au: {len(loaded)}/{len(items)} (authentic only)", flush=True)
    return out


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k/n; d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (max(0.0, c-h), min(1.0, c+h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", default="lora_patch_detector,full_lora_s1,full_lora_s2")
    ap.add_argument("--n-per-source", type=int, default=250)
    ap.add_argument("--out", default=f"{TVC}/main_table_1000neg.json")
    args = ap.parse_args()

    import torch
    from lora_mllm import load_lora_for_eval, PROMPT

    print("=== 采集负样本（4 源）===", flush=True)
    neg_imgs = collect_neg(args.n_per_source, S4)
    pos_dir = f"{TVC}/baseline_imgs/pos_all"
    pos_files = sorted(os.listdir(pos_dir))
    print(f"  正样本: {len(pos_files)}", flush=True)

    res = {}
    for ck in args.ckpts.split(","):
        ckpt = f"{TVC}/{ck}"
        if not os.path.isdir(ckpt): print(f"[skip] {ck}"); continue
        model, proc = load_lora_for_eval(ckpt, 16)
        cfg = json.load(open(f"{ckpt}/train_config.json")); yes_id, no_id = cfg["yes_id"], cfg["no_id"]

        def score_one(im):
            im = im.copy(); im.thumbnail((512, 512))
            msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
            t = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            enc = {k: (v.cuda() if torch.is_tensor(v) else v)
                   for k, v in proc(text=[t], images=[im], return_tensors="pt").items()}
            with torch.no_grad():
                lg = model(**enc).logits[0, -1]
            return float(torch.sigmoid(lg[yes_id] - lg[no_id]))

        from PIL import Image
        pos_s = []
        for i, f in enumerate(pos_files):
            pos_s.append(score_one(Image.open(os.path.join(pos_dir, f)).convert("RGB")))
            if (i+1) % 50 == 0: print(f"  [{ck}] pos {i+1}/{len(pos_files)}", flush=True)
        neg_s = {}
        for tag, items in neg_imgs.items():
            s = []
            for j, (fn, im) in enumerate(items):
                s.append(score_one(im))
                if (j+1) % 100 == 0: print(f"  [{ck}] neg/{tag} {j+1}/{len(items)}", flush=True)
            neg_s[tag] = np.array(s)
        pos = np.array(pos_s)
        out = {"n_pos": len(pos), "n_neg": {k: len(v) for k, v in neg_s.items()},
               "pos_mean": round(float(pos.mean()), 4),
               "neg_mean": {k: round(float(v.mean()), 4) for k, v in neg_s.items()},
               "tpr": {}}
        for scope, keys in (("3src", S3), ("4src", S4)):
            n = np.concatenate([neg_s[k] for k in keys]); row = {}
            for a in (0.01, 0.05, 0.10, 0.20):
                thr = np.percentile(n, 100*(1-a)); k_ = int((pos >= thr).sum())
                lo, hi = wilson(k_, len(pos))
                row[f"@{int(a*100)}pct"] = {"tpr": round(k_/len(pos), 4), "wilson": [round(lo, 4), round(hi, 4)]}
            out["tpr"][scope] = row
            print(f"  [{ck}] {scope} n_neg={len(n)}: " +
                  "  ".join(f"{k}={v['tpr']:.3f}" for k, v in row.items()), flush=True)
        res[ck] = out
        del model
        torch.cuda.empty_cache()

    if len(res) > 1:
        summ = {}
        for scope in ("3src", "4src"):
            for a in (0.01, 0.05, 0.10, 0.20):
                vals = [r["tpr"][scope][f"@{int(a*100)}pct"]["tpr"] for r in res.values()]
                summ[f"{scope}@{int(a*100)}pct"] = {"mean": round(float(np.mean(vals)), 4),
                                                     "std": round(float(np.std(vals)), 4),
                                                     "all": [round(v, 4) for v in vals]}
        res["summary_mean_std"] = summ
        print("\n=== 跨 seed 汇总 ===")
        for k, v in summ.items(): print(f"  {k}: {v['mean']:.4f}±{v['std']:.4f} {v['all']}")
    json.dump(res, open(args.out, "w"), indent=1, ensure_ascii=False)
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
