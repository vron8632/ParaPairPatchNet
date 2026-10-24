#!/usr/bin/env python3
"""
统一协议主表（用户决策：长边 1024 + 1000 负样本）
====================================================
所有方法在同一输入处理（长边 1024）+ 同一负样本集（4 源各 250 = 1000，
authentic columbia）+ 同一阈值规则（负样本分位数）+ 同一正样本集（138）
下评测，报告三源/四源双口径 TPR@FPR + Wilson CI。
本脚本覆盖 NP-PatchLM（3 seed）与 CLIP 线性头；其余基线由 unified_baselines_1024.py 覆盖。
输出: results/tvc/unified_main_1024.json
"""
import os, sys, json, argparse, math
os.environ["HF_HOME"] = os.environ.get("HF_HOME") or "/home/oyp/.cache/huggingface"
os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np
from PIL import Image

WS = "/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard"
TVC = f"{WS}/results/tvc"
D = "/media/oyp/工作/Projects/042_image_forensic/DuetGuard/data"
BASE = f"{TVC}/baseline_imgs"
IMG_EXT = (".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff")
S3 = ["casia_au", "coco_val", "inria_noped"]
S4 = S3 + ["columbia_au"]
sys.path.insert(0, f"{WS}/code/experiments_tvc")


def _fit_long_side(im, size):
    """统一输入契约：按长边缩放到 size（小图放大、大图缩小），保持宽高比。"""
    w, h = im.size
    s = size / max(w, h)
    if abs(s - 1.0) < 1e-6:
        return im
    return im.resize((max(1, int(round(w*s))), max(1, int(round(h*s)))), Image.BILINEAR)

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k/n; d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (round(max(0.0, c-h), 4), round(min(1.0, c+h), 4))


def collect_neg(n_per=250, seed=42):
    """4 源各 n_per 张（authentic columbia）。返回 {src: [(name, PIL.Image)]}"""
    from PIL import Image
    rng = np.random.RandomState(seed)
    out = {}

    def pick_dir(path, tag, filt=None):
        fs = sorted(f for f in os.listdir(path)
                    if f.lower().endswith(IMG_EXT) and (filt is None or filt(f)))
        sel = [fs[i] for i in rng.permutation(len(fs))[:n_per]]
        out[tag] = [(f, Image.open(os.path.join(path, f)).convert("RGB")) for f in sel]
        print(f"  {tag}: {len(out[tag])}/{len(fs)}", flush=True)

    if os.path.isdir(f"{D}/casia2/CASIA2.0_revised/Au"): pick_dir(f"{D}/casia2/CASIA2.0_revised/Au", "casia_au")
    if os.path.isdir(f"{D}/coco/val2017"): pick_dir(f"{D}/coco/val2017", "coco_val")
    if os.path.isdir(f"{D}/inria/data_ped/no_pedestrians"): pick_dir(f"{D}/inria/data_ped/no_pedestrians", "inria_noped")
    col = f"{D}/columbia/ImSpliceDataset"   # 只取 Au-*（authentic）
    items = []
    for sub in sorted(os.listdir(col)):
        d2 = os.path.join(col, sub)
        if not os.path.isdir(d2) or not sub.startswith("Au"): continue
        for f in sorted(os.listdir(d2)):
            if f.lower().endswith(IMG_EXT): items.append((f"{sub}/{f}", os.path.join(d2, f)))
    rng.shuffle(items)
    out["columbia_au"] = [(nm, Image.open(fp).convert("RGB")) for nm, fp in items[:n_per]]
    print(f"  columbia_au (authentic): {len(out['columbia_au'])}/{len(items)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=1024, help="评测长边（统一协议=1024）")
    ap.add_argument("--n-per-source", type=int, default=250)
    ap.add_argument("--ckpts", default="lora_patch_detector,full_lora_s1,full_lora_s2")
    ap.add_argument("--skip-clip", action="store_true")
    ap.add_argument("--out", default=f"{TVC}/unified_main_1024.json")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from lora_mllm import load_lora_for_eval, PROMPT

    print("=== 采集负样本 ===", flush=True)
    neg = collect_neg(args.n_per_source)
    pos_files = sorted(os.listdir(f"{BASE}/pos_all"))
    pos_imgs = [Image.open(f"{BASE}/pos_all/{f}").convert("RGB") for f in pos_files]
    print(f"  正样本: {len(pos_imgs)}", flush=True)

    res = {"protocol": {"long_side": args.size, "n_pos": len(pos_imgs),
                        "n_per_source": args.n_per_source}}

    # ── 各 VLM ckpt ──
    for ck in args.ckpts.split(","):
        ckpt = f"{TVC}/{ck}"
        if not os.path.isdir(ckpt): print(f"[skip] {ck}"); continue
        model, proc = load_lora_for_eval(ckpt, 16)
        cfg = json.load(open(f"{ckpt}/train_config.json")); yi, ni = cfg["yes_id"], cfg["no_id"]

        def score(im):
            im = _fit_long_side(im, args.size)   # 统一契约：长边缩放到 args.size
            msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
            t = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            enc = {k: (v.cuda() if torch.is_tensor(v) else v)
                   for k, v in proc(text=[t], images=[im], return_tensors="pt").items()}
            with torch.no_grad():
                lg = model(**enc).logits[0, -1]
            return float(torch.sigmoid(lg[yi] - lg[ni]))

        p = np.array([score(im) for im in pos_imgs])
        ns = {k: np.array([score(im) for _, im in v]) for k, v in neg.items()}
        row = {"n_pos": len(p), "pos_mean": round(float(p.mean()), 4),
               "neg_mean": {k: round(float(v.mean()), 4) for k, v in ns.items()}, "tpr": {}}
        for scope, keys in (("3src", S3), ("4src", S4)):
            n_all = np.concatenate([ns[k] for k in keys]); rr = {}
            for a in (0.01, 0.05, 0.10, 0.20):
                thr = float(np.percentile(n_all, 100*(1-a))); k_ = int((p >= thr).sum())
                rr[f"@{int(a*100)}pct"] = {"tpr": round(k_/len(p), 4), "wilson": wilson(k_, len(p)),
                                           "thr": round(thr, 4)}
            row["tpr"][scope] = rr
        res[ck] = row
        print(f"  [{ck}] 3src: " + "  ".join(f"{a}={row['tpr']['3src'][f'@{a}pct']['tpr']:.3f}" for a in (1,5,10,20)), flush=True)
        print(f"  [{ck}] 4src: " + "  ".join(f"{a}={row['tpr']['4src'][f'@{a}pct']['tpr']:.3f}" for a in (1,5,10,20)), flush=True)
        del model; torch.cuda.empty_cache()

    # ── CLIP 线性头（同 1024 输入）──
    if not args.skip_clip:
        from transformers import CLIPModel, CLIPImageProcessor
        from sklearn.linear_model import LogisticRegression
        cm = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda().eval()
        cp = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        def cfeat(im):
            px = cp(images=[im], size={"height": args.size, "width": args.size},
                    crop_size={"height": args.size, "width": args.size}, return_tensors="pt").pixel_values.cuda()
            with torch.no_grad():
                po = cm.vision_model(pixel_values=px, interpolate_pos_encoding=(args.size != 224)).pooler_output
                return cm.visual_projection(po)
        idx = json.load(open(f"{TVC}/lora_data/index.json"))
        tr = [r for r in idx if r["split"] == "train"]
        X, Y = [], []
        for i in range(0, len(tr), 32):
            b = tr[i:i+32]
            for r in b:
                X.append(cfeat(Image.open(r["path"]).convert("RGB")))
            Y += [r["label"] for r in b]
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(torch.cat(X).cpu().numpy(), np.array(Y))
        p = clf.predict_proba(torch.cat([cfeat(im) for im in pos_imgs]).cpu().numpy())[:, 1]
        ns = {}
        for k, v in neg.items():
            ns[k] = clf.predict_proba(torch.cat([cfeat(im) for _, im in v]).cpu().numpy())[:, 1]
        row = {"n_pos": len(p), "pos_mean": round(float(p.mean()), 4),
               "neg_mean": {k: round(float(v.mean()), 4) for k, v in ns.items()}, "tpr": {}}
        for scope, keys in (("3src", S3), ("4src", S4)):
            n_all = np.concatenate([ns[k] for k in keys]); rr = {}
            for a in (0.01, 0.05, 0.10, 0.20):
                thr = float(np.percentile(n_all, 100*(1-a))); k_ = int((p >= thr).sum())
                rr[f"@{int(a*100)}pct"] = {"tpr": round(k_/len(p), 4), "wilson": wilson(k_, len(p))}
            row["tpr"][scope] = rr
        res["CLIP-linear"] = row
        print("  [CLIP-linear] 3src: " + "  ".join(f"{a}={row['tpr']['3src'][f'@{a}pct']['tpr']:.3f}" for a in (1,5,10,20)), flush=True)

    # 跨 seed 汇总
    keys = [k for k in res if k not in ("protocol", "CLIP-linear") and isinstance(res[k], dict) and "tpr" in res[k]]
    if len(keys) > 1:
        summ = {}
        for scope in ("3src", "4src"):
            for a in (0.01, 0.05, 0.10, 0.20):
                vals = [res[k]["tpr"][scope][f"@{int(a*100)}pct"]["tpr"] for k in keys]
                summ[f"{scope}@{int(a*100)}pct"] = {"mean": round(float(np.mean(vals)), 4),
                                                     "std": round(float(np.std(vals)), 4),
                                                     "all": [round(v, 4) for v in vals]}
        res["summary_vlm_mean_std"] = summ
        print("\n=== VLM 跨 seed 汇总 ===")
        for k, v in summ.items(): print(f"  {k}: {v['mean']:.4f}±{v['std']:.4f} {v['all']}")
    json.dump(res, open(args.out, "w"), indent=1, ensure_ascii=False)
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
