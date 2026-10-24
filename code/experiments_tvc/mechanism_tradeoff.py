#!/usr/bin/env python3
"""
机制研究 M3+M4
================
M3（低误报 × 跨范式 trade-off）：对 λ∈{0.5, 1.0, 2.0} 的 ckpt，
   分别在 APRICOT（中等跨域）与 APDE（远跨域）上测 TPR@5%，看约束强度与跨范式能力的关系。
M4（尺度匹配假设）：同一模型在多个评测分辨率下测 APRICOT，
   并用掩码真值计算补丁像素面积 → 关联"补丁尺度"与"检出率"。
输出: results/tvc/mechanism_tradeoff.json
"""
import os, sys, json, argparse
os.environ.setdefault("HF_HOME", "/home/oyp/.cache/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np

WS = "/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard"
TVC = f"{WS}/results/tvc"
EQ = f"{TVC}/unified_eval_set"
APDE = f"{TVC}/apde_imgs"
sys.path.insert(0, f"{WS}/code/experiments_tvc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambda-ckpts", default="hp_lam0.5:0.5,abl_both_soft:1.0,hp_lam2.0:2.0")
    ap.add_argument("--resolutions", default="448,640,1024,1536")
    ap.add_argument("--size-ckpt", default="lora_patch_detector")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default=f"{TVC}/mechanism_tradeoff.json")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from lora_mllm import load_lora_for_eval, PROMPT

    def fit_long(im, size):
        w, h = im.size; s = size / max(w, h)
        return im if abs(s-1) < 1e-6 else im.resize((max(1, int(w*s)), max(1, int(h*s))), Image.BILINEAR)

    pos_a = [Image.open(f"{EQ}/pos/{f}").convert("RGB") for f in sorted(os.listdir(f"{EQ}/pos"))[:args.n]]
    pos_p = [Image.open(f"{APDE}/pos/{f}").convert("RGB") for f in sorted(os.listdir(f"{APDE}/pos"))[:args.n]]
    # 负样本用完整统一评测集（4 源各 250 = 1000 张），避免小样本阈值不稳
    neg = [Image.open(f"{EQ}/{s}/{f}").convert("RGB")
           for s in ("casia_au", "coco_val", "inria_noped", "columbia_au")
           for f in sorted(os.listdir(f"{EQ}/{s}"))]
    print(f"  数据: APRICOT {len(pos_a)} / APDE {len(pos_p)} / 干净 {len(neg)}", flush=True)
    res = {}

    # ── M3：λ 与跨范式的关系 ──
    m3 = {}
    for tag, lam in [x.split(":") for x in args.lambda_ckpts.split(",")]:
        ckpt = f"{TVC}/{tag}"
        if not os.path.isdir(ckpt): print(f"  [M3] {tag} 缺失，跳过"); continue
        model, proc = load_lora_for_eval(ckpt, 16)
        cfg = json.load(open(f"{ckpt}/train_config.json")); yi, ni = cfg["yes_id"], cfg["no_id"]

        def sc(im, size=1024):
            im = fit_long(im, size)
            msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
            t = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            enc = {k: (v.cuda() if torch.is_tensor(v) else v)
                   for k, v in proc(text=[t], images=[im], return_tensors="pt").items()}
            with torch.no_grad():
                lg = model(**enc).logits[0, -1]
            return float(torch.sigmoid(lg[yi] - lg[ni]))

        sn = np.array([sc(im) for im in neg])
        sa = np.array([sc(im) for im in pos_a]); sp = np.array([sc(im) for im in pos_p])
        row = {"lam": float(lam)}
        # APRICOT 用统一集负样本作参考；APDE 用其自带负样本（保持各自协议）
        ade_neg = [Image.open(f"{APDE}/neg/{f}").convert("RGB") for f in sorted(os.listdir(f"{APDE}/neg"))[:args.n]]
        sne = np.array([sc(im) for im in ade_neg])
        for a in (0.05, 0.10):
            row[f"apricot@{int(a*100)}"] = round(float((sa >= np.percentile(sn, 100*(1-a))).mean()), 4)
            row[f"apde@{int(a*100)}"] = round(float((sp >= np.percentile(sne, 100*(1-a))).mean()), 4)
        m3[tag] = row
        print(f"  [M3] λ={lam}: APRICOT@5%={row['apricot@5']}  APDE@5%={row['apde@5']}", flush=True)
        del model; torch.cuda.empty_cache()
    res["M3_lambda_vs_crossparadigm"] = m3

    # ── M4：评测分辨率 vs 检出率（同一 ckpt）──
    ckpt = f"{TVC}/{args.size_ckpt}"
    if os.path.isdir(ckpt):
        model, proc = load_lora_for_eval(ckpt, 16)
        cfg = json.load(open(f"{ckpt}/train_config.json")); yi, ni = cfg["yes_id"], cfg["no_id"]

        def sc2(im, size):
            im = fit_long(im, size)
            msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
            t = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            enc = {k: (v.cuda() if torch.is_tensor(v) else v)
                   for k, v in proc(text=[t], images=[im], return_tensors="pt").items()}
            with torch.no_grad():
                lg = model(**enc).logits[0, -1]
            return float(torch.sigmoid(lg[yi] - lg[ni]))

        m4 = {}
        for size in [int(x) for x in args.resolutions.split(",")]:
            sa = np.array([sc2(im, size) for im in pos_a])
            sn = np.array([sc2(im, size) for im in neg])
            m4[str(size)] = {"tpr@5": round(float((sa >= np.percentile(sn, 95)).mean()), 4),
                             "tpr@10": round(float((sa >= np.percentile(sn, 90)).mean()), 4),
                             "pos_mean": round(float(sa.mean()), 4)}
            print(f"  [M4] 评测分辨率 {size}: TPR@5%={m4[str(size)]['tpr@5']} @10%={m4[str(size)]['tpr@10']}", flush=True)
        res["M4_resolution_vs_recall"] = m4
        del model; torch.cuda.empty_cache()

    json.dump(res, open(args.out, "w"), indent=1, ensure_ascii=False)
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
