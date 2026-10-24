#!/usr/bin/env python3
"""
黑盒迁移攻击（扩展实验）
==========================================
攻击者不能访问 VLM，只能访问一个代理检测器（此处为 CLIP ViT-B/32 + 线性头，
即论文主表中 88%@5% 的轻量基线）。在代理模型上做 PGD 制造漏报扰动，
再把扰动图像"迁移"给 VLM 评测 —— 检验超低误报检测器的迁移可攻击性。

与白盒攻击（adv_attack_v2.py）形成对照：白盒已知权重，黑盒仅知代理。
输出: results/tvc/blackbox_transfer.json
用法: $PY blackbox_transfer.py --n 50 --eps 0.03 --steps 20
"""
import os, sys, json, argparse
import numpy as np

WS = "/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard"
TVC = f"{WS}/results/tvc"
BASE = f"{TVC}/baseline_imgs"
sys.path.insert(0, f"{WS}/code/experiments_tvc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=f"{TVC}/lora_patch_detector")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--eps", type=float, default=0.03)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--out", default=f"{TVC}/blackbox_transfer.json")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from sklearn.linear_model import LogisticRegression
    from transformers import CLIPModel, CLIPImageProcessor
    from lora_mllm import load_lora_for_eval, PROMPT
    os.environ.setdefault("HF_HOME", "/home/oyp/.cache/huggingface")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    dev = "cuda"
    # ── 代理模型：CLIP + 线性头（与主表 CLIP 基线同源训练）──
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(dev).eval()
    cproc = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    idx = json.load(open(f"{TVC}/lora_data/index.json"))
    tr = [r for r in idx if r["split"] == "train"]
    X, Y = [], []
    with torch.no_grad():
        for i in range(0, len(tr), 32):
            b = tr[i:i+32]
            ims = [Image.open(r["path"]).convert("RGB") for r in b]
            px = cproc(images=ims, return_tensors="pt").pixel_values.to(dev)
            _feat = clip.visual_projection(clip.vision_model(pixel_values=px).pooler_output)
            X.append(_feat.cpu())
            Y += [r["label"] for r in b]
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(torch.cat(X).numpy(), np.array(Y))
    print("[proxy] CLIP linear head trained", flush=True)

    def clip_p(px):
        f = clip.visual_projection(clip.vision_model(pixel_values=px).pooler_output)
        return torch.tensor(clf.predict_proba(f.detach().cpu().numpy())[:, 1],
                            device=dev, dtype=torch.float32)

    # ── 目标模型：VLM ──
    model, proc = load_lora_for_eval(args.ckpt, 16)
    cfg = json.load(open(f"{args.ckpt}/train_config.json")); yes_id, no_id = cfg["yes_id"], cfg["no_id"]

    pos_dir = f"{BASE}/pos_all" if os.path.isdir(f"{BASE}/pos_all") else f"{BASE}/pos"
    files = sorted(os.listdir(pos_dir))[:args.n]

    def vlm_p(im):
        msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        t = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        enc = {k: (v.cuda() if torch.is_tensor(v) else v)
               for k, v in proc(text=[t], images=[im], return_tensors="pt").items()}
        with torch.no_grad():
            lg = model(**enc).logits[0, -1]
        return float(torch.sigmoid(lg[yes_id] - lg[no_id]))

    def to_img(t):
        a = t.detach().cpu().permute(1, 2, 0).numpy()
        a = np.clip(a * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(a)

    clean, adv = [], []
    for i, f in enumerate(files):
        im = Image.open(os.path.join(pos_dir, f)).convert("RGB")
        im_rs = im.resize((224, 224), Image.BILINEAR)
        clean.append(vlm_p(im_rs))
        # 代理模型上的 PGD（最小化代理的"有补丁"概率 = 制造漏报）
        px0 = cproc(images=[im_rs], return_tensors="pt").pixel_values.to(dev)
        px = px0.clone()
        alpha = args.eps / max(args.steps // 4, 1)
        for _ in range(args.steps):
            px.requires_grad_(True)
            p = clip_p(px)
            g, = torch.autograd.grad(p.sum(), px)
            with torch.no_grad():
                px = px - alpha * g.sign()
                px = torch.clamp(px, px0 - args.eps, px0 + args.eps)
            px = px.detach()
        adv.append(vlm_p(to_img(px[0])))
        if (i+1) % 10 == 0: print(f"  {i+1}/{len(files)}", flush=True)

    c, a = np.array(clean), np.array(adv)
    negs = [vlm_p(Image.open(f"{BASE}/casia_au/{f}").convert("RGB").resize((224, 224)))
            for f in sorted(os.listdir(f"{BASE}/casia_au"))[:60]]
    thr = float(np.percentile(negs, 95))
    res = {"proxy": "CLIP ViT-B/32 + linear head (black-box to the VLM)",
           "target_ckpt": os.path.basename(args.ckpt), "n": len(c),
           "eps": args.eps, "steps": args.steps, "threshold_from_casia_p95": round(thr, 4),
           "tpr_clean": round(float((c >= thr).mean()), 4),
           "tpr_transferred": round(float((a >= thr).mean()), 4),
           "asr_transfer": round(float((c >= thr).mean() - (a >= thr).mean()), 4),
           "p_mean_clean": round(float(c.mean()), 4), "p_mean_adv": round(float(a.mean()), 4)}
    print(f"[blackbox] TPR {res['tpr_clean']:.3f} -> {res['tpr_transferred']:.3f} "
          f"(transfer ASR {res['asr_transfer']:.3f})", flush=True)
    json.dump(res, open(args.out, "w"), indent=1, ensure_ascii=False)
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
