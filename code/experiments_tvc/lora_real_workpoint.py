#!/usr/bin/env python3
"""
真实数据工作点验证 — LoRA-MLLM 分支
=====================================
动机: 真实 print-and-capture。
正样本: APRICOT-Mask 中带 mask 的图（真实打印补丁 + 真实拍摄）
负样本: CASIA v2 Au 真实干净照片
模型: 本地 LoRA 微调的 Qwen2-VL-2B（不依赖 API，可复现）

用法:
  $PY code/experiments_tvc/lora_real_workpoint.py --n-pos 100 --n-neg 200
输出: results/tvc/lora_real_workpoint.json
"""
import argparse, json, os, sys, time

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)
OUT = os.path.join(os.path.dirname(CODE_DIR), "results", "tvc")
APRICOT_D = "/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard/data/apricot/dev/data_mask_v2"
CASIA_AU = "/media/oyp/工作/Projects/042_image_forensic/DuetGuard/data/casia2/CASIA2.0_revised/Au"
CKPT = os.path.join(OUT, "lora_patch_detector")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pos", type=int, default=100)
    ap.add_argument("--n-neg", type=int, default=200)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--ckpt", default=None,
                    help="LoRA 权重目录；默认用 RL 版 (lora_rl_ckpt)，次选 SFT 版")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import numpy as np
    import torch
    from PIL import Image
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from lora_mllm import QWEN_SNAP, PROMPT, load_lora_for_eval

    ckpt = args.ckpt
    if ckpt is None:
        for c in (os.path.join(OUT, "lora_rl_ckpt"), CKPT):
            if os.path.isdir(c):
                ckpt = c
                break
    if not ckpt or not os.path.isdir(ckpt):
        raise SystemExit(f"缺少 LoRA ckpt（tried {OUT}/lora_rl_ckpt 和 {CKPT}）")
    print(f"[real] ckpt = {ckpt}")
    model, proc = load_lora_for_eval(ckpt, args.rank)
    yes_id, no_id = 14004, 8996

    jobs = []
    fs = sorted(os.listdir(APRICOT_D), key=lambda f: int(f.split("_")[1].split(".")[0]))
    np_ = 0
    for f in fs:
        if np_ >= args.n_pos:
            break
        x = torch.load(os.path.join(APRICOT_D, f), map_location="cpu", weights_only=False)
        if x["Mask"].sum() <= 0:
            continue
        arr = np.clip(x["Image"][0].astype(np.float32) * 255.0, 0, 255).astype(np.uint8)
        im = Image.fromarray(arr).convert("RGB")
        im.thumbnail((1024, 1024))          # 原图 4608x2304 不缩放会 OOM
        jobs.append(("apricot", 1, f, im))
        np_ += 1
    au = sorted(fn for fn in os.listdir(CASIA_AU)
                if fn.lower().endswith((".jpg", ".jpeg", ".png")))
    rng = np.random.RandomState(42)
    for i in rng.permutation(len(au))[:args.n_neg]:
        im = Image.open(os.path.join(CASIA_AU, au[i])).convert("RGB")
        im.thumbnail((1024, 1024))
        jobs.append(("casia_au", 0, au[i], im))
    print(f"[real] pos={np_} neg={len(jobs)-np_}")

    scores, labels, names = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for k, (src, lab, name, img) in enumerate(jobs):
            msg = [{"role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
            t = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            enc = {kk: (v.cuda() if torch.is_tensor(v) else v)
                   for kk, v in proc(text=[t], images=[img], return_tensors="pt").items()}
            lg = model(**enc).logits[0, -1]
            scores.append(float(torch.sigmoid(lg[yes_id] - lg[no_id])))
            labels.append(lab); names.append(name)
            if (k + 1) % 25 == 0:
                print(f"  {k+1}/{len(jobs)} ({time.time()-t0:.0f}s)", flush=True)

    p = np.array(scores); y = np.array(labels)
    pos, neg = y == 1, y == 0
    th = np.unique(np.percentile(p[neg], np.linspace(0, 100, 201)))
    sweep = [{"tau": round(float(t), 4),
              "tpr": round(float(((p >= t) & pos).sum() / pos.sum()), 4),
              "fpr": round(float(((p >= t) & neg).sum() / neg.sum()), 4)} for t in th]
    rep = {"dataset": "APRICOT(real pos) vs CASIA-Au(real neg)",
           "model": "LoRA-Qwen2-VL-2B", "n_pos": int(pos.sum()), "n_neg": int(neg.sum()),
           "sweep": sweep, "scores": [round(float(s), 6) for s in p],
           "labels": [int(v) for v in y],
           "seconds": round(time.time() - t0, 1)}
    for b in (0.01, 0.05, 0.10, 0.20, 0.33):
        cand = [s for s in sweep if s["fpr"] <= b]
        rep[f"best_tpr_at_fpr<={int(b*100)}pct"] = max(cand, key=lambda s: s["tpr"]) if cand else None
    out = args.out or os.path.join(OUT, "lora_real_workpoint.json")
    json.dump(rep, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"[real] *** LoRA-MLLM 真实数据:")
    for k in ("best_tpr_at_fpr<=1pct", "best_tpr_at_fpr<=5pct",
              "best_tpr_at_fpr<=10pct", "best_tpr_at_fpr<=20pct"):
        print(f"    {k}: {rep[k]}")
    print(f"[real] wrote {out}")


if __name__ == "__main__":
    main()
