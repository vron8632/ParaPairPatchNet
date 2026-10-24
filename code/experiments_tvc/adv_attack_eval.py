#!/usr/bin/env python3
"""
自适应攻击实验（白盒 PGD）—— 回应 PATCHOUT 已做而我们缺失的评估
================================================================
动机：PATCHOUT (Neural Processing Letters 2025) 报告其检测器可被**白盒
      detector-focused 自适应补丁**完全破坏（≈0%）。需要检查我们的检测器
      对自适应攻击的鲁棒性。本实验给出**首个**答案。

设定：
  - 白盒：攻击者已知模型权重与 prompt。
  - 目标：对**含补丁图（正样本）**最小化 P(YES)，即制造**漏报**。
  - 扰动：在 Qwen2-VL 的 `pixel_values` 空间做 ε-受限 PGD（L∞）。
  - 报告：攻击前后 P(YES) 分布、固定工作点下 TPR 下降、攻击成功率 ASR。

用法:
  $PY code/experiments_tvc/adv_attack_eval.py \
      --ckpt results/tvc/lora_patch_detector --n 50 --eps 0.03 --steps 10 \
      --out results/tvc/adv_attack_pgd.json
"""
import argparse, json, os, sys, time

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)
WS = os.path.dirname(CODE_DIR)
OUT = os.path.join(WS, "results", "tvc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(OUT, "lora_patch_detector"))
    ap.add_argument("--index", default=os.path.join(OUT, "lora_data", "index.json"))
    ap.add_argument("--domain", default="physical_patch", help="攻击的目标域（正样本）")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--eps", type=float, default=0.03, help="L∞ 扰动上限（pixel_values 归一化空间）")
    ap.add_argument("--alpha", type=float, default=None, help="每步步长（默认 eps/4）")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(OUT, "adv_attack_pgd.json"))
    args = ap.parse_args()
    if args.alpha is None:
        args.alpha = args.eps / 4.0

    import numpy as np
    import torch
    from PIL import Image
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from lora_mllm import load_lora_for_eval, PROMPT

    rows = json.load(open(args.index))
    pos = [r for r in rows if r["domain"] == args.domain and r["split"] == "val"][:args.n]
    print(f"[adv] 目标域={args.domain} n={len(pos)} eps={args.eps} alpha={args.alpha} steps={args.steps}")

    model, proc = load_lora_for_eval(args.ckpt, args.rank)
    cfg = json.load(open(os.path.join(args.ckpt, "train_config.json")))
    yes_id, no_id = cfg["yes_id"], cfg["no_id"]

    def prepare(img):
        msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        t = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        enc = {k: (v.cuda() if torch.is_tensor(v) else v)
               for k, v in proc(text=[t], images=[img], return_tensors="pt").items()}
        return enc

    def p_yes(enc, pv=None):
        kw = dict(enc)
        if pv is not None:
            kw["pixel_values"] = pv
        lg = model(**kw).logits[0, -1]
        return torch.sigmoid(lg[yes_id] - lg[no_id])

    clean_scores, adv_scores, t0 = [], [], time.time()
    for i, r in enumerate(pos):
        img = Image.open(r["path"]).convert("RGB")
        enc = prepare(img)
        with torch.no_grad():
            s0 = float(p_yes(enc))
        # ---- PGD ----
        pv0 = enc["pixel_values"].detach().clone()
        pv = pv0.clone()
        for _ in range(args.steps):
            pv = pv.detach().clone().requires_grad_(True)
            s = p_yes(enc, pv)
            loss = torch.log(s + 1e-8)          # 最小化 log P(YES) ⇔ 降低 P(YES)
            g = torch.autograd.grad(loss, pv)[0]
            with torch.no_grad():
                pv = pv - args.alpha * g.sign()
                pv = torch.clamp(pv, pv0 - args.eps, pv0 + args.eps)
        with torch.no_grad():
            s1 = float(p_yes(enc, pv))
        clean_scores.append(s0); adv_scores.append(s1)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(pos)}  clean={np.mean(clean_scores):.3f} "
                  f"adv={np.mean(adv_scores):.3f} ({time.time()-t0:.0f}s)", flush=True)

    c = np.array(clean_scores); a = np.array(adv_scores)
    res = {"ckpt": args.ckpt, "domain": args.domain, "n": len(pos),
           "eps": args.eps, "alpha": args.alpha, "steps": args.steps,
           "clean": {"mean": round(float(c.mean()), 4), "median": round(float(np.median(c)), 4),
                     "min": round(float(c.min()), 4), "max": round(float(c.max()), 4)},
           "adv": {"mean": round(float(a.mean()), 4), "median": round(float(np.median(a)), 4),
                   "min": round(float(a.min()), 4), "max": round(float(a.max()), 4)},
           "delta_mean": round(float(a.mean() - c.mean()), 4),
           "per_sample": {"clean": [round(float(x), 6) for x in c],
                          "adv": [round(float(x), 6) for x in a]},
           "workpoints": {}}
    print("\n=== 结果 ===")
    print(f"clean P(YES): mean={c.mean():.4f} median={np.median(c):.4f}")
    print(f"adv   P(YES): mean={a.mean():.4f} median={np.median(a):.4f}")
    print(f"Δmean = {a.mean()-c.mean():+.4f}")
    for tau in (0.3, 0.5, 0.7):
        tpr_c = float((c >= tau).mean()); tpr_a = float((a >= tau).mean())
        asr = float((c >= tau).mean() - (a >= tau).mean())   # 被攻掉的检出率
        res["workpoints"][f"tau={tau}"] = {"tpr_clean": round(tpr_c, 4),
                                           "tpr_adv": round(tpr_a, 4),
                                           "tpr_drop": round(tpr_c - tpr_a, 4),
                                           "asr_vs_clean": round(asr, 4)}
        print(f"  τ={tau}: TPR {tpr_c:.3f} -> {tpr_a:.3f}  (下降 {tpr_c-tpr_a:+.3f})")
    json.dump(res, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"\n[adv] wrote {args.out}")


if __name__ == "__main__":
    main()
