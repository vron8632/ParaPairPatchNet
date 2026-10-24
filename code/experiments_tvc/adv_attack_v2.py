#!/usr/bin/env python3
"""
自适应攻击强化（扩展实验）
====================================
相对 v1 增强三点：
  1) 多种子：对 3 个 ckpt（seed0/1/2）分别攻击，报 mean±std；
  2) 攻击者模式：fixed-τ（最小化 p）vs adaptive-τ（已知部署阈值 τ_d，
     优化目标为跨越 τ_d 即止 —— 更接近红队设定）；
  3) 步数 10 / 20 两档。
输出: results/tvc/adv_attack_v2.json
"""
import os, sys, json, argparse
import numpy as np

WS = "/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard"
TVC = f"{WS}/results/tvc"
BASE = f"{TVC}/baseline_imgs"
sys.path.insert(0, f"{WS}/code/experiments_tvc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", default="lora_patch_detector,full_lora_s1,full_lora_s2")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--eps", type=float, default=0.03)
    ap.add_argument("--steps", default="10,20")
    ap.add_argument("--tau-d", type=float, default=0.5, help="部署阈值（adaptive 模式目标）")
    ap.add_argument("--out", default=f"{TVC}/adv_attack_v2.json")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from lora_mllm import load_lora_for_eval, PROMPT

    pos_dir = f"{BASE}/pos_all" if os.path.isdir(f"{BASE}/pos_all") else f"{BASE}/pos"
    files = [os.path.join(pos_dir, f) for f in sorted(os.listdir(pos_dir))[:args.n]]
    res = {}

    for ck in args.ckpts.split(","):
        ckpt = f"{TVC}/{ck}"
        if not os.path.isdir(ckpt): print(f"[skip] {ck}"); continue
        model, proc = load_lora_for_eval(ckpt, 16)
        cfg = json.load(open(f"{ckpt}/train_config.json")); yes_id, no_id = cfg["yes_id"], cfg["no_id"]
        VP = cfg.get("vision_start_id", None)

        def prepare(fp):
            im = Image.open(fp).convert("RGB")
            im = im if max(im.size) <= 512 else im.resize((512, int(512*im.height/im.width)))
            msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
            t = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            enc = proc(text=[t], images=[im], return_tensors="pt")
            return {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in enc.items()}

        def p_of(enc, pv=None):
            e = dict(enc)
            if pv is not None: e["pixel_values"] = pv
            with torch.no_grad():
                lg = model(**e).logits[0, -1]
            return float(torch.sigmoid(lg[yes_id] - lg[no_id])), lg

        for mode in ("fixed", "adaptive"):
            for steps in [int(s) for s in args.steps.split(",")]:
                clean, atk = [], []
                for fp in files:
                    enc = prepare(fp)
                    p0, _ = p_of(enc); clean.append(p0)
                    pv = enc["pixel_values"].clone().detach()
                    pv0 = pv.clone()
                    alpha = args.eps / max(steps // 2, 1)
                    for _ in range(steps):
                        pv.requires_grad_(True)
                        _, lg = p_of(enc, pv)
                        p = torch.sigmoid(lg[yes_id] - lg[no_id])
                        if mode == "fixed":
                            loss = p                                  # 最小化 P(YES)
                        else:
                            loss = torch.clamp(p - args.tau_d + 0.05, min=0)  # 跨越 τ_d 即止
                        g, = torch.autograd.grad(loss, pv, retain_graph=False)
                        with torch.no_grad():
                            pv = pv - alpha * g.sign()
                            pv = torch.clamp(pv, pv0 - args.eps, pv0 + args.eps)
                        pv = pv.detach()
                    p1, _ = p_of(enc, pv); atk.append(p1)
                c, a = np.array(clean), np.array(atk)
                thr = float(np.percentile(np.array([p_of(prepare(f"{BASE}/casia_au/{f}"))[0]
                                                    for f in sorted(os.listdir(f"{BASE}/casia_au"))[:60]]), 95))
                row = {"mode": mode, "steps": steps, "n": len(c),
                       "tpr_clean": round(float((c >= thr).mean()), 4),
                       "tpr_attacked": round(float((a >= thr).mean()), 4),
                       "asr": round(float((c >= thr).mean() - (a >= thr).mean()), 4),
                       "p_mean_clean": round(float(c.mean()), 4),
                       "p_mean_attacked": round(float(a.mean()), 4)}
                res.setdefault(ck, {})[f"{mode}_k{steps}"] = row
                print(f"  [{ck}] {mode} k={steps}: TPR {row['tpr_clean']:.3f} -> {row['tpr_attacked']:.3f} "
                      f"(ASR {row['asr']:.3f})", flush=True)
        del model
        torch.cuda.empty_cache()

    # 跨 seed 汇总
    if len(res) > 1:
        summ = {}
        for key in next(iter(res.values())):
            vals = [r[key]["asr"] for r in res.values() if key in r]
            summ[key] = {"asr_mean": round(float(np.mean(vals)), 4), "asr_std": round(float(np.std(vals)), 4)}
        res["summary"] = summ
        for k, v in summ.items(): print(f"  汇总 {k}: ASR {v['asr_mean']:.3f}±{v['asr_std']:.3f}")
    json.dump(res, open(args.out, "w"), indent=1, ensure_ascii=False)
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
