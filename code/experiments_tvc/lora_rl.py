#!/usr/bin/env python3
"""
MLLM 创新 #3 — 轻量强化学习：Reward-Weighted LoRA (RFT/GRPO-lite)
===================================================================
回应要求: "用轻量化的强化学习方法"

方法: Rejection-Sampling Fine-Tuning + Reward-Weighted 损失（RLHF 家族的轻量形式，
不需要完整 PPO/GRPO 训练框架，在 4060Ti 上可跑）。

Reward 设计（针对最关键的 FPR）:
    r = +R_tp            若 y=1 且 p>0.5   （抓到补丁）
    r = +R_tn            若 y=0 且 p<=0.5  （干净图不误报）
    r = -C_fp * (1-y)    若 y=0 且 p>0.5   （误报惩罚，C_fp 可调大 => 强压 FPR）
    r = -C_fn * y        若 y=1 且 p<=0.5  （漏检惩罚）
加权 CE:  L = sum_i w_i * CE_i ,  w_i = softmax(r_i / temperature)
   -> reward 高的样本权重更大；误报样本在 C_fp 大时权重被抑制。

同时记录策略迭代过程中的 TPR/FPR，形成"RL 训练曲线"（论文可作一张图）。

用法:
  $PY code/experiments_tvc/lora_rl.py --stage train --iters 3 --c-fp 3.0
  $PY code/experiments_tvc/lora_rl.py --stage eval
输出: results/tvc/lora_rl_log.json / lora_rl_eval.json
"""
import argparse, json, os, re, sys, time

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)
OUT = os.path.join(os.path.dirname(CODE_DIR), "results", "tvc")
SRC = "/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard"
PD = os.path.join(SRC, "results", "features", "patch_dataset_v2")

from lora_mllm import (QWEN_SNAP, PROMPT, build_model,  # noqa: E402
                       stage_prep as _prep, OUT as LORA_OUT)

CKPT = os.path.join(OUT, "lora_rl_ckpt")


def build_index(args):
    """复用 lora 的数据索引（clean/physical/digital 的 train/val）。"""
    idx_path = os.path.join(OUT, "lora_data", "index.json")
    if not os.path.exists(idx_path):
        _prep(args)
    return json.load(open(idx_path))


def get_batch(proc, rows):
    from PIL import Image
    import torch
    imgs, txts = [], []
    for r in rows:
        imgs.append(Image.open(r["path"]).convert("RGB"))
        msgs = [{"role": "user",
                 "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        txts.append(proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    enc = proc(text=txts, images=imgs, return_tensors="pt", padding=True)
    enc = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in enc.items()}
    return enc


def score_batch(model, proc, rows, yes_id, no_id):
    import torch
    enc = get_batch(proc, rows)
    with torch.no_grad():
        out = model(**enc)
        am = enc["attention_mask"]
        last = am.sum(1).long() - 1
        lg = out.logits[torch.arange(out.logits.shape[0], device=out.logits.device), last]
        s = torch.sigmoid(lg[:, yes_id] - lg[:, no_id])
    return s


def reward_of(scores, labels, c_fp, c_fn, r_tp, r_tn):
    import numpy as np
    y = np.asarray(labels, dtype=int)
    pred = (np.asarray(scores) > 0.5).astype(int)
    r = np.zeros(len(y), dtype=np.float32)
    r += np.where((y == 1) & (pred == 1), r_tp, 0.0)
    r += np.where((y == 0) & (pred == 0), r_tn, 0.0)
    r -= np.where((y == 0) & (pred == 1), c_fp, 0.0)
    r -= np.where((y == 1) & (pred == 0), c_fn, 0.0)
    return r, pred


def wce(logits_pair, labels, w):
    import torch
    import torch.nn.functional as F
    per = F.cross_entropy(logits_pair, labels, reduction="none")
    return (per * w).sum() / (w.sum() + 1e-8)


def stage_train(args):
    import numpy as np
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor  # noqa: F401
    from peft import PeftModel

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    rows = build_index(args)
    tr = [r for r in rows if r["split"] == "train"]

    # 加载起点 LoRA。**必须用 load_lora_for_eval（从纯 base 加载）**：
    # 在已 get_peft_model 的模型上再 PeftModel.from_pretrained 会产生
    # base_model.base_model 双层前缀，权重静默丢失、且保存出的 ckpt 结构也是坏的。
    from lora_mllm import load_lora_for_eval
    yes_id, no_id = 14004, 8996
    start = None
    if os.path.isdir(CKPT):
        start = CKPT
    elif os.path.isdir(os.path.join(OUT, "lora_patch_detector")):
        start = os.path.join(OUT, "lora_patch_detector")
    if start:
        model, proc = load_lora_for_eval(start, args.rank)
        print(f"[rl] 从 {start} 初始化 (SFT -> RL)")
    else:
        model, proc = build_model(args.rank)
        print("[rl] 从随机 LoRA 开始")
    # 显式启用 LoRA 参数（from_pretrained 后可能被重置）
    n_train = 0
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad = True
            n_train += 1
    if n_train == 0:
        raise RuntimeError("没有可训练的 LoRA 参数（检查 target_modules 是否命中）")
    print(f"[rl] 启用 LoRA 参数 {n_train} 个")
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    log = []
    bs = args.bs
    for it in range(args.iters):
        t0 = time.time()
        # 1) 采样当前策略分数 + reward
        all_idx = np.random.permutation(len(tr))
        scores, labels = [], []
        for s in range(0, len(tr), bs):
            b = [tr[i] for i in all_idx[s:s + bs]]
            sc = score_batch(model, proc, b, yes_id, no_id)
            scores.append(sc.cpu().numpy()); labels.extend([r["label"] for r in b])
        scores = np.concatenate(scores)
        labels_np = np.asarray(labels, dtype=int)
        r, pred = reward_of(scores, labels_np, args.c_fp, args.c_fn, args.r_tp, args.r_tn)
        torch.cuda.empty_cache()   # 采样阶段的中间激活已释放，避免训练阶段 OOM
        y = np.asarray(labels, dtype=int)
        acc = float((pred == y).mean())
        fpr = float(pred[y == 0].mean()) if (y == 0).any() else 0.0
        tpr = float(pred[y == 1].mean()) if (y == 1).any() else 0.0

        # 2) reward-weighted CE 更新
        # w 的下标与 all_idx 顺序一致；batch 顺序即 all_idx[s:s+len(b)]，故直接取 w[s:s+len(b)]
        import torch.nn.functional as F
        w = torch.tensor(np.exp((r - r.max()) / max(args.temperature, 1e-3)),
                         dtype=torch.float32).cuda()
        tot = 0.0
        for s in range(0, len(tr), bs):
            ids = all_idx[s:s + bs]
            b = [tr[i] for i in ids]
            enc = get_batch(proc, b)
            out = model(**enc)
            am = enc["attention_mask"]
            last = am.sum(1).long() - 1
            lg = out.logits[torch.arange(out.logits.shape[0], device=out.logits.device), last]
            pair = torch.stack([lg[:, no_id], lg[:, yes_id]], 1)
            lab = torch.tensor([tr[i]["label"] for i in ids]).cuda()
            wb = w[s:s + len(b)]
            per = F.cross_entropy(pair, lab, reduction="none")
            loss = (per * wb).sum() / (wb.sum() + 1e-8)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss)

        m = {"iter": it, "acc": round(acc, 4), "tpr": round(tpr, 4), "fpr": round(fpr, 4),
             "reward_mean": round(float(r.mean()), 4), "loss": round(tot / max(len(tr) // bs, 1), 4),
             "sec": round(time.time() - t0, 1)}
        log.append(m)
        print(f"[rl] iter{it}: acc={acc:.3f} TPR={tpr:.3f} FPR={fpr:.3f} "
              f"reward={r.mean():.3f} loss={m['loss']:.4f} ({m['sec']}s)", flush=True)

    os.makedirs(CKPT, exist_ok=True)
    model.save_pretrained(CKPT)
    json.dump({"log": log, "c_fp": args.c_fp, "c_fn": args.c_fn, "temperature": args.temperature,
               "iters": args.iters, "lr": args.lr, "rank": args.rank},
              open(os.path.join(OUT, "lora_rl_log.json"), "w"), indent=2)
    print(f"[rl] saved {CKPT}; log -> {OUT}/lora_rl_log.json")


def stage_eval(args):
    """评估 RL ckpt 在 lora_data val split 上的 operating point。"""
    import numpy as np
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from lora_mllm import stage_eval as _lora_eval  # 复用评估逻辑

    rows = build_index(args)
    te = [r for r in rows if r["split"] == "val"]
    from lora_mllm import load_lora_for_eval
    model, proc = load_lora_for_eval(CKPT, args.rank)
    yes_id, no_id = 14004, 8996
    scores, labels = [], []
    for k, r in enumerate(te):
        s = float(score_batch(model, proc, [r], yes_id, no_id)[0])
        scores.append(s); labels.append(r["label"])
        if (k + 1) % 50 == 0:
            print(f"  {k+1}/{len(te)}", flush=True)
    p = np.array(scores); y = np.array(labels)
    pos, neg = y == 1, y == 0
    th = np.unique(np.percentile(p[neg], np.linspace(0, 100, 201)))
    sweep = [{"tau": round(float(t), 4),
              "tpr": round(float(((p >= t) & pos).sum() / pos.sum()), 4),
              "fpr": round(float(((p >= t) & neg).sum() / neg.sum()), 4)} for t in th]
    rep = {"method": "reward-weighted LoRA (RL)", "n_val": int(len(y)),
           "n_pos": int(pos.sum()), "n_neg": int(neg.sum()), "sweep": sweep,
           "scores": [round(float(s), 6) for s in p], "labels": [int(v) for v in y]}
    for b in (0.01, 0.05, 0.10, 0.20):
        cand = [s for s in sweep if s["fpr"] <= b]
        rep[f"best_tpr_at_fpr<={int(b*100)}pct"] = max(cand, key=lambda s: s["tpr"]) if cand else None
    json.dump(rep, open(os.path.join(OUT, "lora_rl_eval.json"), "w"), indent=2, ensure_ascii=False)
    print(f"[rl-eval] n={rep['n_val']} pos={rep['n_pos']} neg={rep['n_neg']}")
    for k in ("best_tpr_at_fpr<=1pct", "best_tpr_at_fpr<=5pct",
              "best_tpr_at_fpr<=10pct", "best_tpr_at_fpr<=20pct"):
        print(f"    {k}: {rep[k]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["train", "eval"], default="train")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--c-fp", type=float, default=3.0, help="误报惩罚（越大 FPR 越低）")
    ap.add_argument("--c-fn", type=float, default=1.0, help="漏检惩罚")
    ap.add_argument("--r-tp", type=float, default=1.0)
    ap.add_argument("--r-tn", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=1.0, help="reward softmax 温度")
    ap.add_argument("--per-domain", type=int, default=600)
    args = ap.parse_args()
    if args.stage == "train":
        stage_train(args)
    else:
        stage_eval(args)


if __name__ == "__main__":
    main()
