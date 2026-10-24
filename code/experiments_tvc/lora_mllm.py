#!/usr/bin/env python3
"""
MLLM 创新 #1 — LoRA 微调 Qwen2-VL-2B 做物理补丁检测
=====================================================
回应要求: "MLLM 方法不能直接用现成的，要加创新" + "参考 papers 的数据集与方法"

创新点:
  1. LoRA 微调 (peft) —— 把通用 VLM 变成补丁检测器
  2. 损失函数: CE + FPR 约束项 + 可选对齐损失(借鉴 AdvPatchXAI, CVPR2025 §3.2)
     L = CE + lambda_fp * max(0, mean(p|y=0) - beta)^2
     直接压制干净图的平均预测概率 => 降低 FPR（针对"工作点不可用"）
  3. 输出 logit 可连续化: 取 "YES" 与 "NO" 两个 token 的 logit 差作为分数，
     而不是生成文本后再解析（避免 parse 失败，且可微）

用法:
  PY=/home/oyp/miniconda3/envs/apjf/bin/python
  $PY code/experiments_tvc/lora_mllm.py --stage prep   # 准备训练数据
  $PY code/experiments_tvc/lora_mllm.py --stage train  # LoRA 微调
  $PY code/experiments_tvc/lora_mllm.py --stage eval   # 评估工作点
"""
import argparse, json, os, sys, time

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS = os.path.dirname(CODE_DIR)
QWEN_SNAP = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen2-VL-2B-Instruct/snapshots"
    "/895c3a49bc3fa70a340399125c650a463535e71c")
SRC = "/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard"
PD = os.path.join(SRC, "results", "features", "patch_dataset_v2")
OUT = os.path.join(WS, "results", "tvc")

PROMPT = ("Is there a suspicious printed sticker or adversarial patch attached to an object "
          "in this photo? Answer YES or NO.")


# ───────────────────────── 数据准备 ─────────────────────────
def stage_prep(args):
    import torch
    from PIL import Image
    import numpy as np
    os.makedirs(os.path.join(OUT, "lora_data"), exist_ok=True)
    rows = []
    for dom, lab in [("clean", 0), ("physical_patch", 1), ("digital_patch", 1)]:
        for sp in ("train", "val"):
            p = os.path.join(PD, f"{dom}_{sp}.pt")
            if not os.path.exists(p):
                print(f"skip {p}"); continue
            d = torch.load(p, map_location="cpu", weights_only=False)
            n = min(args.per_domain, len(d["tensors"]))
            # 均匀抽样，避免全取前 n 张
            idx = np.linspace(0, len(d["tensors"]) - 1, n).astype(int)
            for i in idx:
                arr = np.clip(d["tensors"][i].permute(1, 2, 0).numpy() * 255, 0, 255).astype(np.uint8)
                fp = os.path.join(OUT, "lora_data", f"{dom}_{sp}_{i}.jpg")
                if not os.path.exists(fp):
                    Image.fromarray(arr).save(fp, quality=92)
                rows.append({"path": fp, "label": lab, "domain": dom, "split": sp})
    split = os.path.join(OUT, "lora_data", "index.json")
    json.dump(rows, open(split, "w"), indent=2)
    print(f"[prep] {len(rows)} 条 -> {split}")
    from collections import Counter
    print("        ", Counter((r['domain'], r['split']) for r in rows))


# ───────────────────────── 模型与损失 ─────────────────────────
def base_and_proc():
    """返回**未加 LoRA** 的 base 模型与 processor。"""
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        QWEN_SNAP, torch_dtype=torch.float16, local_files_only=True)
    proc = AutoProcessor.from_pretrained(QWEN_SNAP, local_files_only=True)
    return model, proc


def load_lora_for_eval(ckpt, rank=16):
    """**正确的 LoRA 加载方式**：从纯 base 直接 PeftModel.from_pretrained。

    重要: 不能在已 get_peft_model 的模型上再 PeftModel.from_pretrained ——
    会产生 base_model.base_model 前缀错位，adapter 权重静默丢失(missing keys)，
    lora_B 保持零初始化 => LoRA 输出恒为 0 => 实际等同于未微调的 base 模型。
    此前多处评估均因此失效（见 REPORT.md 记录）。
    """
    from peft import PeftModel
    model, proc = base_and_proc()
    model = PeftModel.from_pretrained(model, ckpt)
    model = model.cuda().eval()
    return model, proc


def build_model(lora_rank, freeze_vision=True):
    from peft import LoraConfig, get_peft_model
    model, proc = base_and_proc()
    if freeze_vision:
        for n, p in model.named_parameters():
            if "visual" in n:
                p.requires_grad = False
    cfg = LoraConfig(
        r=lora_rank, lora_alpha=lora_rank * 2, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model.cuda(), proc


def nll_fpr_loss(logits, labels, yes_id, no_id, lam=1.0, beta=0.15, mode="both", tau0=0.5):
    """CE + FPR 约束: 压制干净图(y=0)在 YES 上的平均概率降到 beta 以下。
    这是对 Neyman-Pearson 约束的可微松弛 —— 直接针对"工作点不可用"。

    mode（消融用，对应理论章节 Φ 的两项）:
      "both"      -> (i) 均值约束 + (ii) 工作点惩罚   [默认，历史行为]
      "mean"      -> 仅 (i)  ReLU(mean(p_neg)-beta)^2
      "workpoint" -> 仅 (ii) mean(1[p_neg>0.5])
      "none"      -> 关闭约束（等价于纯 CE / SFT）
    """
    import torch
    import torch.nn.functional as F
    yz = logits[:, yes_id]
    nz = logits[:, no_id]
    sc = torch.stack([nz, yz], 1)                    # [NO, YES]，index = label
    ce = F.cross_entropy(sc, labels)
    neg_mask = labels == 0
    zero = torch.zeros((), device=logits.device)
    mean_term, wp_term = zero, zero
    if neg_mask.any():
        p_yes_neg = torch.sigmoid(yz[neg_mask] - nz[neg_mask])
        mean_term = F.relu(p_yes_neg.mean() - beta) ** 2      # (i) 均值约束（可微）
        # (ii) 工作点惩罚：硬阶跃 1[p>0.5] 不可微（梯度恒 0，训练中不起作用），
        #      改为温度 T=0.05 的 sigmoid 软化 —— 可微的"越过工作点"软计数
        wp_soft = torch.sigmoid((p_yes_neg - tau0) / 0.05)
        wp_term = wp_soft.mean()
        hard_wp = (p_yes_neg > tau0).float().mean()            # 仅作监控指标（不参与梯度）
        if mode == "mean":
            fpr_term = mean_term
        elif mode == "workpoint":
            fpr_term = wp_term
        elif mode == "none":
            fpr_term = zero
        else:
            fpr_term = mean_term + wp_term
    else:
        fpr_term = zero
        hard_wp = zero
    return ce + lam * fpr_term, {"ce": round(float(ce), 4), "fpr": round(float(fpr_term), 4),
                                 "mean": round(float(mean_term), 4),
                                 "wp_soft": round(float(wp_term), 4), "wp_hard": round(float(hard_wp), 4), "mode": mode}


# ───────────────────────── 训练 ─────────────────────────
def stage_train(args):
    import torch, random
    import numpy as _np
    if getattr(args, "seed", 0) is not None:
        random.seed(args.seed); _np.random.seed(args.seed)
        torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
    from torch.utils.data import Dataset, DataLoader
    rows = json.load(open(args.index))
    tr = [r for r in rows if r["split"] == "train"]

    # ── 范式一致性：构造 digital/physical 同索引配对（方法原型）──
    import re as _re2
    _num = lambda p: int(_re2.search(r"_(\d+)\.jpg$", p).group(1)) if _re2.search(r"_(\d+)\.jpg$", p) else -1
    _pairs_by_idx = {}
    for _r in tr:
        _d = _r.get("domain", "")
        if _d not in ("digital_patch", "physical_patch"):
            continue
        _k = _num(_r["path"])
        _pairs_by_idx.setdefault(_k, {})[_d] = _r["path"]
    CONS_PAIRS = [(v["digital_patch"], v["physical_patch"])
                  for v in _pairs_by_idx.values()
                  if "digital_patch" in v and "physical_patch" in v]
    print(f"[cons] 可用配对 {len(CONS_PAIRS)} 对（digital ↔ physical 同索引）", flush=True)
    frac = float(getattr(args, "train_frac", 1.0) or 1.0)
    if frac < 1.0:
        import random as _rnd
        _rnd.seed(int(getattr(args, "seed", 0)))
        pos_ = [r for r in tr if r["label"] == 1]
        neg_ = [r for r in tr if r["label"] == 0]
        tr = (_rnd.sample(pos_, max(1, int(len(pos_) * frac)))
              + _rnd.sample(neg_, max(1, int(len(neg_) * frac))))
        print(f"[train-frac={frac}] n_pos={len(pos_)}->{int(len(pos_)*frac)} "
              f"n_neg={len(neg_)}->{int(len(neg_)*frac)}", flush=True)
    from PIL import Image

    model, proc = build_model(args.rank)
    scope = getattr(args, "train_scope", "lora")
    import re as _re
    def _max_layer(ns):
        ls = [int(m.group(1)) for n_ in ns for m in [_re.search(r"layers\.(\d+)\.", n_)] if m]
        return max(ls) if ls else -1
    names = [n_ for n_, _ in model.named_parameters()]
    if scope == "embed_only":
        # Qwen2-VL tie_word_embeddings=True ⇒ 输出矩阵与 embed_tokens 共享权重，
        # "只训读出矩阵"实际等价于只训词嵌入层（也即输出侧）
        for n_, p_ in model.named_parameters():
            p_.requires_grad = n_.endswith("embed_tokens.weight")
    elif scope == "deep_lora":
        # 只训最后 6 层的 LoRA（近输出端）
        top = _max_layer([n_ for n_ in names if "lora_" in n_])
        for n_, p_ in model.named_parameters():
            if "lora_" in n_:
                m_ = _re.search(r"layers\.(\d+)\.", n_)
                p_.requires_grad = bool(m_) and int(m_.group(1)) >= top - 5
            else:
                p_.requires_grad = False
    elif scope == "embed_deep":
        # 词嵌入（=读出）+ 最后 6 层 LoRA
        top = _max_layer([n_ for n_ in names if "lora_" in n_])
        for n_, p_ in model.named_parameters():
            if n_.endswith("embed_tokens.weight"):
                p_.requires_grad = True
            elif "lora_" in n_:
                m_ = _re.search(r"layers\.(\d+)\.", n_)
                p_.requires_grad = bool(m_) and int(m_.group(1)) >= top - 5
            else:
                p_.requires_grad = False
    n_train = sum(1 for p_ in model.parameters() if p_.requires_grad)
    n_all = sum(1 for _ in model.parameters())
    print(f"[train-scope={scope}] trainable tensors={n_train}", flush=True)
    yes_id = proc.tokenizer.convert_tokens_to_ids("YES")
    no_id = proc.tokenizer.convert_tokens_to_ids("NO")
    # 兜底: 某些 tokenizer 对大写词可能是多 token
    if yes_id is None or no_id is None or yes_id < 0 or no_id < 0:
        yes_id = proc.tokenizer.encode(" YES", add_special_tokens=False)[-1]
        no_id = proc.tokenizer.encode(" NO", add_special_tokens=False)[-1]
    print(f"[train] yes_id={yes_id} no_id={no_id}")

    class DS(Dataset):
        def __init__(self, rows):
            self.rows = rows
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, i):
            return self.rows[i]

    def collate(batch):
        imgs, txts = [], []
        for r in batch:
            img = Image.open(r["path"]).convert("RGB")
            msgs = [{"role": "user",
                     "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
            t = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            imgs.append(img); txts.append(t)
        enc = proc(text=txts, images=imgs, return_tensors="pt", padding=True)
        return enc, torch.tensor([r["label"] for r in batch])

    dl = DataLoader(DS(tr), batch_size=args.bs, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    model.train()
    step = 0
    for ep in range(args.epochs):
        ep_loss, ep_meta = 0.0, {}
        for enc, labels in dl:
            enc = {k: v.cuda() for k, v in enc.items()}
            labels = labels.cuda()
            # ── 在线对抗训练（可选）──
            # 动机：adv_attack_pgd.json 显示干净训练下白盒 PGD 可把 TPR 从 0.96 打到 0.54（ASR 42%）。
            # 做法：在 pixel_values 空间对"最小化 P(YES)"做 FGSM/PGD，再在该扰动样本上正常训练。
            if args.adv_train > 0:
                pv0 = enc["pixel_values"].detach()
                pv = pv0.clone()
                inner = max(1, args.adv_steps)
                for _k in range(inner):
                    pv = pv.detach().clone().requires_grad_(True)
                    e2 = dict(enc); e2["pixel_values"] = pv
                    o2 = model(**e2)
                    l2 = e2["attention_mask"].sum(1).long() - 1
                    lg2 = o2.logits[torch.arange(o2.logits.shape[0], device=o2.logits.device), l2]
                    s2 = torch.sigmoid(lg2[:, yes_id] - lg2[:, no_id])
                    g = torch.autograd.grad(torch.log(s2 + 1e-8).sum(), pv)[0]
                    with torch.no_grad():
                        pv = pv - (args.adv_train / inner) * g.sign()
                        pv = torch.clamp(pv, pv0 - args.adv_train, pv0 + args.adv_train)
                enc["pixel_values"] = pv.detach()
                del o2, lg2, s2
            out = model(**enc)
            # 取每条样本最后一个非 padding 位置的 logits（首个生成 token）
            am = enc["attention_mask"]
            last = am.sum(1).long() - 1
            logits = out.logits[torch.arange(out.logits.shape[0], device=out.logits.device), last]
            # 约束课程（constraint warmup）：前 warmup_frac 比例的步数内 lam 线性从 0 升到目标值。
            # 动机：让模型先学判别表示（CE 主导），再逐步引入 FPR 约束，
            #       避免约束在表示未成型时就主导梯度方向。
            if args.constraint_warmup > 0:
                total_steps = max(len(dl) * args.epochs, 1)
                frac = min(1.0, step / max(total_steps * args.constraint_warmup, 1))
                eff_lam = args.lam * frac
            else:
                eff_lam = args.lam
            loss, meta = nll_fpr_loss(logits, labels, yes_id, no_id, lam=eff_lam,
                                      beta=args.beta, mode=args.constraint_mode,
                                      tau0=getattr(args, "tau0", 0.5))
            # ── 方法原型：范式一致性损失（stop-gradient 单边，省显存）──
            _cw = float(getattr(args, "cons_weight", 0.0) or 0.0)
            if _cw > 0 and CONS_PAIRS:
                import random as _rd
                _dp, _pp_path = _rd.choice(CONS_PAIRS)
                if _rd.random() < 0.5:          # 随机方向，保证对称
                    _grad_path, _tgt_path = _dp, _pp_path
                else:
                    _grad_path, _tgt_path = _pp_path, _dp

                def _score_small(_p, _need_grad):
                    _im = Image.open(_p).convert("RGB")
                    _im.thumbnail((224, 224))
                    _msg = [{"role": "user",
                             "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
                    _t = proc.apply_chat_template(_msg, tokenize=False, add_generation_prompt=True)
                    _enc = {k2: (v2.cuda() if torch.is_tensor(v2) else v2)
                            for k2, v2 in proc(text=[_t], images=[_im], return_tensors="pt").items()}
                    if _need_grad:
                        _lg = model(**_enc).logits[0, -1]
                        return torch.sigmoid(_lg[yes_id] - _lg[no_id])
                    with torch.no_grad():
                        _lg = model(**_enc).logits[0, -1]
                        return torch.sigmoid(_lg[yes_id] - _lg[no_id])

                if getattr(args, "cons_target", "score") == "hidden":
                    # B 方案：表示层（最后 hidden state）一致性 —— 机制指向读出路径
                    def _hid_small(_p, _need_grad):
                        _im = Image.open(_p).convert("RGB"); _im.thumbnail((224, 224))
                        _msg = [{"role": "user",
                                 "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
                        _t = proc.apply_chat_template(_msg, tokenize=False, add_generation_prompt=True)
                        _enc = {k2: (v2.cuda() if torch.is_tensor(v2) else v2)
                                for k2, v2 in proc(text=[_t], images=[_im], return_tensors="pt").items()}
                        if _need_grad:
                            _o = model(**_enc, output_hidden_states=True)
                            return _o.hidden_states[-1][0, -1].float()
                        with torch.no_grad():
                            _o = model(**_enc, output_hidden_states=True)
                            return _o.hidden_states[-1][0, -1].float()
                    _h_grad = _hid_small(_grad_path, True)
                    _h_tgt = _hid_small(_tgt_path, False)
                    _cons = 1.0 - torch.nn.functional.cosine_similarity(
                        _h_grad.unsqueeze(0), _h_tgt.unsqueeze(0), dim=-1).squeeze()
                else:
                    _p_grad = _score_small(_grad_path, True)          # 只有这一边反传
                    _p_tgt = _score_small(_tgt_path, False)           # 目标值（detach）
                    _cons = (_p_grad - _p_tgt) ** 2
                loss = loss + _cw * _cons
                meta["cons"] = float(_cons)
                del _cons
                torch.cuda.empty_cache()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += float(loss); ep_meta = meta
            step += 1
            if step % 20 == 0:
                print(f"  ep{ep} step{step} loss={float(loss):.4f} lam={eff_lam:.3f} {meta}", flush=True)
        print(f"[train] epoch {ep}: loss={ep_loss/max(len(dl),1):.4f} {ep_meta}", flush=True)

    ad = os.path.join(OUT, args.tag)
    model.save_pretrained(ad)
    print(f"[train] LoRA saved -> {ad}")
    json.dump({"rank": args.rank, "lr": args.lr, "lam": args.lam, "beta": args.beta,
               "tau0": float(getattr(args, "tau0", 0.5)),
               "epochs": args.epochs, "bs": args.bs, "yes_id": yes_id, "no_id": no_id,
               "n_train": len(tr), "constraint_mode": args.constraint_mode,
               "train_scope": getattr(args, "train_scope", "lora"),
               "train_frac": float(getattr(args, "train_frac", 1.0) or 1.0),
               "cons_weight": float(getattr(args, "cons_weight", 0.0) or 0.0),
               "cons_mode": getattr(args, "cons_mode", "score"),
               "cons_target": getattr(args, "cons_target", "score"),
               "seed": int(getattr(args, "seed", 0)),
               "constraint_warmup": args.constraint_warmup},
              open(os.path.join(ad, "train_config.json"), "w"), indent=2)


# ───────────────────────── 评估 ─────────────────────────
def stage_eval(args):
    import torch, numpy as np
    from PIL import Image
    ad = os.path.join(OUT, args.tag)
    # 修复: 必须从纯 base 加载 LoRA（此前在已 peft 模型上再 from_pretrained 导致权重静默丢失）
    model, proc = load_lora_for_eval(ad, args.rank)
    cfg = json.load(open(os.path.join(ad, "train_config.json")))
    yes_id, no_id = cfg["yes_id"], cfg["no_id"]

    rows = json.load(open(args.index))
    te = [r for r in rows if r["split"] == "val"]
    scores, labels = [], []
    with torch.no_grad():
        for i, r in enumerate(te):
            img = Image.open(r["path"]).convert("RGB")
            msgs = [{"role": "user",
                     "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
            t = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            enc = {k: (v.cuda() if torch.is_tensor(v) else v)
                   for k, v in proc(text=[t], images=[img], return_tensors="pt").items()}
            lg = model(**enc).logits[0, -1]
            s = float(torch.sigmoid(lg[yes_id] - lg[no_id]))
            scores.append(s); labels.append(r["label"])
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(te)}", flush=True)

    p = np.array(scores); y = np.array(labels)
    pos, neg = y == 1, y == 0
    order = np.argsort(np.concatenate([p[pos], p[neg]]))
    r = np.empty(len(order)); r[order] = np.arange(1, len(order) + 1)
    auc = (r[:pos.sum()].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum())
    # 阈值网格 = 固定网格 ∪ 负类分位数。
    # 注意: 只用负类分位数在**分数饱和**时会退化（COCO/INRIA 上 P(YES) 恰为 0/1，
    # 负类分位数全为 0 => tau=0 => 所有样本判正 => FPR=1、@FPR 档位全 None）。
    grid = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    th = np.unique(np.concatenate([grid, np.percentile(p[neg], np.linspace(0, 100, 201))]))
    sweep = [{"tau": round(float(t), 4),
              "tpr": round(float(((p >= t) & pos).sum() / max(pos.sum(), 1)), 4),
              "fpr": round(float(((p >= t) & neg).sum() / max(neg.sum(), 1)), 4)} for t in th]
    rep = {"method": "LoRA-Qwen2-VL-2B + FPR-constrained loss",
           "n_val": int(len(y)), "n_pos": int(pos.sum()), "n_neg": int(neg.sum()),
           "auc": round(float(auc), 4), "sweep": sweep,
           # per-sample 连续分数，供 adapter_fusion.py 使用
           "scores": [round(float(s), 6) for s in p],
           "labels": [int(v) for v in y],
           "paths": [r["path"] for r in te]}
    for b in (0.01, 0.05, 0.10, 0.20):
        cand = [s for s in sweep if s["fpr"] <= b]
        rep[f"best_tpr_at_fpr<={int(b*100)}pct"] = max(cand, key=lambda s: s["tpr"]) if cand else None
    os.makedirs(OUT, exist_ok=True)
    json.dump(rep, open(os.path.join(OUT, f"lora_mllm_eval_{args.tag}.json"), "w"), indent=2, ensure_ascii=False)
    print(f"[eval] AUC={rep['auc']} n_pos={rep['n_pos']} n_neg={rep['n_neg']}")
    for k in ("best_tpr_at_fpr<=1pct", "best_tpr_at_fpr<=5pct",
              "best_tpr_at_fpr<=10pct", "best_tpr_at_fpr<=20pct"):
        print(f"   {k}: {rep[k]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["prep", "train", "eval", "all"], default="all")
    ap.add_argument("--per-domain", type=int, default=600)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lam", type=float, default=1.0, help="FPR 约束权重")
    ap.add_argument("--beta", type=float, default=0.15, help="FPR 预算(目标平均 P(YES|clean))")
    ap.add_argument("--tau0", type=float, default=0.5, help="工作点惩罚的参考阈值 τ0（超参敏感实验）")
    ap.add_argument("--cons-weight", type=float, default=0.0,
                    help="范式一致性损失权重（方法原型：同一补丁的 digital/physical 配对分数应一致）")
    ap.add_argument("--cons-target", choices=["score", "hidden"], default="score",
                    help="一致性作用点：score=输出分数（A 方案）/ hidden=最后层表示（B 方案，机制指向读出路径）")
    ap.add_argument("--cons-mode", choices=["score", "score+rank"], default="score",
                    help="一致性形式：score=分数贴近 / score+rank=分数+排序一致")
    ap.add_argument("--tag", default="lora_patch_detector", help="输出目录名（消融时用不同 tag 避免覆盖）")
    ap.add_argument("--constraint-warmup", type=float, default=0.0,
                    help="约束课程：前该比例步数内 lam 线性从 0 升至目标（0=关闭）")
    ap.add_argument("--constraint-mode", choices=["both", "mean", "workpoint", "none"],
                    default="both",
                    help="FPR 约束项拆解（消融）：both=(i)+(ii) 均值+工作点，mean=仅 (i)，"
                         "workpoint=仅 (ii)，none=关闭（等价纯 CE）")
    ap.add_argument("--adv-train", type=float, default=0.0,
                    help=">0 时启用在线对抗训练（FGSM on pixel_values），值为 ε（归一化空间）")
    ap.add_argument("--adv-steps", type=int, default=1, help="对抗训练内层步数（1=FGSM）")
    ap.add_argument("--seed", type=int, default=0, help="训练随机种子（多种子统计用）")
    ap.add_argument("--train-frac", type=float, default=1.0,
                    help="训练集使用比例（数据规模消融：0.25/0.5/1.0）")
    ap.add_argument("--train-scope", choices=["lora", "embed_only", "embed_deep", "deep_lora"],
                    default="lora",
                    help="可训练范围: lora=全量LoRA(18.4M) / embed_only=只训词嵌入(=共享读出矩阵) / embed_deep=词嵌入+最后6层LoRA / deep_lora=只训最后6层LoRA")
    ap.add_argument("--index", default=None, help="训练/评估用 index json（默认 lora_data/index.json）")
    args = ap.parse_args()
    if args.index is None:
        args.index = os.path.join(OUT, "lora_data", "index.json")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if args.stage in ("prep", "all"):
        stage_prep(args)
    if args.stage in ("train", "all"):
        stage_train(args)
    if args.stage in ("eval", "all"):
        stage_eval(args)


if __name__ == "__main__":
    main()
