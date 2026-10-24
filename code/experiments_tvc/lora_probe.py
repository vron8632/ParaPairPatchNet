#!/usr/bin/env python3
"""
E3: LoRA 表征探针 —— 微调到底改变了什么表征？
==============================================
思路：在**冻结的隐状态**上训练一个线性分类器（logistic regression），比较
      微调前(base) 与 微调后(LoRA) 的表征线性可分性。
      若微调显著提升线性可分性 → 微调改善的是"判别性表征"；
      若变化不大但端到端性能大增 → 微调主要改变的是**读出/决策**而非表征本身。

实现要点：
  - 取 Qwen2-VL 最后一层 hidden states，做 attention-mask 加权的 mean pooling。
  - 特征只算一次（无梯度），随后用 sklearn LR 交叉验证 AUC。

用法:
  $PY code/experiments_tvc/lora_probe.py --n-per-class 150 --out results/tvc/lora_probe.json
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
    ap.add_argument("--n-per-class", type=int, default=150, help="正/负各取多少张（val split）")
    ap.add_argument("--max-side", type=int, default=512)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(OUT, "lora_probe.json"))
    args = ap.parse_args()

    import numpy as np
    import torch
    from PIL import Image
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel
    from lora_mllm import QWEN_SNAP, PROMPT

    rows = json.load(open(args.index))
    pos = [r for r in rows if r["split"] == "val" and r["domain"] == "physical_patch"][:args.n_per_class]
    neg = [r for r in rows if r["split"] == "val" and r["domain"] == "clean"][:args.n_per_class]
    data = [(r["path"], 1) for r in pos] + [(r["path"], 0) for r in neg]
    print(f"[probe] n_pos={len(pos)} n_neg={len(neg)}")

    proc = AutoProcessor.from_pretrained(QWEN_SNAP, local_files_only=True)

    def extract(model):
        feats, labels = [], []
        model.eval()
        with torch.no_grad():
            for i, (p, y) in enumerate(data):
                im = Image.open(p).convert("RGB")
                im.thumbnail((args.max_side, args.max_side))
                msg = [{"role": "user",
                        "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
                t = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                enc = {k: (v.cuda() if torch.is_tensor(v) else v)
                       for k, v in proc(text=[t], images=[im], return_tensors="pt").items()}
                out = model(**enc, output_hidden_states=True)
                hs = out.hidden_states[-1][0]                       # [T, H]
                m = enc["attention_mask"][0].unsqueeze(-1).float()   # [T,1]
                pooled = (hs * m).sum(0) / m.sum().clamp(min=1)      # [H]
                feats.append(pooled.float().cpu().numpy())
                labels.append(y)
                if (i + 1) % 50 == 0:
                    print(f"    {i+1}/{len(data)}", flush=True)
        return np.asarray(feats), np.asarray(labels)

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    def probe_auc(X, y, tag):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, C=1.0))
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        acc = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
        auc = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
        # 全量拟合的线性可分性（用于报告）
        clf.fit(X, y)
        print(f"  [{tag}] 5-fold acc={acc.mean():.4f}±{acc.std():.4f}  "
              f"AUC={auc.mean():.4f}±{auc.std():.4f}  特征维度={X.shape[1]}")
        return {"acc_mean": round(float(acc.mean()), 4), "acc_std": round(float(acc.std()), 4),
                "auc_mean": round(float(auc.mean()), 4), "auc_std": round(float(auc.std()), 4),
                "dim": int(X.shape[1])}

    res = {"n_pos": len(pos), "n_neg": len(neg), "n_total": len(data),
           "max_side": args.max_side, "ckpt": args.ckpt}
    t0 = time.time()

    # ---- base（未微调）----
    print("[probe] 提取 base 表征 ...")
    base = Qwen2VLForConditionalGeneration.from_pretrained(
        QWEN_SNAP, torch_dtype=torch.float16, local_files_only=True).cuda()
    Xb, yb = extract(base)
    res["base"] = probe_auc(Xb, yb, "base (no LoRA)")
    del base; torch.cuda.empty_cache()

    # ---- LoRA 微调后 ----
    print("[probe] 提取 LoRA 微调后表征 ...")
    m2 = Qwen2VLForConditionalGeneration.from_pretrained(
        QWEN_SNAP, torch_dtype=torch.float16, local_files_only=True)
    m2 = PeftModel.from_pretrained(m2, args.ckpt).cuda()
    Xf, yf = extract(m2)
    res["finetuned"] = probe_auc(Xf, yf, "LoRA finetuned")
    res["elapsed_s"] = round(time.time() - t0, 1)

    # CCA/对齐度：两表征的差异（简单度量：中心化后的余弦相似）
    Xb_c = Xb - Xb.mean(0, keepdims=True); Xf_c = Xf - Xf.mean(0, keepdims=True)
    cos = float(np.mean([np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
                         for a, b in zip(Xb_c[:200], Xf_c[:200])]))
    res["repr_cosine_similarity"] = round(cos, 4)
    print(f"\n  表征余弦相似度(base vs 微调, 中心化后) = {cos:.4f}")

    json.dump(res, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"[probe] wrote {args.out}")


if __name__ == "__main__":
    main()
