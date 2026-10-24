#!/usr/bin/env python3
"""
基线评估 — 未微调的 Qwen2-VL-2B（base，无 LoRA）
====================================================
用途：回答"LoRA 微调比纯 MLLM 高多少"——需要 base 模型在同一 val 集上的数字。
与 lora_mllm.stage_eval 完全同协议（同 val、同打分方式），唯一区别是不加载 LoRA。

用法:
  $PY code/experiments_tvc/eval_base_qwen.py
输出: results/tvc/eval_base_qwen.json
"""
import argparse, json, os, sys
import numpy as np

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS = os.path.dirname(CODE_DIR)
OUT = os.path.join(WS, "results", "tvc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=os.path.join(OUT, "lora_data", "index.json"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    from PIL import Image
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from lora_mllm import QWEN_SNAP, PROMPT, base_and_proc

    model, proc = base_and_proc()
    model = model.cuda().eval()
    yes_id, no_id = 14004, 8996

    rows = json.load(open(args.index))
    val = [r for r in rows if r["split"] == "val"]
    print(f"[base-qwen] val n={len(val)}")

    scores, labels = [], []
    for i, r in enumerate(val):
        img = Image.open(r["path"]).convert("RGB")
        msg = [{"role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        t = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        enc = {k: (v.cuda() if torch.is_tensor(v) else v)
               for k, v in proc(text=[t], images=[img], return_tensors="pt").items()}
        with torch.no_grad():
            lg = model(**enc).logits[0, -1]
        scores.append(float(torch.sigmoid(lg[yes_id] - lg[no_id])))
        labels.append(r["label"])
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(val)}", flush=True)

    p = np.array(scores); y = np.array(labels)
    pos, neg = y == 1, y == 0
    order = np.argsort(np.concatenate([p[pos], p[neg]]))
    rk = np.empty(len(order)); rk[order] = np.arange(1, len(order) + 1)
    auc = (rk[:pos.sum()].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum())

    grid = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    th = np.unique(np.concatenate([grid, np.percentile(p[neg], np.linspace(0, 100, 201))]))
    sweep = [{"tau": round(float(t), 4),
              "tpr": round(float(((p >= t) & pos).sum() / max(pos.sum(), 1)), 4),
              "fpr": round(float(((p >= t) & neg).sum() / max(neg.sum(), 1)), 4)} for t in th]

    rep = {"method": "Qwen2-VL-2B base (未微调)", "index": args.index,
           "n_val": int(len(y)), "n_pos": int(pos.sum()), "n_neg": int(neg.sum()),
           "auc": round(float(auc), 4), "sweep": sweep,
           "scores": [round(float(s), 6) for s in p], "labels": [int(v) for v in y]}
    for b in (0.01, 0.05, 0.10, 0.20):
        c = [s for s in sweep if s["fpr"] <= b]
        rep[f"best_tpr_at_fpr<={int(b*100)}pct"] = max(c, key=lambda s: s["tpr"]) if c else None

    out = args.out or os.path.join(OUT, "eval_base_qwen.json")
    json.dump(rep, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"[base-qwen] AUC={rep['auc']} n_pos={rep['n_pos']} n_neg={rep['n_neg']}")
    for k in ("best_tpr_at_fpr<=1pct", "best_tpr_at_fpr<=5pct",
              "best_tpr_at_fpr<=10pct", "best_tpr_at_fpr<=20pct"):
        print(f"    {k}: {rep[k]}")
    print(f"[base-qwen] wrote {out}")


if __name__ == "__main__":
    main()
