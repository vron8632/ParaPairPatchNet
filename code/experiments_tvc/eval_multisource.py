#!/usr/bin/env python3
"""
多源评估 — 分来源报告 FPR/TPR（任务2 的核心证据）
====================================================
confound 的表现是「同一模型对不同来源干净图的 FPR 差异大」。要证明训练后 confound 缓解，
必须**按来源分别报告 FPR**，而不是只给混合 FPR。

用法:
  $PY code/experiments_tvc/eval_multisource.py \
      --tag multi_src --index results/tvc/lora_data/index_multi.json
输出: results/tvc/eval_multisource_<tag>.json
"""
import argparse, json, os, sys
import numpy as np

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS = os.path.dirname(CODE_DIR)
OUT = os.path.join(WS, "results", "tvc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="ckpt 目录名（results/tvc/<tag>）")
    ap.add_argument("--index", default=os.path.join(OUT, "lora_data", "index_multi.json"))
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    from PIL import Image
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from lora_mllm import QWEN_SNAP, PROMPT, load_lora_for_eval

    ckpt = os.path.join(OUT, args.tag)
    if not os.path.isdir(ckpt):
        raise SystemExit(f"缺少 ckpt: {ckpt}")
    # 修复: 必须从纯 base 加载（见 lora_mllm.load_lora_for_eval 注释）
    model, proc = load_lora_for_eval(ckpt, args.rank)
    yes_id, no_id = 14004, 8996

    rows = json.load(open(args.index))
    val = [r for r in rows if r["split"] == "val"]
    # 按 domain 分组
    groups = {}
    for r in val:
        groups.setdefault(r["domain"], []).append(r)
    print(f"[eval] tag={args.tag}  val 域: " +
          ", ".join(f"{k}({len(v)})" for k, v in sorted(groups.items())))

    def score(path):
        img = Image.open(path).convert("RGB")
        msg = [{"role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        t = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        enc = {k: (v.cuda() if torch.is_tensor(v) else v)
               for k, v in proc(text=[t], images=[img], return_tensors="pt").items()}
        with torch.no_grad():
            lg = model(**enc).logits[0, -1]
        return float(torch.sigmoid(lg[yes_id] - lg[no_id]))

    # 全部打分（按域存）
    scores, labels, doms = [], [], []
    for dom, items in sorted(groups.items()):
        for i, r in enumerate(items):
            scores.append(score(r["path"])); labels.append(r["label"]); doms.append(dom)
            if (i + 1) % 50 == 0:
                print(f"  {dom} {i+1}/{len(items)}", flush=True)
    p = np.array(scores); y = np.array(labels); d = np.array(doms)

    # 分域统计
    per_dom = {}
    for dom in sorted(groups):
        m = d == dom
        per_dom[dom] = {"n": int(m.sum()),
                        "n_pos": int((y[m] == 1).sum()),
                        "n_neg": int((y[m] == 0).sum()),
                        "mean": round(float(p[m].mean()), 4),
                        "median": round(float(np.median(p[m])), 4)}
        print(f"  {dom:16s} n={m.sum():4d} mean={p[m].mean():.4f}")

    # 全局 ROC + 分域 FPR
    pos, neg = y == 1, y == 0
    order = np.argsort(np.concatenate([p[pos], p[neg]]))
    rk = np.empty(len(order)); rk[order] = np.arange(1, len(order) + 1)
    auc = float((rk[:pos.sum()].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))

    rep = {"tag": args.tag, "index": args.index, "n": int(len(y)),
           "n_pos": int(pos.sum()), "n_neg": int(neg.sum()),
           "auc_all_neg": round(auc, 4), "per_domain": per_dom,
           "clean_domains": [], "fpr_by_clean_domain": {}}
    # 干净域（负样本来源）之间的分数差异 => confound 指标
    clean_doms = [dd for dd in sorted(groups) if all(
        r["label"] == 0 for r in groups[dd])]
    rep["clean_domains"] = clean_doms
    if clean_doms:
        means = {dd: float(p[d == dd].mean()) for dd in clean_doms}
        rep["clean_domain_means"] = {k: round(v, 4) for k, v in means.items()}
        rep["confound_ratio"] = round(max(means.values()) / max(min(means.values()), 1e-6), 2)
        print(f"\n[confound] 干净域均值: " +
              ", ".join(f"{k}={v:.4f}" for k, v in means.items()))
        print(f"[confound] 最大/最小 比值 = {rep['confound_ratio']}（越接近 1 越无 confound）")

    for b in (0.05, 0.10, 0.20):
        thr = float(np.percentile(p[neg], 100 * (1 - b)))
        rep[f"@FPR{int(b*100)}pct"] = {
            "threshold": round(thr, 4),
            "tpr": round(float((p[pos] >= thr).sum() / max(pos.sum(), 1)), 4),
            "fpr_per_clean_domain": {dd: round(float((p[d == dd] >= thr).sum() /
                                                    max((d == dd).sum(), 1)), 4)
                                     for dd in clean_doms}}
        print(f"  @FPR{int(b*100)}%: TPR={rep[f'@FPR{int(b*100)}pct']['tpr']} "
              f"分域FPR={rep[f'@FPR{int(b*100)}pct']['fpr_per_clean_domain']}")

    out = args.out or os.path.join(OUT, f"eval_multisource_{args.tag}.json")
    json.dump(rep, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"[eval] wrote {out}")


if __name__ == "__main__":
    main()
