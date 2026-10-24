#!/usr/bin/env python3
"""
失败模式分析 — 模型错分的系统性归类（补"系统性分析"缺口）
============================================================
对 val 集逐样本打分，把错误分为：
  FP-clean  : 干净图被误判为补丁（假阳性）——按来源域细分
  FN-digital: 数字补丁漏检
  FN-physical: 物理补丁漏检
并输出：
  * 各域错误率与错误样本的分数分布
  * 错误样本清单（路径+分数），供后续人工/聚类分析
  * 分数直方图数据（论文可画分布图）

用法:
  $PY code/experiments_tvc/error_analysis.py --tag lora_patch_detector
输出: results/tvc/error_analysis_<tag>.json
"""
import argparse, json, os, sys
import numpy as np

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS = os.path.dirname(CODE_DIR)
OUT = os.path.join(WS, "results", "tvc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="lora_patch_detector")
    ap.add_argument("--index", default=os.path.join(OUT, "lora_data", "index.json"))
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    from PIL import Image
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from lora_mllm import PROMPT, load_lora_for_eval

    ckpt = os.path.join(OUT, args.tag)
    model, proc = load_lora_for_eval(ckpt, args.rank)
    yes_id, no_id = 14004, 8996

    rows = [r for r in json.load(open(args.index)) if r["split"] == "val"]
    print(f"[err] val n={len(rows)}  ckpt={args.tag}")

    scores, labels, doms, paths = [], [], [], []
    for i, r in enumerate(rows):
        img = Image.open(r["path"]).convert("RGB")
        msg = [{"role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        t = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        enc = {k: (v.cuda() if torch.is_tensor(v) else v)
               for k, v in proc(text=[t], images=[img], return_tensors="pt").items()}
        with torch.no_grad():
            lg = model(**enc).logits[0, -1]
        scores.append(float(torch.sigmoid(lg[yes_id] - lg[no_id])))
        labels.append(r["label"]); doms.append(r["domain"]); paths.append(r["path"])
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)

    p = np.array(scores); y = np.array(labels); dm = np.array(doms)
    pred = (p >= args.tau).astype(int)

    cats = {
        "FP-clean": (y == 0) & (pred == 1),
        "FN-digital": (y == 1) & (dm == "digital_patch") & (pred == 0),
        "FN-physical": (y == 1) & (dm == "physical_patch") & (pred == 0),
    }
    summary = {}
    for name, m in cats.items():
        items = [{"path": paths[i], "domain": dm[i], "score": round(float(p[i]), 4)}
                 for i in np.where(m)[0]]
        # 分数排序（最接近阈值的在前=最难样本）
        items.sort(key=lambda x: -min(abs(x["score"] - args.tau), 1))
        summary[name] = {"n": int(m.sum()), "items": items}
        print(f"  {name:12s}: {m.sum():4d} 个")
        if items:
            sc = [x["score"] for x in items]
            print(f"                分数范围 {min(sc):.3f}~{max(sc):.3f}")

    # 各域分数统计（论文可画分布图）
    dist = {}
    for dom in sorted(set(doms)):
        m = dm == dom
        dist[dom] = {"n": int(m.sum()),
                     "hist_edges": [round(float(x), 3) for x in np.linspace(0, 1, 21)],
                     "hist_counts": [int(c) for c in
                                     np.histogram(p[m], bins=20, range=(0, 1))[0]],
                     "mean": round(float(p[m].mean()), 4)}

    # 跨域错误相关性：FP-clean 集中在哪个来源？
    fp_by_dom = {}
    for dom in sorted(set(dm[(y == 0) & (pred == 1)])):
        fp_by_dom[dom] = int(((dm == dom) & (y == 0) & (pred == 1)).sum())

    rep = {"tag": args.tag, "tau": args.tau, "n_val": len(rows),
           "auc_rough": None,
           "error_summary": {k: v["n"] for k, v in summary.items()},
           "fp_by_clean_domain": fp_by_dom,
           "score_distribution": dist,
           "detail": {k: v["items"][:100] for k, v in summary.items()}}
    # AUC
    pos, neg = y == 1, y == 0
    order = np.argsort(np.concatenate([p[pos], p[neg]]))
    rk = np.empty(len(order)); rk[order] = np.arange(1, len(order) + 1)
    rep["auc_rough"] = round(float((rk[:pos.sum()].sum() - pos.sum() * (pos.sum() + 1) / 2)
                                   / (pos.sum() * neg.sum())), 4)

    out = args.out or os.path.join(OUT, f"error_analysis_{args.tag}.json")
    json.dump(rep, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"[err] AUC={rep['auc_rough']}  错误分布: {rep['error_summary']}")
    print(f"[err] FP 按来源: {fp_by_dom}")
    print(f"[err] wrote {out}")


if __name__ == "__main__":
    main()
