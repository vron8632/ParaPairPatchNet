#!/usr/bin/env python3
"""
mimo-v2.6 zero-shot 在同源 val 上的评估（与 LoRA/Qwen-base 同一评估集）
========================================================================
目的：公平对比三种 MLLM 方案——
  (a) Qwen2-VL-2B base（未微调）
  (b) Qwen2-VL-2B + LoRA + FPR 约束（ours）
  (c) mimo-v2.6-pro zero-shot（云端 API，不可微调）
评估集：results/tvc/lora_data/index.json 的 val（clean 200 + digital 400 + physical 400）
API key 从环境变量 MIMO_KEY 读取。

用法:
  export MIMO_KEY=...
  $PY code/experiments_tvc/mimo_val_eval.py --n 1000 --workers 6
输出: results/tvc/mimo_val_eval.json
"""
import argparse, base64, io, json, os, sys, time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import urllib.request

URL = "https://api.xiaomimimo.com/v1/chat/completions"
MODEL = os.environ.get("MIMO_MODEL", "mimo-v2.6-pro")
PROMPT = ("Is there a suspicious printed sticker or adversarial patch attached to an object "
          "in this photo? Answer YES or NO.")


def call(jpeg_bytes, key, retries=3):
    b64 = base64.b64encode(jpeg_bytes).decode()
    body = {"model": MODEL, "messages": [{"role": "user", "content": [
        {"type": "text", "text": PROMPT},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}}]}],
        "max_tokens": 1500}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + key,
                                          "Content-Type": "application/json"})
    for _ in range(retries):
        try:
            j = json.loads(urllib.request.urlopen(req, timeout=120).read().decode())
            m = j["choices"][0]["message"]
            return (m.get("content") or "").strip()
        except Exception:
            time.sleep(2)
    return "ERR"


def parse(t):
    t = t.strip().upper()
    if t.startswith("YES"):
        return 1
    if t.startswith("NO"):
        return 0
    return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..", "results", "tvc", "lora_data", "index.json"))
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    key = os.environ.get("MIMO_KEY", "")
    if not key:
        print("需要 MIMO_KEY"); sys.exit(1)

    idxp = os.path.abspath(args.index)
    rows = [r for r in json.load(open(idxp)) if r["split"] == "val"][:args.n]
    print(f"[mimo-val] n={len(rows)}")

    jobs = []
    for r in rows:
        with open(r["path"], "rb") as f:
            jobs.append((r["domain"], r["label"], r["path"], f.read()))

    t0 = time.time()
    res = []

    def work(j):
        dom, lab, path, b = j
        return {"domain": dom, "label": lab, "path": path, "pred": parse(call(b, key))}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for k, r in enumerate(ex.map(work, jobs)):
            res.append(r)
            if (k + 1) % 50 == 0:
                print(f"  {k+1}/{len(jobs)} ({time.time()-t0:.0f}s)", flush=True)

    y = np.array([r["label"] for r in res])
    p = np.array([1.0 if r["pred"] == 1 else (0.0 if r["pred"] == 0 else 0.5) for r in res])
    pos, neg = y == 1, y == 0
    # mimo 只给二值，AUC 用平滑处理（0.5 计为平局）
    order = np.argsort(np.concatenate([p[pos], p[neg]]))
    rk = np.empty(len(order)); rk[order] = np.arange(1, len(order) + 1)
    auc = (rk[:pos.sum()].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum())

    per_dom = {}
    for dom in sorted(set(r["domain"] for r in res)):
        m = np.array([r["domain"] == dom for r in res])
        per_dom[dom] = {"n": int(m.sum()), "mean": round(float(p[m].mean()), 4)}
        print(f"  {dom:16s} n={m.sum():4d} mean={p[m].mean():.4f}")

    rep = {"baseline": "mimo-v2.6-pro zero-shot (不可微调的云端 MLLM)",
           "index": idxp, "n": int(len(y)),
           "n_pos": int(pos.sum()), "n_neg": int(neg.sum()),
           "auc": round(float(auc), 4), "per_domain": per_dom,
           "parse_fail": int(sum(1 for r in res if r["pred"] == -1)),
           "seconds": round(time.time() - t0, 1),
           "results": res}
    for b in (0.01, 0.05, 0.10, 0.20):
        thr = float(np.percentile(p[neg], 100 * (1 - b))) if neg.sum() else 0.5
        tpr = float((p[pos] >= thr).sum() / max(pos.sum(), 1))
        rep[f"@FPR{int(b*100)}pct"] = {"threshold": round(thr, 4), "tpr": round(tpr, 4)}

    out = args.out or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   "..", "results", "tvc", "mimo_val_eval.json")
    json.dump(rep, open(os.path.abspath(out), "w"), indent=2, ensure_ascii=False)
    print(f"[mimo-val] AUC={rep['auc']} @5%={rep['@FPR5pct']['tpr']} "
          f"@10%={rep['@FPR10pct']['tpr']} @20%={rep['@FPR20pct']['tpr']}")


if __name__ == "__main__":
    main()
