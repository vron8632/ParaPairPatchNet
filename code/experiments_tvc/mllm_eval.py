#!/usr/bin/env python3
"""
MLLM (mimo) zero-shot 补丁检测评估
====================================
用小米 mimo-v2.6-pro 的视觉能力做 zero-shot 判断："图中是否有可疑的方形/贴纸区域"。
API key 从环境变量 MIMO_KEY 读取（不写入仓库）。

用法:
  export MIMO_KEY=...
  PY=/home/oyp/miniconda3/envs/apjf/bin/python
  $PY code/experiments_tvc/mllm_eval.py --base <SRC> --n 50 --workers 4
输出: results/tvc/mllm_eval_n<N>.json
"""
import argparse, base64, io, json, os, sys, time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "https://api.xiaomimimo.com/v1/chat/completions"
MODEL = os.environ.get("MIMO_MODEL", "mimo-v2.6-pro")
PROMPT = ("Look at this photo. Is there any suspicious square/rectangular region, sticker, or "
          "pasted block that looks out of place (e.g. a colored or textured square on top of a "
          "natural scene)? Reply exactly YES or NO as the FIRST word.")


def call(img_bytes, key, retries=3):
    b64 = base64.b64encode(img_bytes).decode()
    body = {"model": MODEL, "messages": [{"role": "user", "content": [
        {"type": "text", "text": PROMPT},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}}]}],
        "max_tokens": 1500}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + key,
                                          "Content-Type": "application/json"})
    last = ""
    for _ in range(retries):
        try:
            j = json.loads(urllib.request.urlopen(req, timeout=120).read().decode())
            m = j["choices"][0]["message"]
            return (m.get("content") or "").strip()
        except Exception as e:
            last = "ERR " + str(e)
            time.sleep(2)
    return last


def parse(text):
    t = text.strip().upper()
    if t.startswith("YES"):
        return 1
    if t.startswith("NO"):
        return 0
    # 回退：找首个 YES/NO
    for w in t.replace(".", " ").replace(",", " ").split():
        if w in ("YES", "NO"):
            return 1 if w == "YES" else 0
    return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    key = os.environ.get("MIMO_KEY", "")
    if not key:
        print("需要设置环境变量 MIMO_KEY"); sys.exit(1)
    import torch
    from PIL import Image

    pd = os.path.join(args.base, "results", "features", "patch_dataset_v2")
    jobs = []
    for dom, lab in [("clean", 0), ("physical_patch", 1)]:
        d = torch.load(os.path.join(pd, f"{dom}_test.pt"), map_location="cpu", weights_only=False)
        for i in range(args.n):
            t = d["tensors"][i]
            img = Image.fromarray((t.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype("uint8"))
            buf = io.BytesIO(); img.save(buf, format="JPEG", quality=90)
            jobs.append((dom, lab, i, buf.getvalue()))

    t0 = time.time()
    results = []
    def work(j):
        dom, lab, i, b = j
        txt = call(b, key)
        return {"domain": dom, "label": lab, "idx": i, "pred": parse(txt), "raw": txt[:200]}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for k, r in enumerate(ex.map(work, jobs)):
            results.append(r)
            if (k + 1) % 10 == 0:
                print(f"  {k+1}/{len(jobs)} ({time.time()-t0:.0f}s)", flush=True)

    tp = sum(1 for r in results if r["label"] == 1 and r["pred"] == 1)
    fn = sum(1 for r in results if r["label"] == 1 and r["pred"] != 1)
    tn = sum(1 for r in results if r["label"] == 0 and r["pred"] == 0)
    fp = sum(1 for r in results if r["label"] == 0 and r["pred"] != 0)
    rep = {"model": MODEL, "n_per_class": args.n, "tp": tp, "fn": fn, "tn": tn, "fp": fp,
           "tpr": round(tp / max(tp + fn, 1), 4), "fpr": round(fp / max(fp + tn, 1), 4),
           "accuracy": round((tp + tn) / len(results), 4),
           "parse_fail": sum(1 for r in results if r["pred"] == -1),
           "seconds": round(time.time() - t0, 1), "results": results}
    out = args.out or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   "results", "tvc", f"mllm_eval_n{args.n}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print(f"[mllm] TPR={rep['tpr']} FPR={rep['fpr']} acc={rep['accuracy']} "
          f"(tp={tp} fn={fn} tn={tn} fp={fp}, fail={rep['parse_fail']})")
    print(f"[mllm] wrote {out}")


if __name__ == "__main__":
    main()
