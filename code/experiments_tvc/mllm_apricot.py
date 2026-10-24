#!/usr/bin/env python3
"""
MLLM (mimo) 在真实 APRICOT 上的 zero-shot 评估（回应 sim-to-real）
==================================================================
用法:
  export MIMO_KEY=...
  $PY code/experiments_tvc/mllm_apricot.py --n 60 --workers 6
输出: results/tvc/mllm_apricot_n<N>.json
"""
import argparse, base64, io, json, os, sys, time, urllib.request
import numpy as np
from concurrent.futures import ThreadPoolExecutor

URL = "https://api.xiaomimimo.com/v1/chat/completions"
MODEL = os.environ.get("MIMO_MODEL", "mimo-v2.6-pro")
PROMPT = ("Look at this photo. Is there any suspicious square/rectangular region, sticker, or "
          "pasted block that looks out of place? Reply exactly YES or NO as the FIRST word.")
DEFAULT_D = "/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard/data/apricot/dev/data_mask_v2"


def call(b, key, retries=3):
    b64 = base64.b64encode(b).decode()
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
            return (j["choices"][0]["message"].get("content") or "").strip()
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
    ap.add_argument("--dir", default=DEFAULT_D)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    key = os.environ.get("MIMO_KEY", "")
    if not key:
        print("需要 MIMO_KEY"); sys.exit(1)
    import torch
    from PIL import Image

    fs = sorted(os.listdir(args.dir), key=lambda f: int(f.split("_")[1].split(".")[0]))[:args.n]
    jobs = []
    for f in fs:
        x = torch.load(os.path.join(args.dir, f), map_location="cpu", weights_only=False)
        img, mask = x["Image"][0], x["Mask"]
        lab = 1 if mask.sum() > 0 else 0
        # Image 是 float32 [0,1] —— 必须 *255 再转 uint8，否则全黑（此前 bug 导致 TPR=0%）
        arr = np.clip(img.astype(np.float32) * 255.0, 0, 255).astype(np.uint8)
        im = Image.fromarray(arr).convert("RGB")
        im.thumbnail((1024, 1024))
        buf = io.BytesIO(); im.save(buf, format="JPEG", quality=88)
        jobs.append((f, lab, buf.getvalue()))

    t0 = time.time()

    def work(j):
        f, lab, b = j
        return {"file": f, "label": lab, "pred": parse(call(b, key))}

    res = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(work, jobs):
            res.append(r)
            if len(res) % 10 == 0:
                print(f"  {len(res)}/{len(jobs)} ({time.time()-t0:.0f}s)", flush=True)

    tp = sum(1 for r in res if r["label"] == 1 and r["pred"] == 1)
    fn = sum(1 for r in res if r["label"] == 1 and r["pred"] != 1)
    tn = sum(1 for r in res if r["label"] == 0 and r["pred"] == 0)
    fp = sum(1 for r in res if r["label"] == 0 and r["pred"] != 0)
    rep = {"dataset": "APRICOT-Mask (real)", "n": len(res), "model": MODEL,
           "n_patch": tp + fn, "n_clean": tn + fp,
           "tp": tp, "fn": fn, "tn": tn, "fp": fp,
           "tpr": round(tp / max(tp + fn, 1), 4), "fpr": round(fp / max(fp + tn, 1), 4),
           "accuracy": round((tp + tn) / max(len(res), 1), 4), "seconds": round(time.time() - t0, 1),
           "results": res}
    out = args.out or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   "results", "tvc", f"mllm_apricot_n{args.n}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print(f"[apricot-mllm] TPR={rep['tpr']} ({tp}/{tp+fn})  FPR={rep['fpr']} ({fp}/{fp+tn})  "
          f"acc={rep['accuracy']}  (patch={tp+fn}, clean={tn+fp})")
    print(f"[apricot-mllm] wrote {out}")


if __name__ == "__main__":
    main()
