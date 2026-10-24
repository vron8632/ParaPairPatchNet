#!/usr/bin/env python3
"""
真实数据工作点评估 — 动机: print-and-capture
======================================================================
论文的补丁是 PC 仿真（审阅者质疑"模型识别的是模拟器"）。本脚本用**完全真实**的数据测：
  正样本: APRICOT-Mask 中带 mask 的图（真实打印补丁 + 真实拍摄）
  负样本: CASIA v2 Au 干净照片（真实拍摄、无补丁）
模型: mimo-v2.6-pro zero-shot（无任何训练），得到真实数据上的 (TPR, FPR)。

用法:
  export MIMO_KEY=...
  $PY code/experiments_tvc/mllm_real_workpoint.py --n-pos 100 --n-neg 200 --workers 6
输出: results/tvc/mllm_real_workpoint.json
"""
import argparse, base64, io, json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np

URL = "https://api.xiaomimimo.com/v1/chat/completions"
MODEL = os.environ.get("MIMO_MODEL", "mimo-v2.6-pro")
PROMPT = ("Look at this photo. Is there any suspicious square/rectangular region, sticker, or "
          "pasted block that looks out of place? Reply exactly YES or NO as the FIRST word.")
APRICOT_D = "/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard/data/apricot/dev/data_mask_v2"
CASIA_AU = "/media/oyp/工作/Projects/042_image_forensic/DuetGuard/data/casia2/CASIA2.0_revised/Au"


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
    ap.add_argument("--n-pos", type=int, default=100, help="APRICOT 含补丁图")
    ap.add_argument("--n-neg", type=int, default=200, help="CASIA Au 干净图")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    key = os.environ.get("MIMO_KEY", "")
    if not key:
        print("需要 MIMO_KEY"); sys.exit(1)
    import torch
    from PIL import Image

    jobs = []
    # 正样本: APRICOT 含 mask 的图
    fs = sorted(os.listdir(APRICOT_D), key=lambda f: int(f.split("_")[1].split(".")[0]))
    pos = 0
    for f in fs:
        if pos >= args.n_pos:
            break
        x = torch.load(os.path.join(APRICOT_D, f), map_location="cpu", weights_only=False)
        if x["Mask"].sum() <= 0:
            continue
        arr = np.clip(x["Image"][0].astype(np.float32) * 255.0, 0, 255).astype(np.uint8)
        im = Image.fromarray(arr).convert("RGB"); im.thumbnail((1024, 1024))
        buf = io.BytesIO(); im.save(buf, format="JPEG", quality=88)
        jobs.append(("apricot", 1, f, buf.getvalue()))
        pos += 1

    # 负样本: CASIA Au 真实干净照片（打散选样，避免顺序偏差）
    au = sorted(fn for fn in os.listdir(CASIA_AU)
                if fn.lower().endswith((".jpg", ".jpeg", ".png")))
    rng = np.random.RandomState(42)
    pick = [au[i] for i in rng.permutation(len(au))[:args.n_neg]]
    for fn in pick:
        im = Image.open(os.path.join(CASIA_AU, fn)).convert("RGB")
        im.thumbnail((1024, 1024))
        buf = io.BytesIO(); im.save(buf, format="JPEG", quality=88)
        jobs.append(("casia_au", 0, fn, buf.getvalue()))

    print(f"[mllm-real] 正样本 {sum(1 for j in jobs if j[1]==1)} / 负样本 {sum(1 for j in jobs if j[1]==0)}")
    t0 = time.time()

    def work(j):
        src, lab, name, b = j
        return {"src": src, "label": lab, "name": name, "pred": parse(call(b, key))}

    res = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(work, jobs):
            res.append(r)
            if len(res) % 20 == 0:
                print(f"  {len(res)}/{len(jobs)} ({time.time()-t0:.0f}s)", flush=True)

    tp = sum(1 for r in res if r["label"] == 1 and r["pred"] == 1)
    fn = sum(1 for r in res if r["label"] == 1 and r["pred"] != 1)
    tn = sum(1 for r in res if r["label"] == 0 and r["pred"] == 0)
    fp = sum(1 for r in res if r["label"] == 0 and r["pred"] != 0)
    rep = {"dataset": "APRICOT(pos, real print+capture) vs CASIA-Au(neg, real photos)",
           "model": MODEL, "prompt": PROMPT,
           "n_pos": tp + fn, "n_neg": tn + fp, "tp": tp, "fn": fn, "tn": tn, "fp": fp,
           "tpr": round(tp / max(tp + fn, 1), 4), "fpr": round(fp / max(fp + tn, 1), 4),
           "clean_acc": round(tn / max(tn + fp, 1), 4),
           "balanced_acc": round(0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)), 4),
           "seconds": round(time.time() - t0, 1), "results": res}
    out = args.out or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   "results", "tvc", "mllm_real_workpoint.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print(f"[mllm-real] *** 真实数据工作点: TPR={rep['tpr']} ({tp}/{tp+fn})  "
          f"FPR={rep['fpr']} ({fp}/{fp+tn})  clean_acc={rep['clean_acc']}  "
          f"balanced={rep['balanced_acc']}")
    print(f"[mllm-real] wrote {out}")


if __name__ == "__main__":
    main()
