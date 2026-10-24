#!/usr/bin/env python3
"""
YOLO26n-cls 统一协议评测（复用其原生 imgsz=512 推理，评测集改为 unified_eval_set）
================================================================================
正样本 138 + 负样本 4 源各 250（authentic columbia），与 unified_main/baselines 完全同一评测集。
输出: results/tvc/unified_yolo26.json
"""
import os, json, math
import numpy as np
from PIL import Image

WS = "/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard"
TVC = f"{WS}/results/tvc"
EVAL = f"{TVC}/unified_eval_set"
S3 = ["casia_au", "coco_val", "inria_noped"]
S4 = S3 + ["columbia_au"]


def _fit_long_side(im, size):
    """统一输入契约：按长边缩放到 size（小图放大、大图缩小），保持宽高比。"""
    w, h = im.size
    s = size / max(w, h)
    if abs(s - 1.0) < 1e-6:
        return im
    return im.resize((max(1, int(round(w*s))), max(1, int(round(h*s)))), Image.BILINEAR)

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k/n; d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (round(max(0.0, c-h), 4), round(min(1.0, c+h), 4))


def main():
    from ultralytics import YOLO
    from PIL import Image
    m = YOLO(f"{TVC}/yolo26_cls/weights/best.pt")
    print(f"[yolo26] classes = {m.names}")

    def score_dir(d):
        out = []
        fs = sorted(os.listdir(d))
        for i, f in enumerate(fs):
            arr = np.asarray(Image.open(os.path.join(d, f)).convert("RGB"))
            r = m.predict(arr, imgsz=512, verbose=False)[0]
            pi = next(i for i, nm in r.names.items() if "patch" in nm.lower())
            out.append(float(r.probs.data.tolist()[pi]))
            if (i+1) % 200 == 0: print(f"    {os.path.basename(d)} {i+1}/{len(fs)}", flush=True)
        return out

    p = np.array(score_dir(f"{EVAL}/pos"))
    ns = {s: np.array(score_dir(f"{EVAL}/{s}")) for s in S4}
    res = {"model": "YOLO26n-cls (fine-tuned, imgsz=512)", "protocol": "unified_eval_set",
           "n_pos": len(p), "pos_mean": round(float(p.mean()), 4),
           "neg_mean": {k: round(float(v.mean()), 4) for k, v in ns.items()}, "tpr": {}}
    for scope, keys in (("3src", S3), ("4src", S4)):
        n_all = np.concatenate([ns[k] for k in keys]); rr = {}
        for a in (0.01, 0.05, 0.10, 0.20):
            thr = float(np.percentile(n_all, 100*(1-a))); k_ = int((p >= thr).sum())
            rr[f"@{int(a*100)}pct"] = {"tpr": round(k_/len(p), 4), "wilson": wilson(k_, len(p)),
                                       "thr": round(thr, 4)}
        res["tpr"][scope] = rr
        print(f"  [{scope}] " + "  ".join(f"{a}={rr[f'@{a}pct']['tpr']:.3f}" for a in (1,5,10,20)), flush=True)
    json.dump(res, open(f"{TVC}/unified_yolo26.json", "w"), indent=1, ensure_ascii=False)
    print(f"→ {TVC}/unified_yolo26.json")


if __name__ == "__main__":
    main()
