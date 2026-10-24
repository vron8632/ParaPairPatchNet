#!/usr/bin/env python3
"""
主表 v2（扩展版）：多口径 × 多 seed × 扩容正样本
================================================================
对每个 ckpt（seed0/1/2）在以下口径下评测：
  正样本：pos_all（138，本地全部）与 pos（100，旧口径，兼容对比）
  负样本：3 源 180 / 4 源 240（旧）；3 源 750 / 4 源 1000（扩容集）
输出 TPR@FPR + Wilson CI + Bootstrap CI，供决定主表口径。
输出: results/tvc/main_table_v2.json
用法: $PY main_table_v2.py            # 全部 ckpt
      $PY main_table_v2.py --ckpt results/tvc/lora_patch_detector
"""
import os, sys, json, argparse, math, time
import numpy as np

WS = "/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard"
TVC = f"{WS}/results/tvc"
BASE = f"{TVC}/baseline_imgs"
S3 = ["casia_au", "coco_val", "inria_noped"]
S4 = S3 + ["columbia_au"]


_IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
def _list_imgs(d):
    return sorted(f for f in os.listdir(d) if f.lower().endswith(_IMG_EXT))

D_DATA = "/media/oyp/工作/Projects/042_image_forensic/DuetGuard/data"
_IMG_EXT = (".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff")


def collect_columbia_au(n_per=60, seed=42):
    """Columbia **authentic** 图（只取 Au-*；Sp-* 是拼接篡改图，绝不能当干净负样本）。
    修复记录：旧逻辑 `startswith("Au-S")` 会采到 56% 的 Sp-* 拼接图。"""
    from PIL import Image
    rng = np.random.RandomState(seed)
    col = f"{D_DATA}/columbia/ImSpliceDataset"
    items = []
    if os.path.isdir(col):
        for sub in sorted(os.listdir(col)):
            d2 = os.path.join(col, sub)
            if not os.path.isdir(d2) or not sub.startswith("Au"):
                continue
            for f in sorted(os.listdir(d2)):
                if f.lower().endswith(_IMG_EXT):
                    items.append(f"{sub}/{f}")
    rng.shuffle(items)
    out = []
    for nm in items[:n_per]:
        try:
            im = Image.open(os.path.join(col, nm)).convert("RGB")
            im.thumbnail((1024, 1024)); out.append((nm, im))
        except Exception:
            pass
    return out


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (max(0.0, c-h), min(1.0, c+h))


def score_dir(model, proc, d, yes_id, no_id, limit=None):
    """用 VLM 对目录内图像打分（复用 confound_check 的推理路径）"""
    import torch
    from PIL import Image
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from lora_mllm import PROMPT
    files = _list_imgs(d)
    if limit: files = files[:limit]
    out = []
    for i, f in enumerate(files):
        im = Image.open(os.path.join(d, f)).convert("RGB")
        im.thumbnail((512, 512))
        msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        t = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        enc = {k: (v.cuda() if torch.is_tensor(v) else v)
               for k, v in proc(text=[t], images=[im], return_tensors="pt").items()}
        with torch.no_grad():
            lg = model(**enc).logits[0, -1]
        out.append(float(torch.sigmoid(lg[yes_id] - lg[no_id])))
        if (i+1) % 100 == 0: print(f"    {os.path.basename(d)} {i+1}/{len(files)}", flush=True)
    return out


def score_images(model, proc, imgs, yes_id, no_id):
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from lora_mllm import PROMPT
    out = []
    for nm, im in imgs:
        im = im.copy(); im.thumbnail((512, 512))
        msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        t = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        enc = {k: (v.cuda() if torch.is_tensor(v) else v)
               for k, v in proc(text=[t], images=[im], return_tensors="pt").items()}
        with torch.no_grad():
            lg = model(**enc).logits[0, -1]
        out.append(float(torch.sigmoid(lg[yes_id] - lg[no_id])))
    return out


def eval_ckpt(ckpt, res):
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from lora_mllm import load_lora_for_eval
    model, proc = load_lora_for_eval(ckpt, 16)
    cfg = json.load(open(f"{ckpt}/train_config.json"))
    yes_id, no_id = cfg["yes_id"], cfg["no_id"]
    out = {"config": {k: cfg.get(k) for k in ("constraint_mode", "train_scope", "train_frac", "seed")}}
    # 正样本：138 优先，回退 100
    pos_dir = f"{BASE}/pos_all" if os.path.isdir(f"{BASE}/pos_all") else f"{BASE}/pos"
    pos = np.array(score_dir(model, proc, pos_dir, yes_id, no_id))
    out["n_pos"] = len(pos)
    # columbia 不再从 baseline_imgs 读（该目录不存在且历史上采到了拼接图），
    # 改为按正确逻辑从原始数据集采集 authentic 图
    neg = {s: np.array(score_dir(model, proc, f"{BASE}/{s}", yes_id, no_id))
           for s in S3}
    col_imgs = collect_columbia_au(60)
    print(f"    columbia_au (authentic): {len(col_imgs)}", flush=True)
    neg["columbia_au"] = np.array(score_images(model, proc, col_imgs, yes_id, no_id))
    out["neg_n"] = {s: len(v) for s, v in neg.items()}
    out["pos_scores"] = [round(float(x), 4) for x in pos]
    out["neg_scores_by_source"] = {s: [round(float(x), 4) for x in v] for s, v in neg.items()}
    # 多口径 TPR@FPR
    out["tpr"] = {}
    for tag, keys in (("3src", S3), ("4src", S4)):
        n = np.concatenate([neg[k] for k in keys])
        row = {}
        for a in (0.01, 0.05, 0.10, 0.20):
            thr = np.percentile(n, 100*(1-a)); k_ = int((pos >= thr).sum())
            lo, hi = wilson(k_, len(pos))
            row[f"@{int(a*100)}pct"] = {"tpr": round(k_/len(pos), 4),
                                        "wilson": [round(lo, 4), round(hi, 4)]}
        out["tpr"][tag] = row
        print(f"  [{tag}] " + "  ".join(f"{k}={v['tpr']:.3f}" for k, v in row.items()), flush=True)
    del model
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", default="lora_patch_detector,full_lora_s1,full_lora_s2")
    ap.add_argument("--out", default=f"{TVC}/main_table_v2.json")
    args = ap.parse_args()
    res = {}
    for name in args.ckpts.split(","):
        ck = f"{TVC}/{name}"
        if not os.path.isdir(ck): print(f"[skip] {name} 不存在"); continue
        print(f"=== {name} ===", flush=True)
        res[name] = eval_ckpt(ck, res)
    # 汇总 mean±std（跨 seed）
    if len(res) > 1:
        summ = {}
        for tag in ("3src", "4src"):
            for a in (0.01, 0.05, 0.10, 0.20):
                vals = [r["tpr"][tag][f"@{int(a*100)}pct"]["tpr"] for r in res.values()]
                summ[f"{tag}@{int(a*100)}pct"] = {"mean": round(float(np.mean(vals)), 4),
                                                  "std": round(float(np.std(vals)), 4),
                                                  "all": [round(v, 4) for v in vals]}
        res["summary_mean_std"] = summ
        print("\n=== 跨 seed 汇总 ===")
        for k, v in summ.items(): print(f"  {k}: {v['mean']:.4f}±{v['std']:.4f}  {v['all']}")
    json.dump(res, open(args.out, "w"), indent=1, ensure_ascii=False)
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
