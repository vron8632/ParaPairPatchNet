#!/usr/bin/env python3
"""
负样本规模扩容评测（回应理论章节命题 3 的 DKW 界要求：n≈1000 才能把 ±8.8pt 压到 ±4pt）
=====================================================================================
与 confound_check.py 的差别：
  - 只采集**负样本**（4 源各 n_per 张），**不读 APRICOT**（省 17 分钟外置盘 IO）
  - 正样本分数**复用** confound_main_withscores.json 的 pos_scores（同一 ckpt 的确定性输出）
  - 额外输出 DKW 一致置信半径 ε 与"FPR 真实上界"

用法:
  $PY code/experiments_tvc/eval_neg_scale.py \
      --ckpt results/tvc/lora_patch_detector --n-per-source 250 \
      --pos-json results/tvc/confound_main_withscores.json \
      --out results/tvc/confound_scale1k.json
"""
import argparse, json, math, os, sys, time

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)
OUT = os.path.join(os.path.dirname(CODE_DIR), "results", "tvc")
D = "/media/oyp/工作/Projects/042_image_forensic/DuetGuard/data"
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def collect_neg(n_per, sources, seed=42):
    """只采集指定来源的负样本（不碰 APRICOT）。"""
    import numpy as np
    from PIL import Image
    rng = np.random.RandomState(seed)
    out = {}

    def _pick(dirpath, tag, filt=None):
        if not os.path.isdir(dirpath):
            print(f"  [skip] {tag}: 路径不存在 {dirpath}")
            return
        fs = sorted(f for f in os.listdir(dirpath)
                    if f.lower().endswith(IMG_EXT) and (filt is None or filt(f)))
        if not fs:
            print(f"  [skip] {tag}: 无图像")
            return
        pick = [fs[i] for i in rng.permutation(len(fs))[:n_per]]
        items = []
        for f in pick:
            try:
                im = Image.open(os.path.join(dirpath, f)).convert("RGB")
                im.thumbnail((1024, 1024))
                items.append((f"{tag}/{f}", im))
            except Exception:
                continue
        out[tag] = items
        print(f"  {tag}: {len(items)}/{len(fs)} (seed={seed})")

    if "casia_au" in sources:
        _pick(f"{D}/casia2/CASIA2.0_revised/Au", "casia_au")
    if "coco_val" in sources:
        _pick(f"{D}/coco/val2017", "coco_val")
    if "inria_noped" in sources:
        _pick(f"{D}/inria/data_ped/no_pedestrians", "inria_noped")
    if "columbia_au" in sources:
        col = f"{D}/columbia/ImSpliceDataset"
        if os.path.isdir(col):
            items = []
            for sub in sorted(os.listdir(col)):
                d2 = os.path.join(col, sub)
            # ⚠️ 修复（2026-10-16）：只取 Au-*（authentic）。旧逻辑 startswith("Au-S")
            # 只排除 Au-S，却把 Sp-*（拼接篡改图）当成干净负样本（实测 56% 污染）。
                if not os.path.isdir(d2) or sub.not sub.startswith("Au"):
                    continue
                for fn in sorted(os.listdir(d2)):
                    if fn.lower().endswith(IMG_EXT):
                        items.append((f"columbia_au/{sub}/{fn}", d2, fn))
            rng.shuffle(items)
            keep = items[:n_per]
            loaded = []
            for nm, d2, fn in keep:
                try:
                    im = Image.open(os.path.join(d2, fn)).convert("RGB")
                    im.thumbnail((1024, 1024))
                    loaded.append((nm, im))
                except Exception:
                    continue
            out["columbia_au"] = loaded
            print(f"  columbia_au: {len(loaded)}/{len(items)} (seed={seed})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-per-source", type=int, default=250)
    ap.add_argument("--sources", default="casia_au,coco_val,inria_noped,columbia_au")
    ap.add_argument("--pos-json", default=os.path.join(OUT, "confound_main_withscores.json"),
                    help="复用其中的 pos_scores（省去 APRICOT 读取）")
    ap.add_argument("--out", default=os.path.join(OUT, "confound_scale1k.json"))
    ap.add_argument("--rank", type=int, default=16)
    args = ap.parse_args()

    import numpy as np
    import torch
    from PIL import Image
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from lora_mllm import load_lora_for_eval, PROMPT

    srcs = [s.strip() for s in args.sources.split(",") if s.strip()]

    # 1) 复用正样本分数
    pj = json.load(open(args.pos_json))
    pos = np.asarray(pj["pos_scores"], dtype=float)
    print(f"[scale] 复用正样本分数 n={len(pos)} (来自 {os.path.basename(args.pos_json)}, "
          f"ckpt={os.path.basename(pj.get('ckpt','?'))})")

    # 2) 采集负样本
    neg = collect_neg(args.n_per_source, srcs)
    n_neg = sum(len(v) for v in neg.values())
    print(f"[scale] 负样本合计 n={n_neg}  来源={ {k: len(v) for k, v in neg.items()} }")

    # 3) 模型
    model, proc = load_lora_for_eval(args.ckpt, args.rank)
    cfgp = os.path.join(args.ckpt, "train_config.json")
    if os.path.exists(cfgp):
        c = json.load(open(cfgp)); yes_id, no_id = c["yes_id"], c["no_id"]
    else:
        yes_id, no_id = 14004, 8996

    def score(img):
        msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        t = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        enc = {k: (v.cuda() if torch.is_tensor(v) else v)
               for k, v in proc(text=[t], images=[img], return_tensors="pt").items()}
        with torch.no_grad():
            lg = model(**enc).logits[0, -1]
        return float(torch.sigmoid(lg[yes_id] - lg[no_id]))

    t0 = time.time()
    neg_scores = {}
    for src, items in neg.items():
        ss = []
        for i, (nm, im) in enumerate(items):
            ss.append(score(im))
            if (i + 1) % 50 == 0:
                print(f"  {src} {i+1}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
        neg_scores[src] = ss

    # 4) 统计
    allneg = np.concatenate([np.asarray(neg_scores[s]) for s in srcs if neg_scores.get(s)])
    n = len(allneg)
    eps = math.sqrt(math.log(2 / 0.05) / (2 * n))       # DKW, delta=0.05
    per_source = {s: {"n": len(v), "mean": round(float(np.mean(v)), 4),
                      "median": round(float(np.median(v)), 4),
                      "p90": round(float(np.percentile(v, 90)), 4)} for s, v in neg_scores.items()}

    result = {"ckpt": args.ckpt, "n_pos": int(len(pos)), "n_neg": int(n),
              "n_per_source": args.n_per_source, "sources": srcs,
              "pos_stats": {"mean": round(float(pos.mean()), 4),
                            "median": round(float(np.median(pos)), 4)},
              "neg_per_source": per_source,
              "dkw_epsilon_delta0.05": round(eps, 4),
              "pos_names": pj.get("pos_names"),
              "pos_scores": [round(float(s), 6) for s in pos],
              "neg_scores_by_source": {k: [round(float(x), 6) for x in v] for k, v in neg_scores.items()}}

    print(f"\n=== 结果 n_neg={n} DKW ε={eps:.4f} ===")
    for a in (0.01, 0.05, 0.10, 0.20, 0.33):
        thr = float(np.percentile(allneg, 100 * (1 - a)))
        tpr = float((pos >= thr).mean())
        fprs = {s: round(float((np.asarray(neg_scores[s]) >= thr).mean()), 4)
                for s in srcs if neg_scores.get(s)}
        result[f"@target_FPR{int(a*100)}pct"] = {
            "threshold": round(thr, 4), "tpr": round(tpr, 4),
            "fpr_per_source": fprs,
            "dkw_upper_bound": round(a + eps, 4)}
        print(f"目标 FPR={int(a*100):2d}%: τ={thr:.4f} TPR={tpr:.4f} "
              f"DKW上界={a+eps:.4f}  FPR/源={fprs}")

    json.dump(result, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"\n[scale] wrote {args.out}")


if __name__ == "__main__":
    main()
