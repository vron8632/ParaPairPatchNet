#!/usr/bin/env python3
"""
Confound 排除实验 — 多来源负样本交叉检验（回应"正负不同源"的致命质疑）
========================================================================
此前 lora_real_workpoint 的正样本 100% 来自 APRICOT、负样本 100% 来自 CASIA-Au，
模型可能学到"数据集来源"而非"有无补丁"。

本实验设计：
  正样本 : APRICOT 中带 mask 的真实补丁图（多张）
  负样本 : **五个不同来源**的无补丁图像
           - apricot_clean : APRICOT 自身 6 张无补丁图（同源对照，最关键）
           - casia_au      : CASIA v2 Au 真实照片
           - coco_val      : COCO val2017（与 APRICOT 完全不同的相机/场景分布）
           - columbia_au   : Columbia 无篡改图
           - casia_tamper_free: CASIA 未篡改子集
判据: 若模型对**所有来源**的干净图都给出低分（FPR 一致低），则 confound 被排除；
      若只对 CASIA 低分而对 COCO/Columbia 高分，则坐实"在识别数据集"。

用法:
  $PY code/experiments_tvc/confound_check.py --n-per-source 60
输出: results/tvc/confound_check.json
"""
import argparse, json, os, sys, time

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)
OUT = os.path.join(os.path.dirname(CODE_DIR), "results", "tvc")
D = "/media/oyp/工作/Projects/042_image_forensic/DuetGuard/data"
MM = "/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard"
APRICOT_D = f"{MM}/data/apricot/dev/data_mask_v2"

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def collect_neg(n_per, seed=42):
    """收集各来源的无补丁图像。"""
    import numpy as np
    from PIL import Image
    rng = np.random.RandomState(seed)
    out = {}

    # 1) APRICOT 自身的 clean（同源对照）
    files = sorted(os.listdir(APRICOT_D), key=lambda f: int(f.split("_")[1].split(".")[0]))
    items = []
    for f in files:
        x = torch_load(os.path.join(APRICOT_D, f))
        if x["Mask"].sum() <= 0:
            arr = np.clip(x["Image"][0].astype(np.float32) * 255.0, 0, 255).astype(np.uint8)
            items.append((f"apricot_clean/{f}", Image.fromarray(arr).convert("RGB")))
    out["apricot_clean"] = items
    print(f"  apricot_clean: {len(items)} (APRICOT 全部无补丁图)")

    # 2) CASIA Au
    au = f"{D}/casia2/CASIA2.0_revised/Au"
    if os.path.isdir(au):
        fs = sorted(f for f in os.listdir(au) if f.lower().endswith(IMG_EXT))
        pick = [fs[i] for i in rng.permutation(len(fs))[:n_per]]
        out["casia_au"] = [(f"casia_au/{f}", Image.open(os.path.join(au, f)).convert("RGB"))
                           for f in pick]

    # 3) COCO val2017
    coco = f"{D}/coco/val2017"
    if os.path.isdir(coco):
        fs = sorted(f for f in os.listdir(coco) if f.lower().endswith(IMG_EXT))
        pick = [fs[i] for i in rng.permutation(len(fs))[:n_per]]
        out["coco_val"] = [(f"coco_val/{f}", Image.open(os.path.join(coco, f)).convert("RGB"))
                           for f in pick]

    # 4) Columbia Au（无篡改）
    col = f"{D}/columbia/ImSpliceDataset"
    if os.path.isdir(col):
        items = []
        for sub in sorted(os.listdir(col)):
            d2 = os.path.join(col, sub)
            # ⚠️ 修复（2026-10-16）：必须只取 Au-*（authentic）。
            # 旧逻辑 `sub.startswith("Au-S")` 只排除了 Au-S，却把 Sp-*（spliced，
            # 拼接篡改图）当成干净负样本 —— 实测旧逻辑采到的 1626 张中 56% 是拼接图。
            if not os.path.isdir(d2) or not sub.startswith("Au"):
                continue
            for fn in sorted(os.listdir(d2)):
                if fn.lower().endswith(IMG_EXT):
                    items.append((f"columbia_au/{sub}/{fn}",
                                  Image.open(os.path.join(d2, fn)).convert("RGB")))
        rng.shuffle(items)
        out["columbia_au"] = items[:n_per]

    # 5) INRIA 无行人图（**与 TruFor/PSCC 基线的评估集对齐**，公平对比必需）
    inria = f"{D}/inria/data_ped/no_pedestrians"
    if os.path.isdir(inria):
        fs = sorted(f for f in os.listdir(inria) if f.lower().endswith(IMG_EXT))
        items = []
        for i in rng.permutation(len(fs))[:n_per]:
            try:
                items.append((f"inria_noped/{fs[i]}",
                              Image.open(os.path.join(inria, fs[i])).convert("RGB")))
            except Exception:
                continue
        out["inria_noped"] = items

    for k, v in out.items():
        for _, im in v:
            im.thumbnail((1024, 1024))
    return out


def torch_load(p):
    import torch
    return torch.load(p, map_location="cpu", weights_only=False)


def collect_pos(n_pos, seed=42):
    import numpy as np
    from PIL import Image
    items = []
    files = sorted(os.listdir(APRICOT_D), key=lambda f: int(f.split("_")[1].split(".")[0]))
    for f in files:
        if len(items) >= n_pos:
            break
        x = torch_load(os.path.join(APRICOT_D, f))
        if x["Mask"].sum() > 0:
            arr = np.clip(x["Image"][0].astype(np.float32) * 255.0, 0, 255).astype(np.uint8)
            im = Image.fromarray(arr).convert("RGB"); im.thumbnail((1024, 1024))
            items.append((f"apricot_patch/{f}", im))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-source", type=int, default=60)
    ap.add_argument("--n-pos", type=int, default=100)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--sources", default=None,
                    help="逗号分隔的负样本源白名单（用于与 SOTA 基线严格对齐评估集），"
                         "如 casia_au,coco_val,inria_noped")
    args = ap.parse_args()

    import numpy as np
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from lora_mllm import QWEN_SNAP, PROMPT

    ckpt = args.ckpt
    if ckpt is None:
        for c in (os.path.join(OUT, "lora_rl_ckpt"), os.path.join(OUT, "lora_patch_detector")):
            if os.path.isdir(c):
                ckpt = c; break
    print(f"[confound] ckpt={ckpt}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        QWEN_SNAP, torch_dtype=torch.float16, local_files_only=True)
    proc = AutoProcessor.from_pretrained(QWEN_SNAP, local_files_only=True)
    model = PeftModel.from_pretrained(model, ckpt).cuda().eval()
    yes_id, no_id = 14004, 8996

    pos = collect_pos(args.n_pos)
    neg = collect_neg(args.n_per_source)
    if args.sources:
        keep = {s.strip() for s in args.sources.split(",") if s.strip()}
        neg = {k: v for k, v in neg.items() if k in keep}
        print(f"[confound] 负样本源白名单过滤 -> {sorted(neg)}")
    print(f"[confound] pos={len(pos)}  neg sources: { {k: len(v) for k, v in neg.items()} }")

    def score(img):
        msg = [{"role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        t = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        enc = {k: (v.cuda() if torch.is_tensor(v) else v)
               for k, v in proc(text=[t], images=[img], return_tensors="pt").items()}
        with torch.no_grad():
            lg = model(**enc).logits[0, -1]
        return float(torch.sigmoid(lg[yes_id] - lg[no_id]))

    t0 = time.time()
    p_scores, p_names = [], []
    for i, (nm, im) in enumerate(pos):
        p_scores.append(score(im)); p_names.append(nm)
        if (i + 1) % 25 == 0:
            print(f"  pos {i+1}/{len(pos)} ({time.time()-t0:.0f}s)", flush=True)

    neg_scores = {}
    for src, items in neg.items():
        ss = []
        for i, (nm, im) in enumerate(items):
            ss.append(score(im))
            if (i + 1) % 25 == 0:
                print(f"  {src} {i+1}/{len(items)}", flush=True)
        neg_scores[src] = ss

    pos_arr = np.array(p_scores)
    print("\n=== 结果 ===")
    print(f"POS APRICOT-patch: n={len(pos_arr)} mean={pos_arr.mean():.4f} "
          f"median={np.median(pos_arr):.4f}")
    per_source = {}
    all_neg = []
    for src, ss in neg_scores.items():
        a = np.array(ss)
        if len(a) == 0:
            per_source[src] = {"n": 0}
            print(f"NEG {src:20s}: n=0 (该来源未收集到图像，跳过)")
            continue
        all_neg.extend(ss)
        # 每个来源的分数分布（跨来源对比是 confound 判据）
        per_source[src] = {"n": len(a), "mean": round(float(a.mean()), 4),
                           "median": round(float(np.median(a)), 4),
                           "p90": round(float(np.percentile(a, 90)), 4)}
        print(f"NEG {src:20s}: n={len(a):4d} mean={a.mean():.4f} median={np.median(a):.4f} "
              f"p90={np.percentile(a,90):.4f}")

    all_neg = np.array(all_neg)
    # 用全局负样本分位数定阈值，报告每来源 FPR（空来源跳过，避免崩溃）
    result = {"ckpt": ckpt, "n_pos": int(len(pos_arr)),
              "pos_stats": {"mean": round(float(pos_arr.mean()), 4),
                            "median": round(float(np.median(pos_arr)), 4)},
              "neg_per_source": per_source,
              "neg_all_mean": round(float(all_neg.mean()), 4) if len(all_neg) else None,
              # 原始逐样本分数（供箱线图 / 分布分析用）
              "pos_names": p_names,
              "pos_scores": [round(float(s), 6) for s in p_scores],
              "neg_scores_by_source": {k: [round(float(s), 6) for s in v]
                                       for k, v in neg_scores.items()}}
    if len(all_neg) == 0:
        raise SystemExit("没有收集到任何负样本，检查路径")

    for fpr_budget in (0.01, 0.05, 0.10, 0.20, 0.33):
        thr = float(np.percentile(all_neg, 100 * (1 - fpr_budget)))
        fprs = {}
        for src, ss in neg_scores.items():
            if len(ss) == 0:
                continue
            a = np.array(ss)
            fprs[src] = round(float((a >= thr).sum() / max(len(a), 1)), 4)
        tpr = round(float((pos_arr >= thr).sum() / max(len(pos_arr), 1)), 4)
        result[f"@target_FPR{int(fpr_budget*100)}pct"] = {"threshold": round(thr, 4),
                                                          "tpr": tpr, "fpr_per_source": fprs}
        print(f"\n目标 FPR={int(fpr_budget*100)}%: 阈值={thr:.4f}  TPR={tpr}")
        for s, v in fprs.items():
            print(f"    FPR[{s:18s}] = {v}")

    out = args.out or os.path.join(OUT, "confound_check.json")
    json.dump(result, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"\n[confound] wrote {out}")


if __name__ == "__main__":
    main()
