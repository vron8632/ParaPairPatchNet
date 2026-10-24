#!/usr/bin/env python3
"""
YOLO26 微调 baseline（分类模式）—— 回应"为什么不用 YOLO26"
================================================================
任务设定：与主表一致的图像级二分类（clean vs patch）。
训练数据：构建集 A（CASIA 照片 + 合成物理/数字补丁 + 干净图，lora_data/train 1500 张）
评估数据：统一 APRICOT 协议（results/tvc/baseline_imgs：APRICOT 100 正 + 3 源各 60 负）
用法: $PY code/experiments_tvc/yolo26_baseline.py
输出: results/tvc/yolo26.json
"""
import os, json, shutil
import numpy as np

WS = "/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard"
TVC = f"{WS}/results/tvc"
BASE_IMG = f"{TVC}/baseline_imgs"
LORA_DATA = f"{TVC}/lora_data"
S3 = ["casia_au", "coco_val", "inria_noped"]


def build_imagedir(split_src, dst):
    """index.json → ImageFolder(dst/{clean,patch}/xxx.jpg)"""
    import shutil
    rows = json.load(open(f"{LORA_DATA}/index.json"))
    n = 0
    for r in rows:
        if r["split"] != split_src:
            continue
        sub = "patch" if r["label"] == 1 else "clean"
        d = os.path.join(dst, sub)
        os.makedirs(d, exist_ok=True)
        shutil.copy(r["path"], os.path.join(d, os.path.basename(r["path"])))
        n += 1
    return n


def main():
    from ultralytics import YOLO
    import torch
    from PIL import Image
    import torchvision.transforms as T

    ds_dir = f"{TVC}/yolo26_ds"
    print("=== 构建 ImageFolder（train/val 9:1）===", flush=True)
    n = build_imagedir("train", f"{ds_dir}/train_full")
    print(f"  train_full: {n}", flush=True)
    # 分出 10% 作 val（分类训练需要 val 目录）
    import random
    random.seed(0)
    for sub in ("clean", "patch"):
        src = f"{ds_dir}/train_full/{sub}"
        dst_tr = f"{ds_dir}/train/{sub}"; dst_va = f"{ds_dir}/val/{sub}"
        os.makedirs(dst_tr, exist_ok=True); os.makedirs(dst_va, exist_ok=True)
        fs = sorted(os.listdir(src)); random.shuffle(fs)
        k = max(1, len(fs) // 10)
        for i, f in enumerate(fs):
            shutil.copy(os.path.join(src, f), os.path.join(dst_va if i < k else dst_tr, f))
    shutil.rmtree(f"{ds_dir}/train_full")

    print("=== 微调 YOLO26n-cls（30 epochs, imgsz=512）===", flush=True)
    m = YOLO("yolo26n-cls.pt")
    m.train(data=ds_dir, epochs=30, imgsz=512, batch=64, lr0=0.01,
            device=0, workers=4, project=TVC, name="yolo26_cls", exist_ok=True,
            plots=False, verbose=True)

    # ── 统一评估（与主表同一负样本池）──
    m = YOLO(f"{TVC}/yolo26_cls/weights/best.pt")
    tf = T.Compose([T.ToTensor()])
    def score_dir(d):
        files = sorted(os.listdir(d))
        scores = []
        for f in files:
            im = Image.open(os.path.join(d, f)).convert("RGB")
            im.thumbnail((512, 512))
            arr = np.asarray(im).copy()
            with torch.no_grad():
                r = m.predict(arr, imgsz=512, verbose=False)[0]
                names = r.names   # {'0': 'clean', '1': 'patch'} (字母序)
                probs = r.probs.data.tolist()
                idx = 1 if "patch" in names[1].lower() or names[1] == "1" else 0
                # 按名字找 patch 类概率
                patch_idx = next(i for i, nm in names.items() if "patch" in nm.lower())
                scores.append(probs[patch_idx])
        return scores

    pos = score_dir(f"{BASE_IMG}/pos")
    neg = {s: score_dir(f"{BASE_IMG}/{s}") for s in S3}
    p = np.array(pos); nall = np.concatenate([np.array(v) for v in neg.values()])
    res = {"model": "YOLO26n-cls (fine-tuned)", "n_pos": len(p), "n_neg": len(nall),
           "pos_mean": round(float(p.mean()), 4), "neg_mean": round(float(nall.mean()), 4),
           "neg_scores_by_source": {k: [round(float(x), 4) for x in v] for k, v in neg.items()},
           "pos_scores": [round(float(x), 4) for x in pos]}
    print(f"\npos mean={p.mean():.4f}  neg mean={nall.mean():.4f}")
    for a in (0.01, 0.05, 0.10, 0.20, 0.33):
        thr = float(np.percentile(nall, 100 * (1 - a)))
        tpr = float((p >= thr).mean())
        res[f"TPR@{int(a*100)}pct"] = {"threshold": round(thr, 4), "tpr": round(tpr, 4)}
        print(f"  目标 FPR={int(a*100):2d}%: τ={thr:.4f} TPR={tpr:.4f}")
    json.dump(res, open(f"{TVC}/yolo26.json", "w"), indent=1, ensure_ascii=False)
    print(f"[yolo26] wrote {TVC}/yolo26.json")


if __name__ == "__main__":
    main()
