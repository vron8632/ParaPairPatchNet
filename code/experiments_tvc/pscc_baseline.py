#!/usr/bin/env python3
"""
SOTA 基线评测 — PSCC-Net（回应"零 SOTA 对比"）
================================================
PSCC-Net (ICCV 2021, "Progressive Spatio-Chromatic Convolution Network") 是通用图像
篡改检测/定位模型，权重由用户在 weights/pscc/ 提供（完整）。作为**攻击无关的取证基线**：
它并非为对抗补丁设计，因此若我们的方法在补丁检测上显著优于它，即构成有效对比；
同时它也能给出定位掩膜，可与我们的定位能力对照。

推理: feat=HRNet(img) -> ClsNet(feat) 给图像级 authentic/forged; SegNet(feat) 给掩膜。

评测: APRICOT(真实补丁, 正) vs 多来源干净图(负)，统一阈值扫 ROC。

用法:
  $PY code/experiments_tvc/pscc_baseline.py --n-pos 100 --n-neg 60
输出: results/tvc/pscc_baseline.json
"""
import argparse, json, os, sys, time
import numpy as np

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS = os.path.dirname(CODE_DIR)
PSCC = os.path.join(WS, "weights", "pscc", "PSCC-Net-main")
OUT = os.path.join(WS, "results", "tvc")
D = "/media/oyp/工作/Projects/042_image_forensic/DuetGuard/data"
APRICOT_D = f"{D}/../mmm-forensicguard/data/apricot/dev/data_mask_v2"
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pos", type=int, default=100)
    ap.add_argument("--n-neg", type=int, default=60)
    ap.add_argument("--resize", type=int, default=512)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from PIL import Image

    sys.path.insert(0, PSCC)
    _CWD0 = os.getcwd()                    # 记录 chdir 前的工作目录（供相对 --out 解析）
    os.chdir(PSCC)                     # test.py 用相对路径 ./checkpoint
    from models.seg_hrnet import get_seg_model
    from models.seg_hrnet_config import get_hrnet_cfg
    from utils.config import get_pscc_args
    from models.NLCDetection import NLCDetection
    from models.detection_head import DetectionHead

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = get_pscc_args()
    FENet = get_seg_model(get_hrnet_cfg()).to(device)
    SegNet = NLCDetection(cfg).to(device)
    ClsNet = DetectionHead(cfg).to(device)
    for net, name in ((FENet, "HRNet"), (SegNet, "NLCDetection"), (ClsNet, "DetectionHead")):
        p = f"./checkpoint/{name}_checkpoint/{name}.pth"
        sd = torch.load(p, map_location=device)
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        net.load_state_dict(sd); net.eval()
        print(f"[pscc] loaded {name}")

    def score(img):
        x = F.interpolate(img.unsqueeze(0).float(), size=(args.resize, args.resize),
                          mode="bilinear", align_corners=False).to(device)
        with torch.no_grad():
            feat = FENet(x)
            logit = ClsNet(feat)
            prob = torch.softmax(logit, 1)[0, 1]        # index1 = forged
        return float(prob)

    # 正样本: APRICOT（真实补丁）
    files = sorted(os.listdir(APRICOT_D), key=lambda f: int(f.split("_")[1].split(".")[0]))
    pos_scores, pos_names = [], []
    for f in files:
        if len(pos_scores) >= args.n_pos:
            break
        x = torch.load(os.path.join(APRICOT_D, f), map_location="cpu", weights_only=False)
        if x["Mask"].sum() <= 0:
            continue
        arr = np.clip(x["Image"][0].astype(np.float32) * 255.0, 0, 255).astype(np.uint8)
        im = Image.fromarray(arr).convert("RGB").resize((args.resize, args.resize))
        t = torch.from_numpy(np.asarray(im)).permute(2, 0, 1) / 255.0
        pos_scores.append(score(t)); pos_names.append(f)
        if len(pos_scores) % 25 == 0:
            print(f"  pos {len(pos_scores)}/{args.n_pos}", flush=True)

    # 负样本: 多来源
    rng = np.random.RandomState(42)
    neg_sets = {}
    for src, d in (("casia_au", f"{D}/casia2/CASIA2.0_revised/Au"),
                   ("coco_val", f"{D}/coco/val2017"),
                   ("inria_noped", f"{D}/inria/data_ped/no_pedestrians")):
        if not os.path.isdir(d):
            continue
        fs = sorted(f for f in os.listdir(d) if f.lower().endswith(IMG_EXT))
        pick = [fs[i] for i in rng.permutation(len(fs))[:args.n_neg]]
        ss = []
        for i, fn in enumerate(pick):
            try:
                im = Image.open(os.path.join(d, fn)).convert("RGB").resize((args.resize, args.resize))
            except Exception:
                continue     # 跳过损坏/不完整文件
            t = torch.from_numpy(np.asarray(im)).permute(2, 0, 1) / 255.0
            ss.append(score(t))
            if (i + 1) % 20 == 0:
                print(f"  {src} {i+1}/{len(pick)}", flush=True)
        neg_sets[src] = ss
        print(f"[pscc] {src}: n={len(ss)} mean={np.mean(ss):.4f}")

    p = np.array(pos_scores)
    rep = {"baseline": "PSCC-Net (ICCV2021) — generic tamper detector, non-patch-specific",
           "n_pos": len(p), "pos_mean": round(float(p.mean()), 4),
           "pos_median": round(float(np.median(p)), 4), "neg_per_source": {}}
    allneg = []
    for src, ss in neg_sets.items():
        a = np.array(ss); allneg.extend(ss)
        rep["neg_per_source"][src] = {"n": len(a), "mean": round(float(a.mean()), 4)}
    allneg = np.array(allneg)
    order = np.argsort(np.concatenate([p, allneg]))
    rk = np.empty(len(order)); rk[order] = np.arange(1, len(order) + 1)
    auc = (rk[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(allneg))
    rep["auc"] = round(float(auc), 4)
    for b in (0.01, 0.05, 0.10, 0.20):
        thr = float(np.percentile(allneg, 100 * (1 - b)))
        tpr = float((p >= thr).sum() / len(p))
        fprs = {s: round(float((np.array(v) >= thr).sum() / max(len(v), 1)), 4)
                for s, v in neg_sets.items()}
        rep[f"@FPR{int(b*100)}pct"] = {"threshold": round(thr, 4), "tpr": round(tpr, 4),
                                       "fpr_per_source": fprs}
        print(f"[pscc] @FPR{int(b*100)}%: thr={thr:.4f} TPR={tpr:.4f} FPR={fprs}")

    out = args.out or os.path.join(OUT, "pscc_baseline.json")
    if not os.path.isabs(out):
        out = os.path.join(_CWD0, out)          # chdir 后相对路径失效 → 用原始工作目录解析
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rep, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"[pscc] wrote {out}")


if __name__ == "__main__":
    main()
