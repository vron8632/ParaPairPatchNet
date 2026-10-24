#!/usr/bin/env python3
"""
重采样对照实验：低误报评测中的「来源/重采样偏置」（provenance bias）

问题：APRICOT 正样本原图 4608×2304，导出到评测集时被缩到长边 1024（约 9 倍下采样）；
      而负样本（casia / inria / columbia / COCO）基本是原生尺寸。
      于是"正样本"带有一段独特的高频/插值历史 → 对高频敏感的检测器（PSCC/TruFor 这类
      篡改检测器）可能把"重采样历史"当成"被篡改"，得到虚高的低误报指标。

实测证据（同一张图，仅改输入尺寸）：
  data_0.png  原尺寸 512×1024 → PSCC 0.0237 ; resize 512×512 → PSCC 0.8095
  data_1.png  原尺寸 768×1024 → PSCC 0.0079 ; resize 512×512 → PSCC 0.9204

本实验：把**负样本**按与正样本相同的方式做尺度归一（长边→1024），重测各方法，
看它们的"优势"是否随重采样历史消失。

方法：pscc（最敏感）、yolo26、ours(sem/moe_att 的分数用已保存的 npz 无法重采样，
      故本脚本只跑需要前向的 SOTA；我们的模型用 lowfpr_protocol 另跑对照）

输出：results/tvc/resample_control.json
"""
import os, sys, json, argparse
os.environ.setdefault("HF_HOME", "/home/oyp/.cache/huggingface")
WS = "/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard"
TVC = f"{WS}/results/tvc"; EVAL = f"{TVC}/unified_eval_set"
NEGS_IDX = "/home/oyp/fastdata/apde_negs/index.json"

import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--methods", default="pscc,yolo26")
ap.add_argument("--n-coco", type=int, default=2000)
ap.add_argument("--n-src", type=int, default=250)
ap.add_argument("--resize", type=int, default=512)
ap.add_argument("--long-side", type=int, default=1024)
ap.add_argument("--out", default=f"{TVC}/resample_control.json")
a = ap.parse_args()
S3 = ["casia_au", "coco_val", "inria_noped"]


def build_rows():
    pos = [{"path": f"{EVAL}/pos/{f}", "src": "apricot_pos"} for f in sorted(os.listdir(f"{EVAL}/pos"))]
    neg = [{"path": f"{EVAL}/{s}/{f}", "src": s}
           for s in S3 + ["columbia_au"] for f in sorted(os.listdir(f"{EVAL}/{s}"))[:a.n_src]]
    coco = json.load(open(NEGS_IDX))[:a.n_coco]
    neg += [{"path": r["path"], "src": "coco_train"} for r in coco]
    return pos, neg


def to_long_side(im, L):
    w, h = im.size
    s = L / max(w, h)
    return im if abs(s - 1) < 1e-6 else im.resize((max(1, int(round(w * s))), max(1, int(round(h * s)))), Image.BILINEAR)


def auc(sp, sn):
    from sklearn.metrics import roc_auc_score
    y = np.r_[np.ones(len(sp)), np.zeros(len(sn))]
    return float(roc_auc_score(y, np.r_[sp, sn]))


def pauc1(sp, sn, npts=4000):
    thr = np.unique(np.r_[sp, sn]); thr = np.sort(thr)[::max(1, len(thr)//npts)]
    fpr = np.array([(sn >= t).mean() for t in thr]); tpr = np.array([(sp >= t).mean() for t in thr])
    o = np.argsort(fpr); fpr, tpr = fpr[o], tpr[o]
    m = fpr <= 0.01
    return float(np.trapz(tpr[m], fpr[m]) / fpr[m][-1]) if m.sum() > 2 else 0.0


def crossfit1(sp, sn, folds=10):
    rng = np.random.default_rng(0); parts = np.array_split(rng.permutation(len(sn)), folds)
    tp = []
    for k in range(folds):
        tr = np.concatenate([parts[j] for j in range(folds) if j != k])
        thr = float(np.percentile(sn[tr], 99))
        tp.append(float((sp >= thr).mean()))
    return round(float(np.mean(tp)), 4)


# ── PSCC ──
def load_pscc():
    import torch
    PSCC = os.path.join(WS, "weights", "pscc", "PSCC-Net-main")
    sys.path.insert(0, PSCC); os.chdir(PSCC)
    from models.seg_hrnet import get_seg_model
    from models.seg_hrnet_config import get_hrnet_cfg
    from utils.config import get_pscc_args
    from models.NLCDetection import NLCDetection
    from models.detection_head import DetectionHead
    dev = "cuda"; c = get_pscc_args()
    FENet = get_seg_model(get_hrnet_cfg()).to(dev)
    SegNet = NLCDetection(c).to(dev); ClsNet = DetectionHead(c).to(dev)
    for net, name in ((FENet, "HRNet"), (SegNet, "NLCDetection"), (ClsNet, "DetectionHead")):
        sd = torch.load(f"./checkpoint/{name}_checkpoint/{name}.pth", map_location=dev)
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        net.load_state_dict(sd); net.eval()
    def sc(im):
        import torch.nn.functional as F
        t = torch.from_numpy(np.asarray(im.convert("RGB")).copy()).permute(2, 0, 1).float().div(255).unsqueeze(0).to(dev)
        t = F.interpolate(t, size=(a.resize, a.resize), mode="bilinear", align_corners=False)
        with torch.no_grad():
            return float(torch.softmax(ClsNet(FENet(t)), 1)[0, 1])
    return sc


def main():
    import torch
    pos, neg = build_rows()
    print(f"pos {len(pos)} / neg {len(neg)}", flush=True)
    res = {}
    for m in [x.strip() for x in a.methods.split(",") if x.strip()]:
        print(f"\n───── {m} ─────", flush=True)
        if m == "pscc":
            sc = load_pscc()
        elif m == "yolo26":
            from ultralytics import YOLO
            y = YOLO(f"{TVC}/yolo26_cls/weights/best.pt")
            pi = next(i for i, nm in y.names.items() if "patch" in nm.lower())
            def sc(im):
                r = y.predict(np.asarray(im.convert("RGB")), imgsz=512, verbose=False)[0]
                return float(r.probs.data.tolist()[pi])
        else:
            print("  skip", flush=True); continue

        def run(rows, mode):
            out = []
            for i, r in enumerate(rows):
                try:
                    im = Image.open(r["path"]).convert("RGB")
                    if mode == "long1024":
                        im = to_long_side(im, a.long_side)
                    out.append(sc(im))
                except Exception:
                    out.append(0.0)
                if (i + 1) % 500 == 0: print(f"    {mode} {i+1}/{len(rows)}", flush=True)
            return np.array(out)

        sp = run(pos, "native")
        entry = {"pos_mean": round(float(sp.mean()), 4)}
        for mode in ("native", "long1024"):
            sn = run(neg, mode)
            entry[mode] = {"neg_mean": round(float(sn.mean()), 4),
                           "auc": round(auc(sp, sn), 4),
                           "pAUC@1pct": round(pauc1(sp, sn), 4),
                           "TPR@1pct_crossfit": crossfit1(sp, sn),
                           "neg_mean_by_src": {s: round(float(sn[np.array([r["src"] for r in neg]) == s].mean()), 4)
                                               for s in sorted({r["src"] for r in neg})}}
            print(f"  [{mode:9s}] neg_mean={entry[mode]['neg_mean']} AUC={entry[mode]['auc']} "
                  f"pAUC@1%={entry[mode]['pAUC@1pct']} TPR@1%={entry[mode]['TPR@1pct_crossfit']}", flush=True)
            if m == "pscc":
                torch.cuda.empty_cache()
        res[m] = entry
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"\n→ {a.out}")


if __name__ == "__main__":
    main()
