#!/usr/bin/env python3
"""
SOTA 分数导出（到低误报协议所需的 21k 负样本评测集）

评测集（与 lowfpr_protocol.py 完全一致）：
  正样本 138（unified_eval_set/pos）
  负样本 21,000 = APRICOT 4 源各 250（casia_au / coco_val / inria_noped / columbia_au）
                + COCO-train2017 20,000（只评测、不入训练）

对每个 SOTA 只导出**图像级分数**到 npz，指标统一由 lowfpr_report.py 计算
（pAUC / 交叉拟合 TPR@FPR / 分源 FPR），保证所有方法同一口径。

方法：
  nutnet13 / nutnet26  NutNet (CCS'24) 官方 AE 权重（n_13 = 32 块×13px，n_26 = 16 块×26px）
  yolo26               YOLO26n-cls 微调权重（imgsz=512）
  trufor               TruFor (CVPR'23) 官方 phase-3 检测头
  pscc                 PSCC-Net (ICCV'21)
（AdvPatchXAI 由 eval_advpatchxai.py --emit-npz 单独导出，避免重复实现其组装逻辑）

用法：
  python sota_score_dump.py --methods nutnet13,nutnet26,yolo26
  python sota_score_dump.py --methods trufor,pscc
输出：results/tvc/sota_scores_<method>.npz
"""
import os, sys, json, argparse, glob
os.environ.setdefault("HF_HOME", "/home/oyp/.cache/huggingface")
WS = "/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard"
TVC = f"{WS}/results/tvc"; EVAL = f"{TVC}/unified_eval_set"
NEGS_IDX = "/home/oyp/fastdata/apde_negs/index.json"
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")

import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--methods", default="nutnet13,nutnet26,yolo26")
ap.add_argument("--n-coco", type=int, default=20000)
ap.add_argument("--resize", type=int, default=512)
ap.add_argument("--yolo-weights", default=f"{TVC}/yolo26_cls/weights/best.pt",
                help="YOLO26 分类权重（可指向 mix 重训版本）")
a = ap.parse_args()

S3 = ["casia_au", "coco_val", "inria_noped"]


def build_rows():
    rows = [{"path": f"{EVAL}/pos/{f}", "label": 1, "src": "apricot_pos"}
            for f in sorted(os.listdir(f"{EVAL}/pos"))]
    rows += [{"path": f"{EVAL}/{s}/{f}", "label": 0, "src": s}
             for s in S3 + ["columbia_au"] for f in sorted(os.listdir(f"{EVAL}/{s}"))]
    rows += [{"path": r["path"], "label": 0, "src": "coco_train"}
             for r in json.load(open(NEGS_IDX))[:a.n_coco]]
    return rows


def save(name, scores, rows):
    out = f"{TVC}/sota_scores_{name}.npz"
    np.savez_compressed(out, scores=np.asarray(scores, np.float32),
                        src=np.array([r["src"] for r in rows]),
                        label=np.array([r["label"] for r in rows]),
                        path=np.array([r["path"] for r in rows]))
    s = np.asarray(scores, float)
    pos = s[np.array([r["label"] for r in rows]) == 1]
    print(f"  → {out}  (pos mean {pos.mean():.4f} / neg mean {s[len(pos):].mean() if False else s[np.array([r['label'] for r in rows])==0].mean():.4f})", flush=True)


# ───────────────────── NutNet ─────────────────────
def run_nutnet(rows, weight):
    import torch, torch.nn as nn
    NUT = f"{WS}/code/third_party/NutNet"

    class AutoEncoder32(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 8, 2, 1, 1), nn.ReLU(), nn.Conv2d(8, 16, 2, 2, 1), nn.ReLU(),
                nn.Conv2d(16, 32, 2, 2, 1), nn.ReLU())
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(32, 16, 2, 2, 1), nn.ReLU(),
                nn.ConvTranspose2d(16, 8, 2, 2, 1), nn.ReLU(),
                nn.ConvTranspose2d(8, 3, 2, 1, 1), nn.Tanh())
        def forward(self, x): return self.decoder(self.encoder(x))

    class AutoEncoder16(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 8, 2, 2, 1), nn.ReLU(), nn.Conv2d(8, 16, 2, 2, 1), nn.ReLU(),
                nn.Conv2d(16, 32, 2, 2, 1), nn.ReLU())
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(32, 16, 2, 2, 1), nn.ReLU(),
                nn.ConvTranspose2d(16, 8, 2, 2, 1), nn.ReLU(),
                nn.ConvTranspose2d(8, 3, 2, 2, 1), nn.Tanh())
        def forward(self, x): return self.decoder(self.encoder(x))

    ae = (AutoEncoder32() if weight == "n_13" else AutoEncoder16()).cuda().eval()
    ae.load_state_dict(torch.load(f"{NUT}/ae_weights/{weight}.pth", map_location="cuda"))
    INPUT = 416
    BOX = 32 if weight == "n_13" else 16
    BL = INPUT // BOX
    out = []
    with torch.no_grad():
        for i, r in enumerate(rows):
            try:
                im = Image.open(r["path"]).convert("RGB").resize((INPUT, INPUT), Image.BILINEAR)
            except Exception:
                out.append(0.0); continue
            x = torch.from_numpy(np.asarray(im).copy()).float().permute(2, 0, 1).unsqueeze(0).cuda() / 255.0
            x = x * 2 - 1
            blk = x.unfold(2, BL, BL).unfold(3, BL, BL)
            blk = blk.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, 3, BL, BL)
            mse = torch.mean((ae(blk) - blk) ** 2, dim=(1, 2, 3))
            out.append(float(mse.max()))
            if (i + 1) % 3000 == 0: print(f"    {weight} {i+1}/{len(rows)}", flush=True)
    return out


# ───────────────────── YOLO26 ─────────────────────
def run_yolo26(rows):
    from ultralytics import YOLO
    m = YOLO(a.yolo_weights)
    pi = next(i for i, nm in m.names.items() if "patch" in nm.lower())
    out = []
    for i, r in enumerate(rows):
        try:
            arr = np.asarray(Image.open(r["path"]).convert("RGB"))
            res = m.predict(arr, imgsz=512, verbose=False)[0]
            out.append(float(res.probs.data.tolist()[pi]))
        except Exception:
            out.append(0.0)
        if (i + 1) % 3000 == 0: print(f"    yolo26 {i+1}/{len(rows)}", flush=True)
    return out


# ───────────────────── TruFor ─────────────────────
def run_trufor(rows, resize):
    import torch
    import torch.nn.functional as F
    W = os.path.join(WS, "weights", "TruFor")
    for req in ("mit_b2.pth", "noiseprint.pth"):
        if not os.path.exists(os.path.join(W, req)):
            raise SystemExit(f"缺少 {req}")
    ck = os.path.join(W, "trufor_phase3_best.pth.tar")
    if not os.path.exists(ck):
        raise SystemExit(f"缺少 {ck}")
    _ol = torch.load
    torch.load = lambda *ar, **kw: _ol(*ar, **{**kw, "weights_only": kw.get("weights_only", False)})
    from IMDLBenCo.model_zoo.trufor.trufor import Trufor
    import IMDLBenCo as _ib
    cfg = os.path.join(os.path.dirname(_ib.__file__), "statics", "model_zoo", "configs", "trufor.yaml")
    model = Trufor(phase=2, np_pretrain_weights=os.path.join(W, "noiseprint.pth"),
                   mit_b2_pretrain_weights=os.path.join(W, "mit_b2.pth"), config_path=cfg)
    sd = _ol(ck, map_location="cpu", weights_only=False)
    sd = sd["state_dict"] if isinstance(sd, dict) and "state_dict" in sd else sd
    sd = {(k if k.startswith("model.") else "model." + k): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model = model.to("cuda").eval()

    out = []
    with torch.no_grad():
        for i, r in enumerate(rows):
            try:
                im = Image.open(r["path"]).convert("RGB").resize((resize, resize), Image.BILINEAR)
                t = torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float().div(255).unsqueeze(0).cuda()
                o = model.model(t)
                det = o[2] if isinstance(o, (list, tuple)) and len(o) > 2 else o[-1]
                out.append(float(torch.sigmoid(det).flatten()[0]))
            except Exception:
                out.append(0.0)
            if (i + 1) % 2000 == 0: print(f"    trufor {i+1}/{len(rows)}", flush=True)
    return out


# ───────────────────── PSCC-Net ─────────────────────
def run_pscc(rows, resize):
    import torch
    import torch.nn.functional as F
    PSCC = os.path.join(WS, "weights", "pscc", "PSCC-Net-main")
    sys.path.insert(0, PSCC)
    cwd = os.getcwd(); os.chdir(PSCC)
    try:
        from models.seg_hrnet import get_seg_model
        from models.seg_hrnet_config import get_hrnet_cfg
        from utils.config import get_pscc_args
        from models.NLCDetection import NLCDetection
        from models.detection_head import DetectionHead
        dev = "cuda"
        c = get_pscc_args()
        FENet = get_seg_model(get_hrnet_cfg()).to(dev)
        SegNet = NLCDetection(c).to(dev)
        ClsNet = DetectionHead(c).to(dev)
        for net, name in ((FENet, "HRNet"), (SegNet, "NLCDetection"), (ClsNet, "DetectionHead")):
            sd = torch.load(f"./checkpoint/{name}_checkpoint/{name}.pth", map_location=dev)
            sd = {k.replace("module.", ""): v for k, v in sd.items()}
            net.load_state_dict(sd); net.eval()
    finally:
        pass
    out = []
    with torch.no_grad():
        for i, r in enumerate(rows):
            try:
                im = Image.open(r["path"]).convert("RGB").resize((resize, resize), Image.BILINEAR)
                t = torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float().div(255).unsqueeze(0).to(dev)
                prob = torch.softmax(ClsNet(FENet(t)), 1)[0, 1]
                out.append(float(prob))
            except Exception:
                out.append(0.0)
            if (i + 1) % 2000 == 0: print(f"    pscc {i+1}/{len(rows)}", flush=True)
    os.chdir(cwd)
    return out


def main():
    rows = build_rows()
    print(f"评测集：{len(rows)}（pos {sum(r['label']==1 for r in rows)}，neg {sum(r['label']==0 for r in rows)}）", flush=True)
    for m in [x.strip() for x in a.methods.split(",") if x.strip()]:
        print(f"\n───── {m} ─────", flush=True)
        if m == "nutnet13":
            save(m, run_nutnet(rows, "n_13"), rows)
        elif m == "nutnet26":
            save(m, run_nutnet(rows, "n_26"), rows)
        elif m == "yolo26":
            save(m, run_yolo26(rows), rows)
        elif m == "trufor":
            save(m, run_trufor(rows, a.resize), rows)
        elif m == "pscc":
            save(m, run_pscc(rows, a.resize), rows)
        else:
            print(f"  未知方法 {m}", flush=True)


if __name__ == "__main__":
    main()
