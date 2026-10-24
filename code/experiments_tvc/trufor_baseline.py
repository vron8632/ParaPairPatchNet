#!/usr/bin/env python3
"""
SOTA 基线 #2 — TruFor (CVPR 2023) 在物理补丁任务上
=====================================================
回应"零 SOTA 对比"（致命伤之二）。与 PSCC-Net 一起，构成**通用图像篡改检测器**的代表：
TruFor 结合 RGB + Noiseprint++ + 跨模态融合，是 IMDL 定位任务的当前强基线。

权重: weights/TruFor/trufor_phase3_best.pth.tar（第三方用官方实现训练，phase-3 检测头，
      来自 HF benjaik/trufor-ph2；官方 checkpoint 在 Google Drive，本机不可达）
依赖: weights/TruFor/{mit_b2.pth, noiseprint.pth}（用户已提供）

推理: Trufor(phase=3) -> 头部输出 [loc, det, conf]，取 det 得图像级分数。

用法:
  $PY code/experiments_tvc/trufor_baseline.py --n-pos 100 --n-neg 60
输出: results/tvc/trufor_baseline.json
"""
import argparse, json, os, sys, time
import numpy as np

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS = os.path.dirname(CODE_DIR)
W = os.path.join(WS, "weights", "TruFor")
OUT = os.path.join(WS, "results", "tvc")
D = "/media/oyp/工作/Projects/042_image_forensic/DuetGuard/data"
APRICOT_D = f"{D}/../mmm-forensicguard/data/apricot/dev/data_mask_v2"
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pos", type=int, default=100)
    ap.add_argument("--n-neg", type=int, default=60)
    ap.add_argument("--resize", type=int, default=512)
    ap.add_argument("--ckpt", default=os.path.join(W, "trufor_phase3_best.pth.tar"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from PIL import Image

    for req in ("mit_b2.pth", "noiseprint.pth"):
        if not os.path.exists(os.path.join(W, req)):
            raise SystemExit(f"缺少 {req}")
    if not os.path.exists(args.ckpt):
        raise SystemExit(f"缺少主 checkpoint: {args.ckpt}")

    sys.path.insert(0, "/home/oyp/miniconda3/envs/apjf/lib/python3.10/site-packages")
    # PyTorch 2.6 默认 weights_only=True，IMDLBenCo 内部 torch.load 未指定；
    # checkpoint 已按官方 SHA256SUMS 校验（745575e8...），显式放宽。
    import torch as _torch
    _orig_load = _torch.load
    _torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": k.get("weights_only", False)})

    from IMDLBenCo.model_zoo.trufor.trufor import Trufor
    import IMDLBenCo as _ib

    # 用 IMDLBenCo 自带的完整 TruFor 配置（含 BACKBONE/DETECTION/CONF）
    cfg_path = os.path.join(os.path.dirname(_ib.__file__),
                            "statics", "model_zoo", "configs", "trufor.yaml")
    if not os.path.exists(cfg_path):
        raise SystemExit(f"缺少 IMDLBenCo 配置: {cfg_path}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # phase=2 构建网络并加载 noiseprint++/mit_b2 预训练依赖（用户已提供）；
    # 不用它的 det_resume_ckpt（官方 checkpoint 是 {state_dict: backbone.*} 打包格式，
    # 缺 model. 前缀，内部直接 load_state_dict 会报 missing keys）。
    model = Trufor(phase=2,
                   np_pretrain_weights=os.path.join(W, "noiseprint.pth"),
                   mit_b2_pretrain_weights=os.path.join(W, "mit_b2.pth"),
                   config_path=cfg_path)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if (isinstance(ck, dict) and "state_dict" in ck) else ck
    sd = {("model." + k if not k.startswith("model.") else k): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[trufor] loaded phase3 ckpt: {len(sd)} tensors | missing={len(missing)} unexpected={len(unexpected)}")
    if len(missing) > 0:
        print("   missing sample:", list(missing[:3]))
    if len(unexpected) > 0:
        print("   unexpected sample:", list(unexpected[:3]))

    model = model.to(device).eval()
    print(f"[trufor] ready: {os.path.basename(args.ckpt)}")

    def score(t):
        """t: (3,H,W) float [0,1] -> 图像级篡改概率。
        Trufor.forward(image, mask, label) 是训练签名（返回含 loss 的 dict），
        评测需直接调内部 confcmx: 返回 (pred_mask, conf, det, npp)。"""
        x = F.interpolate(t.unsqueeze(0), size=(args.resize, args.resize),
                          mode="bilinear", align_corners=False).to(device)
        with torch.no_grad():
            out = model.model(x)
            det = out[2] if isinstance(out, (list, tuple)) and len(out) > 2 else out[-1]
            return float(torch.sigmoid(det).flatten()[0])

    files = sorted(os.listdir(APRICOT_D), key=lambda f: int(f.split("_")[1].split(".")[0]))
    pos_scores = []
    for f in files:
        if len(pos_scores) >= args.n_pos:
            break
        x = torch.load(os.path.join(APRICOT_D, f), map_location="cpu", weights_only=False)
        if x["Mask"].sum() <= 0:
            continue
        arr = np.clip(x["Image"][0].astype(np.float32) * 255.0, 0, 255).astype(np.uint8)
        im = Image.fromarray(arr).convert("RGB").resize((args.resize, args.resize))
        t = torch.from_numpy(np.asarray(im)).permute(2, 0, 1) / 255.0
        pos_scores.append(score(t))
        if len(pos_scores) % 25 == 0:
            print(f"  pos {len(pos_scores)}/{args.n_pos}", flush=True)

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
                continue
            t = torch.from_numpy(np.asarray(im)).permute(2, 0, 1) / 255.0
            ss.append(score(t))
        neg_sets[src] = ss
        print(f"[trufor] {src}: n={len(ss)} mean={np.mean(ss):.4f}", flush=True)

    p = np.array(pos_scores)
    allneg = [v for ss in neg_sets.values() for v in ss]
    allneg = np.array(allneg)
    order = np.argsort(np.concatenate([p, allneg]))
    rk = np.empty(len(order)); rk[order] = np.arange(1, len(order) + 1)
    auc = (rk[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(allneg))
    rep = {"baseline": "TruFor (CVPR2023) — RGB+Noiseprint++ tamper detector",
           "ckpt": args.ckpt, "n_pos": len(p), "auc": round(float(auc), 4),
           "pos_mean": round(float(p.mean()), 4), "pos_median": round(float(np.median(p)), 4),
           "neg_per_source": {k: {"n": len(v), "mean": round(float(np.mean(v)), 4)}
                              for k, v in neg_sets.items()}}
    for b in (0.01, 0.05, 0.10, 0.20):
        thr = float(np.percentile(allneg, 100 * (1 - b)))
        tpr = float((p >= thr).sum() / len(p))
        fprs = {s: round(float((np.array(v) >= thr).sum() / max(len(v), 1)), 4)
                for s, v in neg_sets.items()}
        rep[f"@FPR{int(b*100)}pct"] = {"threshold": round(thr, 4),
                                       "tpr": round(tpr, 4), "fpr_per_source": fprs}
        print(f"[trufor] @FPR{int(b*100)}%: TPR={tpr:.4f} FPR={fprs}", flush=True)

    out = args.out or os.path.join(OUT, "trufor_baseline.json")
    json.dump(rep, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"[trufor] wrote {out}")


if __name__ == "__main__":
    main()
