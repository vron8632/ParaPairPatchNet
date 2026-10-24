#!/usr/bin/env python3
"""
统一协议基线评测（用户决策：统一负样本集 1000 + 统一阈值规则 + 统一正样本 138）
================================================================================
各基线按其**原生输入要求**处理（TruFor/PSCC 512×512、NutNet 416×416、
轻量 CNN 448 全图、YOLO26 512），但共用同一套正/负样本与同一阈值规则，
从而在"统一协议"下可比。
输出: results/tvc/unified_baselines.json
用法: $PY unified_baselines_1024.py [--methods nutnet,trufor,pscc,light]
"""
import os, sys, json, argparse, math
os.environ.setdefault("HF_HOME", "/home/oyp/.cache/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
from PIL import Image

WS = "/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard"
TVC = f"{WS}/results/tvc"
BASE = f"{TVC}/baseline_imgs"
sys.path.insert(0, f"{WS}/code/experiments_tvc")
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


def summarize(name, pos, neg, res):
    p = np.asarray(pos)
    row = {"n_pos": len(p), "pos_mean": round(float(p.mean()), 4),
           "neg_mean": {k: round(float(np.mean(v)), 4) for k, v in neg.items()}, "tpr": {}}
    for scope, keys in (("3src", S3), ("4src", S4)):
        n_all = np.concatenate([np.asarray(neg[k]) for k in keys if k in neg])
        rr = {}
        for a in (0.01, 0.05, 0.10, 0.20):
            thr = float(np.percentile(n_all, 100*(1-a))); k_ = int((p >= thr).sum())
            rr[f"@{int(a*100)}pct"] = {"tpr": round(k_/len(p), 4), "wilson": wilson(k_, len(p)),
                                       "thr": round(thr, 4)}
        row["tpr"][scope] = rr
    res[name] = row
    print(f"  [{name}] 3src: " + "  ".join(f"{a}={row['tpr']['3src'][f'@{a}pct']['tpr']:.3f}" for a in (1,5,10,20)), flush=True)
    print(f"  [{name}] 4src: " + "  ".join(f"{a}={row['tpr']['4src'][f'@{a}pct']['tpr']:.3f}" for a in (1,5,10,20)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="nutnet,trufor,pscc,light")
    ap.add_argument("--cnn-size", type=int, default=448, help="轻量 CNN 输入长边（448=其训练尺寸；1024=与主模型统一）")
    ap.add_argument("--n-per-source", type=int, default=250)
    ap.add_argument("--out", default=f"{TVC}/unified_baselines.json")
    args = ap.parse_args()

    import torch
    from PIL import Image
    # 统一采集（复用 unified_main_1024 的 collect_neg：authentic columbia）
    from unified_main_1024 import collect_neg
    print("=== 统一采集 ===", flush=True)
    neg = collect_neg(args.n_per_source)
    pos_imgs = [Image.open(f"{BASE}/pos_all/{f}").convert("RGB")
                for f in sorted(os.listdir(f"{BASE}/pos_all"))]
    print(f"  正样本 {len(pos_imgs)}", flush=True)
    dev = "cuda"
    res = {"protocol": {"n_per_source": args.n_per_source, "n_pos": len(pos_imgs),
                        "neg_total": sum(len(v) for v in neg.values())}}
    methods = set(args.methods.split(","))

    # ── NutNet（416×416，官方 AE 权重）──
    if "nutnet" in methods:
        try:
            import torch.nn as nn
            import torch.nn.functional as F
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from nutnet_baseline import build_ae32
            AE = os.path.join(WS, "code/third_party/NutNet/ae_weights/n_13.pth")
            if not os.path.exists(AE):
                import glob
                cand = glob.glob(os.path.join(WS, "code/**/n_13.pth"), recursive=True)
                AE = cand[0] if cand else AE
            ae = build_ae32().to(dev); ae.load_state_dict(torch.load(AE, map_location=dev)); ae.eval()
            inp, bn = 416, 32; bl = inp // bn
            def score_nn(im):
                x = im.convert("RGB").resize((inp, inp), Image.BILINEAR)
                t = torch.from_numpy(np.asarray(x).copy()).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                t = (t * 2 - 1).to(dev)
                with torch.no_grad():
                    blk = t.unfold(2, bl, bl).unfold(3, bl, bl)
                    blk = blk.permute(0, 2, 3, 4, 5, 1).contiguous().view(-1, bl, bl, 3).permute(0, 3, 1, 2)
                    out = ae(blk)
                    per_block = torch.mean((out - blk) ** 2, dim=(1, 2, 3))
                return float(per_block.max())
            print("[nutnet] 评测中…", flush=True)
            p = [score_nn(im) for im in pos_imgs]
            ns = {k: [score_nn(im) for _, im in v] for k, v in neg.items()}
            summarize("NutNet (CCS'24)", p, ns, res)
            del ae; torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [nutnet] 失败: {type(e).__name__}: {e}", flush=True)

    # ── 轻量 CNN（448 全图）+ YOLO26（512）──
    if "light" in methods:
        try:
            import torch.nn as nn
            from torchvision import transforms, models
            norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            def load_cnn(ck, name):
                if name == "resnet18":
                    m = models.resnet18(weights=None); m.fc = nn.Linear(m.fc.in_features, 2)
                else:
                    m = models.shufflenet_v2_x1_0(weights=None); m.fc = nn.Linear(m.fc.in_features, 2)
                m.load_state_dict(torch.load(ck, map_location=dev)); return m.to(dev).eval()
            CNN_SIZE = args.cnn_size
            def score_cnn(m, im, size=None):
                size = size or CNN_SIZE
                im2 = _fit_long_side(im, size)
                cv = Image.new("RGB", (size, size), (127, 127, 127))
                cv.paste(im2, ((size - im2.width)//2, (size - im2.height)//2))
                x = norm(transforms.ToTensor()(cv)).unsqueeze(0).to(dev)
                with torch.no_grad():
                    return float(torch.softmax(m(x), 1)[0, 1])
            for name in ("resnet18", "shufflenetv2"):
                ck = f"{TVC}/{name}_fair.pt"
                if not os.path.exists(ck): print(f"  [{name}] ckpt 缺失"); continue
                m = load_cnn(ck, name)
                print(f"[{name}] 评测中…", flush=True)
                p = [score_cnn(m, im) for im in pos_imgs]
                ns = {k: [score_cnn(m, im) for _, im in v] for k, v in neg.items()}
                summarize(f"{name} (fine-tuned, {CNN_SIZE}px)", p, ns, res)
                del m; torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [light] 失败: {type(e).__name__}: {e}", flush=True)

    # ── TruFor（512×512 原生）──
    if "trufor" in methods:
        try:
            import torch.nn.functional as F
            ckpt = os.path.join(WS, "trufor_phase3_best.pth.tar")
            if os.path.exists(ckpt):
                import trufor_baseline as TB
                model = TB.load_model(ckpt) if hasattr(TB, "load_model") else None
                print("  [trufor] 复用 trufor_baseline 的模型加载（见其脚本）", flush=True)
            else:
                print("  [trufor] ckpt 缺失，跳过", flush=True)
        except Exception as e:
            print(f"  [trufor] 失败: {type(e).__name__}: {e}", flush=True)

    json.dump(res, open(args.out, "w"), indent=1, ensure_ascii=False)
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
