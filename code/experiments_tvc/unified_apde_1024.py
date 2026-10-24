#!/usr/bin/env python3
"""
APDE 新协议复测（长边 1024 统一契约 + 按攻击族拆分）
========================================================
数据：results/tvc/apde_imgs/{pos(396, 6 族), neg(300)}（已落盘，无需重下）
评测：NP-PatchLM 3 seed + 轻量 CNN（448 原生 / 1024 统一）
输出：results/tvc/unified_apde_1024.json（总体 + 分攻击族 + Wilson CI）
"""
import os, sys, json, argparse, math
os.environ.setdefault("HF_HOME", "/home/oyp/.cache/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np

WS = "/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard"
TVC = f"{WS}/results/tvc"
IMGS = f"{TVC}/apde_imgs"
sys.path.insert(0, f"{WS}/code/experiments_tvc")


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k/n; d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (round(max(0.0, c-h), 4), round(min(1.0, c+h), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--ckpts", default="lora_patch_detector,full_lora_s1,full_lora_s2")
    ap.add_argument("--cnn-sizes", default="448,1024")
    ap.add_argument("--out", default=f"{TVC}/unified_apde_1024.json")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from lora_mllm import load_lora_for_eval, PROMPT

    def fit(im, size):
        w, h = im.size; s = size / max(w, h)
        return im if abs(s-1) < 1e-6 else im.resize((max(1, int(w*s)), max(1, int(h*s))), Image.BILINEAR)

    pos_files = sorted(os.listdir(f"{IMGS}/pos"))
    neg_files = sorted(os.listdir(f"{IMGS}/neg"))
    fams = {}
    for f in pos_files: fams.setdefault(f.split("_")[0], []).append(f)
    print(f"  正样本 {len(pos_files)}（{len(fams)} 族：{list(fams)}）  负样本 {len(neg_files)}", flush=True)

    res = {"protocol": {"long_side": args.size, "n_pos": len(pos_files), "n_neg": len(neg_files),
                        "note": "APDE 图像集为已落盘副本（results/tvc/apde_imgs）"}}

    def report(name, pos_s, neg_s, pos_files_order):
        p = np.asarray(pos_s); n = np.asarray(neg_s)
        thr = float(np.percentile(n, 95)); k_ = int((p >= thr).sum())
        row = {"pos_mean": round(float(p.mean()), 4), "neg_mean": round(float(n.mean()), 4),
               "thr5": round(thr, 4),
               "TPR@1pct": round(float((p >= np.percentile(n, 99)).mean()), 4),
               "TPR@5pct": round(k_/len(p), 4), "wilson5": wilson(k_, len(p)),
               "TPR@10pct": round(float((p >= np.percentile(n, 90)).mean()), 4),
               "by_family": {}}
        idx = {f: i for i, f in enumerate(pos_files_order)}
        for fam, fs in fams.items():
            ii = [idx[f] for f in fs]
            sf = p[ii]; kf = int((sf >= thr).sum())
            row["by_family"][fam] = {"n": len(sf), "tpr5": round(kf/len(sf), 4),
                                     "wilson": wilson(kf, len(sf))}
        res[name] = row
        print(f"  [{name}] @1%={row['TPR@1pct']:.3f} @5%={row['TPR@5pct']:.3f}"
              f"[{row['wilson5'][0]:.2f},{row['wilson5'][1]:.2f}] @10%={row['TPR@10pct']:.3f}", flush=True)
        print("      分族: " + "  ".join(f"{k}={v['tpr5']:.3f}" for k, v in row["by_family"].items()), flush=True)

    # ── NP-PatchLM（3 seed）──
    for ck in args.ckpts.split(","):
        ckpt = f"{TVC}/{ck}"
        if not os.path.isdir(ckpt): print(f"[skip] {ck}"); continue
        model, proc = load_lora_for_eval(ckpt, 16)
        cfg = json.load(open(f"{ckpt}/train_config.json")); yi, ni = cfg["yes_id"], cfg["no_id"]

        def sc(im):
            im = fit(im, args.size)
            msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
            t = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            enc = {k: (v.cuda() if torch.is_tensor(v) else v)
                   for k, v in proc(text=[t], images=[im], return_tensors="pt").items()}
            with torch.no_grad():
                lg = model(**enc).logits[0, -1]
            return float(torch.sigmoid(lg[yi] - lg[ni]))

        ps = [sc(Image.open(f"{IMGS}/pos/{f}").convert("RGB")) for f in pos_files]
        ns = [sc(Image.open(f"{IMGS}/neg/{f}").convert("RGB")) for f in neg_files]
        report(ck, ps, ns, pos_files)
        del model; torch.cuda.empty_cache()

    # ── 轻量 CNN（各尺寸）──
    import torch.nn as nn
    from torchvision import transforms, models
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    _sizes = [int(x) for x in args.cnn_sizes.split(",") if x.strip().isdigit()]
    for cnn_size in _sizes:
        for name in ("resnet18", "shufflenetv2"):
            ck = f"{TVC}/{name}_fair.pt"
            if not os.path.exists(ck): continue
            m = (models.resnet18(weights=None) if name == "resnet18" else models.shufflenet_v2_x1_0(weights=None))
            m.fc = nn.Linear(m.fc.in_features, 2)
            m.load_state_dict(torch.load(ck, map_location="cuda")); m = m.cuda().eval()

            def scn(im):
                im2 = fit(im, cnn_size)
                x = norm(transforms.ToTensor()(im2)).unsqueeze(0).cuda()
                with torch.no_grad():
                    return float(torch.softmax(m(x), 1)[0, 1])
            ps = [scn(Image.open(f"{IMGS}/pos/{f}").convert("RGB")) for f in pos_files]
            ns = [scn(Image.open(f"{IMGS}/neg/{f}").convert("RGB")) for f in neg_files]
            report(f"{name}@{cnn_size}", ps, ns, pos_files)
            del m; torch.cuda.empty_cache()

    json.dump(res, open(args.out, "w"), indent=1, ensure_ascii=False)
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
