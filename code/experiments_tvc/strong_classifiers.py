#!/usr/bin/env python3
"""
强分类器基线（回应"为何只对比弱分类器"）
============================================
在**完全相同的统一契约**下评测：EfficientNet-B0/B4、ConvNeXt-T/B、
ViT-B/16、Swin-T/B（均可由 torchvision 提供，ImageNet 预训练权重）。
同域 APRICOT（138 正 + 1000 负）与跨域 APDE（396 正 + 300 负）都测。
输出: results/tvc/strong_classifiers.json
用法: $PY strong_classifiers.py [--models efficientnet_b0,vit_b_16,...] [--size 1024] [--epochs 8]
"""
import os, sys, json, argparse, math, time
import numpy as np

WS = "/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard"
TVC = f"{WS}/results/tvc"
EVAL = f"{TVC}/unified_eval_set"
APDE = f"{TVC}/apde_imgs"
sys.path.insert(0, f"{WS}/code/experiments_tvc")

MODELS = {
    "efficientnet_b0": ("efficientnet_b0", "EfficientNet_B0_Weights"),
    "efficientnet_b4": ("efficientnet_b4", "EfficientNet_B4_Weights"),
    "convnext_tiny":   ("convnext_tiny",   "ConvNeXt_Tiny_Weights"),
    "convnext_base":   ("convnext_base",   "ConvNeXt_Base_Weights"),
    "vit_b_16":        ("vit_b_16",        "ViT_B_16_Weights"),
    "swin_t":          ("swin_t",          "Swin_T_Weights"),
    "swin_b":          ("swin_b",          "Swin_B_Weights"),
}


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k/n; d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (round(max(0.0, c-h), 4), round(min(1.0, c+h), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="efficientnet_b0,convnext_tiny,vit_b_16,swin_t")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4, help="学习率（ViT 类需 3e-4）")
    ap.add_argument("--bs", type=int, default=4, help="显存受限，默认 4（原 16 在 1024 输入下 OOM）")
    ap.add_argument("--out", default=f"{TVC}/strong_classifiers.json")
    args = ap.parse_args()

    import torch, torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms, models
    from PIL import Image

    def fit(im, size, square=False):
        w, h = im.size; s = size / max(w, h)
        im2 = im if abs(s-1) < 1e-6 else im.resize((max(1, int(round(w*s))), max(1, int(round(h*s)))), Image.BILINEAR)
        if square:   # ViT 要求严格正方形 → letterbox 填充
            canvas = Image.new("RGB", (size, size), (127, 127, 127))
            canvas.paste(im2, ((size-im2.width)//2, (size-im2.height)//2))
            return canvas
        return im2

    idx = json.load(open(f"{TVC}/lora_data/index.json"))
    tr = [r for r in idx if r["split"] == "train"]

    class DS(Dataset):
        def __init__(s, rows, tf): s.rows, s.tf = rows, tf
        def __len__(s): return len(s.rows)
        def __getitem__(s, i):
            r = s.rows[i]
            return s.tf(fit(Image.open(r["path"]).convert("RGB"), _cur_size[0], _cur_size[1])), r["label"]
    tf_tr = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.ColorJitter(0.2, 0.2, 0.2),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
    tf_ev = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
    _cur_size = [args.size, False]   # [size, square]
    dl = DataLoader(DS(tr, tf_tr), batch_size=args.bs, shuffle=True, num_workers=0, pin_memory=False)

    def eval_dirs(m, dirs_pos, dirs_neg):
        p, n = [], []
        for d in dirs_pos:
            for f in sorted(os.listdir(d)):
                with torch.no_grad():
                    x = tf_ev(fit(Image.open(f"{d}/{f}").convert("RGB"), _cur_size[0], _cur_size[1])).unsqueeze(0).cuda()
                    p.append(float(torch.softmax(m(x), 1)[0, 1]))
        for d in dirs_neg:
            for f in sorted(os.listdir(d)):
                with torch.no_grad():
                    x = tf_ev(fit(Image.open(f"{d}/{f}").convert("RGB"), _cur_size[0], _cur_size[1])).unsqueeze(0).cuda()
                    n.append(float(torch.softmax(m(x), 1)[0, 1]))
        return np.array(p), np.array(n)

    # 增量保存前提：合并已有结果（避免新进程覆盖历史模型）
    res = {}
    if os.path.exists(args.out):
        try:
            res = json.load(open(args.out))
            print(f"  [merge] 已有结果: {[k for k in res if k != 'protocol']}", flush=True)
        except Exception:
            res = {}
    res["protocol"] = {"size": args.size, "epochs": args.epochs, "train_n": len(tr)}
    for name in args.models.split(","):
        if name not in MODELS: print(f"[skip] {name}"); continue
        fn, wcls = MODELS[name]
        try:
            W = getattr(models, wcls)
            m = getattr(models, fn)(weights=W.IMAGENET1K_V1)
            # ViT-B/16 位置编码固定 224，无法直接吃 1024 输入 → 该模型用原生 224
            # 输入尺寸：ViT 固定 224；ConvNeXt 系列在 1024 下会 OOM → 用 512；其余用 args.size
            if fn.startswith("vit"):
                _size = 224
            elif fn.startswith("convnext"):
                _size = min(args.size, 512)
            else:
                _size = args.size
            _cur_size[0] = _size          # 让 Dataset/评测使用该模型的原生输入尺寸
            _cur_size[1] = fn.startswith("vit")   # ViT 需要正方形 letterbox
            # 通用分类头替换：不同架构的最后一层位置不同
            if hasattr(m, "fc"):
                m.fc = nn.Linear(m.fc.in_features, 2)
            elif hasattr(m, "head") and hasattr(m.head, "in_features"):
                m.head = nn.Linear(m.head.in_features, 2)                       # swin
            elif hasattr(m, "heads") and hasattr(getattr(m.heads, "head", None), "in_features"):
                m.heads.head = nn.Linear(m.heads.head.in_features, 2)           # vit
            elif hasattr(m, "classifier"):
                cl = m.classifier
                if isinstance(cl, nn.Sequential):
                    for i in range(len(cl) - 1, -1, -1):
                        if isinstance(cl[i], nn.Linear):
                            cl[i] = nn.Linear(cl[i].in_features, 2); break      # efficientnet/convnext
                elif isinstance(cl, nn.Linear):
                    m.classifier = nn.Linear(cl.in_features, 2)
            else:
                raise RuntimeError(f"无法定位 {name} 的分类头")
        except Exception as e:
            print(f"  [{name}] 构建失败: {e}"); continue
        m = m.cuda()
        opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)
        # 类平衡权重（与轻量 CNN 的公平版对齐：patch:clean = 1000:500）
        from collections import Counter
        _c = Counter(r["label"] for r in tr)
        _w = torch.tensor([len(tr)/(2*_c[0]), len(tr)/(2*_c[1])]).cuda()
        crit = nn.CrossEntropyLoss(weight=_w)
        print(f"    [{name}] class weights: {_w.tolist()}", flush=True)
        t0 = time.time()
        m.train()
        for ep in range(args.epochs):
            tot = 0
            opt.zero_grad()
            for i, (x, y) in enumerate(dl):
                x, y = x.cuda(), y.cuda()
                loss = crit(m(x), y) / 4.0          # 梯度累积 4 步 ≈ bs16
                loss.backward()
                if (i + 1) % 4 == 0:
                    opt.step(); opt.zero_grad()
                tot += float(loss) * 4.0
            opt.step(); opt.zero_grad()
            print(f"    [{name}] ep{ep+1}/{args.epochs} loss={tot/max(len(dl),1):.4f}", flush=True)
        m.eval()
        row = {"train_min": round((time.time()-t0)/60, 1), "params_M": round(sum(p.numel() for p in m.parameters())/1e6, 1)}
        # 同域（统一评测集）
        p, n = eval_dirs(m, [f"{EVAL}/pos"], [f"{EVAL}/{s}" for s in ("casia_au","coco_val","inria_noped","columbia_au")])
        for scope, kk in (("3src", 3), ("4src", 4)):
            nn_ = n[:250*kk]
            for a in (0.05, 0.10):
                thr = float(np.percentile(nn_, 100*(1-a))); k_ = int((p >= thr).sum())
                row[f"apricot_{scope}@{int(a*100)}"] = round(k_/len(p), 4)
        row["apricot_params_M"] = row["params_M"]
        # 跨域（APDE）
        pa, na = eval_dirs(m, [f"{APDE}/pos"], [f"{APDE}/neg"])
        for a in (0.05, 0.10):
            thr = float(np.percentile(na, 100*(1-a))); k_ = int((pa >= thr).sum())
            row[f"apde@{int(a*100)}"] = round(k_/len(pa), 4)
        row["input_size"] = _cur_size[0]
        res[name] = row
        json.dump(res, open(args.out, "w"), indent=1, ensure_ascii=False)   # 增量保存
        print(f"  [{name}] 同域@5%: 3src={row.get('apricot_3src@5')} 4src={row.get('apricot_4src@5')}"
              f" | 跨域APDE@5%={row.get('apde@5')} | {row['params_M']}M params, {row['train_min']}min", flush=True)
        del m, opt
        import gc; gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    json.dump(res, open(args.out, "w"), indent=1, ensure_ascii=False)
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
