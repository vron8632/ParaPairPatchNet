#!/usr/bin/env python3
"""
P1：族级泛化（APDE 严格按族留一）
==================================
训练：APDE train（8 族）
测试：APDE test（13 族），其中 5 族 train 从未出现：
      AA / AdvCloak / AdvSticker / AdvTshirt / UPC
对比：ShuffleNetV2 用 1,500 张（旧）vs APDE train ~50k 张（新）
指标：@5%FPR TPR —— 总 / 见过 8 族 / 未见 5 族 / 分族
"""
import os, sys, json, argparse, math
import numpy as np
from PIL import Image

WS = "/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard"
TVC = f"{WS}/results/tvc"
TRAIN_JPG = "/home/oyp/fastdata/apde_train_jpg"
TEST_JPG = "/home/oyp/fastdata/apde_test_jpg"

SEEN = {"advpatch", "DM-NAP", "GNAP", "TCA", "TCEGA", "tsea", "tsea-mim", "tsea-pgd"}
UNSEEN = {"AA", "AdvCloak", "AdvSticker", "AdvTshirt", "UPC"}


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k/n; d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (round(max(0.0, c-h), 4), round(min(1.0, c+h), 4))


def fit(im, size):
    w, h = im.size; s = size / max(w, h)
    return im if abs(s-1) < 1e-6 else im.resize((max(1, int(w*s)), max(1, int(h*s))), Image.BILINEAR)


def random_shape_overlay(im, rng):
    """在图上叠加一个非矩形的"伪补丁"（圆/椭圆/多边形/环），用于修复形状偏差"""
    from PIL import ImageDraw, ImageFilter
    import random as _r
    w, h = im.size
    draw = ImageDraw.Draw(im, "RGBA")
    frac = rng.uniform(0.02, 0.10)
    sw = max(12, int((w * h * frac) ** 0.5))
    sh = max(12, int(sw * rng.uniform(0.6, 1.6)))
    cx = rng.randint(sw // 2, max(sw // 2 + 1, w - sw // 2))
    cy = rng.randint(sh // 2, max(sh // 2 + 1, h - sh // 2))
    box = [cx - sw // 2, cy - sh // 2, cx + sw // 2, cy + sh // 2]
    col = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255), 255)
    kind = rng.choice(["circle", "ellipse", "polygon", "ring"])
    if kind == "circle":
        draw.ellipse(box, fill=col)
    elif kind == "ellipse":
        draw.ellipse([box[0], box[1], box[2], box[1] + int(sh * rng.uniform(1.2, 2.2))], fill=col)
    elif kind == "ring":
        draw.ellipse(box, outline=col, width=max(4, sw // 6))
    else:
        n = rng.randint(5, 8)
        import math
        pts = []
        for i in range(n):
            a = 2 * math.pi * i / n + rng.uniform(-0.3, 0.3)
            rx, ry = sw / 2 * rng.uniform(0.7, 1.0), sh / 2 * rng.uniform(0.7, 1.0)
            pts.append((cx + rx * math.cos(a), cy + ry * math.sin(a)))
        draw.polygon(pts, fill=col)
    return im


def fpr_penalty(logits, labels, tau0, mode):
    """低误报约束：压制负样本(y=0)的正类概率超过 tau0"""
    import torch, torch.nn.functional as F
    p = torch.softmax(logits, 1)[:, 1]
    neg = labels == 0
    if not neg.any():
        return torch.zeros((), device=logits.device)
    pn = p[neg]
    if mode == "hinge":
        return F.relu(pn - tau0).mean()
    return torch.sigmoid((pn - tau0) / 0.05).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--limit-train", type=int, default=0, help="0=全量")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=8)
    # ① FPR 约束训练
    ap.add_argument("--fpr-lambda", type=float, default=0.0, help="0=关闭（纯CE）")
    ap.add_argument("--tau0", type=float, default=0.3, help="低误报工作点阈值")
    ap.add_argument("--fpr-mode", default="hinge", choices=["hinge", "soft"])
    # ② 形状增广（针对 UPC 形状偏差）
    ap.add_argument("--shape-aug", type=float, default=0.0, help="在正样本上叠加随机形状伪补丁的概率")
    ap.add_argument("--out", default=f"{TVC}/p1_family_generalization.json")
    args = ap.parse_args()

    import torch, torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms, models
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    tr = json.load(open(f"{TRAIN_JPG}/index.json"))
    te = json.load(open(f"{TEST_JPG}/index.json"))
    if args.limit_train:
        import random; random.Random(0).shuffle(tr); tr = tr[:args.limit_train]
    print(f"[P1] train={len(tr)}  test={len(te)}", flush=True)

    tf_tr = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.ColorJitter(0.2, 0.2, 0.2),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
    tf_ev = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])

    import random as _rand
    _rng = _rand.Random(args.seed)

    class DS(Dataset):
        def __init__(s, rows, tf, shape_aug=0.0): s.rows, s.tf, s.sa = rows, tf, shape_aug
        def __len__(s): return len(s.rows)
        def __getitem__(s, i):
            r = s.rows[i]
            try: im = Image.open(r["path"]).convert("RGB")
            except Exception: im = Image.new("RGB", (args.size, args.size))
            if s.sa > 0 and r["label"] == 1 and _rng.random() < s.sa:
                im = random_shape_overlay(im.copy(), _rng)
            return s.tf(fit(im, args.size)), r["label"]

    m = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, 2); m = m.cuda()
    from collections import Counter
    c = Counter(r["label"] for r in tr)
    w = torch.tensor([len(tr)/(2*max(c[0], 1)), len(tr)/(2*max(c[1], 1))]).cuda()
    ce = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    dl = DataLoader(DS(tr, tf_tr, args.shape_aug), batch_size=args.bs, shuffle=True,
                    num_workers=args.num_workers, pin_memory=False, drop_last=True, persistent_workers=(args.num_workers>0))
    for ep in range(args.epochs):
        m.train(); tot = 0.0
        for j, (x, y) in enumerate(dl):
            x, y = x.cuda(), y.cuda()
            logits = m(x)
            loss = ce(logits, y)
            if args.fpr_lambda > 0:
                loss = loss + args.fpr_lambda * fpr_penalty(logits, y, args.tau0, args.fpr_mode)
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
            if (j+1) % 300 == 0: print(f"  ep{ep+1} {j+1}/{len(dl)} loss={tot/(j+1):.4f}", flush=True)
        sch.step()
        print(f"  [ep{ep+1}/{args.epochs}] loss={tot/max(len(dl),1):.4f}", flush=True)

    # 评测
    m.eval()
    @torch.no_grad()
    def score(rows):
        out = []
        dl2 = DataLoader(DS(rows, tf_ev), batch_size=64, shuffle=False,
                         num_workers=args.num_workers, pin_memory=False)
        for x, _ in dl2:
            out.extend(torch.softmax(m(x.cuda()), 1)[:, 1].float().cpu().numpy().tolist())
        return np.array(out)

    neg = [r for r in te if r["label"] == 0]
    pos = [r for r in te if r["label"] == 1]
    print(f"[P1] 评测: pos={len(pos)} neg={len(neg)}", flush=True)
    sn = score(neg); sp = score(pos)
    thr = float(np.percentile(sn, 95))
    res = {"params_M": round(sum(p.numel() for p in m.parameters())/1e6, 2),
           "fpr_lambda": args.fpr_lambda, "tau0": args.tau0, "fpr_mode": args.fpr_mode,
           "shape_aug": args.shape_aug, "size": args.size, "epochs": args.epochs,
           "n_train": len(tr), "n_pos": len(pos), "n_neg": len(neg), "thr@5pct": round(thr, 4)}
    by_fam = {}
    for r, s in zip(pos, sp):
        by_fam.setdefault(r["method"], []).append(s)
    fam_tpr = {k: round(float((np.array(v) >= thr).mean()), 4) for k, v in by_fam.items()}
    seen_tpr = [v for k, v in fam_tpr.items() if k in SEEN]
    unseen_tpr = [v for k, v in fam_tpr.items() if k in UNSEEN]
    k_all = int((sp >= thr).sum())
    # 多工作点：5% / 1% / 0.1%
    multi = {}
    for a in (0.05, 0.01, 0.001):
        thr_a = float(np.percentile(sn, 100*(1-a)))
        k_a = int((sp >= thr_a).sum())
        multi[f"@{a*100:g}pct"] = {"thr": round(thr_a,4), "tpr": round(k_a/len(pos),4),
                                   "wilson": wilson(k_a, len(pos))}
    res.update({
        "workpoints": multi,
        "overall@5pct": {"tpr": round(k_all/len(pos), 4), "wilson": wilson(k_all, len(pos))},
        "seen_fams": {k: fam_tpr[k] for k in fam_tpr if k in SEEN},
        "unseen_fams": {k: fam_tpr[k] for k in fam_tpr if k in UNSEEN},
        "seen_mean": round(float(np.mean(seen_tpr)), 4) if seen_tpr else None,
        "unseen_mean": round(float(np.mean(unseen_tpr)), 4) if unseen_tpr else None,
        "by_family": fam_tpr,
    })
    print(f"\n  ✅ ShuffleNetV2({res['params_M']}M) 训练 {len(tr)} 张")
    print(f"     总体@5%={res['overall@5pct']['tpr']}  见过8族均值={res['seen_mean']}  **未见5族均值={res['unseen_mean']}**")
    print(f"     分族: {json.dumps(fam_tpr, ensure_ascii=False)}")
    ckpt = args.out.replace(".json", ".pt")
    torch.save({"state_dict": m.state_dict(), "size": args.size, "n_train": len(tr)}, ckpt)
    res["ckpt"] = ckpt
    json.dump(res, open(args.out, "w"), indent=1, ensure_ascii=False)
    print(f"→ {args.out}\n   ckpt: {ckpt}")


if __name__ == "__main__":
    main()
