#!/usr/bin/env python3
"""
排查：低误报阈值是被什么样的"干净图"抬高的？

背景：把评测负样本从 APRICOT 的 1,000 扩到 21,000（+COCO-train 20,000）后，
      @1% FPR 的 TPR 从 0.29 掉到 0.06；分源 FPR 显示误报全部集中在 COCO-train。
      需要判断这些高分干净图是
        (a) 含"类补丁"内容（标牌/文字/贴纸/强纹理）→ 负样本标签不可靠，属评测偏差
        (b) 普通自然照片 → 检测器存在系统性误报（真实现象）

做法：取分数最高的 N 张干净图，拼成网格图（带分数与源标签）+ 打印清单，人工核查。

输出：/tmp/fp_inspect/grid_<ckpt>.png + results/tvc/fp_inspect.json
"""
import os, json, argparse
os.environ.setdefault("HF_HOME", "/home/oyp/.cache/huggingface")
WS = "/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard"
TVC = f"{WS}/results/tvc"; EVAL = f"{TVC}/unified_eval_set"; CACHE = f"{TVC}/feat_cache"
NEGS_IDX = "/home/oyp/fastdata/apde_negs/index.json"
OUTDIR = "/tmp/fp_inspect"

import numpy as np
import torch, torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="fab_moeatt_s2_moe_att")
ap.add_argument("--top", type=int, default=40)
ap.add_argument("--out", default=f"{TVC}/fp_inspect.json")
a = ap.parse_args()

S3 = ["casia_au", "coco_val", "inria_noped"]


class Net(nn.Module):
    def __init__(s, cdim, fmode):
        super().__init__()
        b = models.shufflenet_v2_x1_0(weights=None)
        td = b.fc.in_features; b.fc = nn.Identity(); s.tex = b
        s.fmode = fmode; s.cdim = cdim
        if fmode == "attention":
            s.q = nn.Linear(td, 128); s.k = nn.Linear(cdim, 128); s.v = nn.Linear(cdim, 256)
            s.head = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2))
        elif fmode in ("moe", "moe_att"):
            s.e1 = nn.Sequential(nn.LayerNorm(td), nn.Linear(td, 256), nn.ReLU())
            s.e2 = nn.Sequential(nn.LayerNorm(cdim), nn.Linear(cdim, 256), nn.ReLU())
            s.router = nn.Sequential(nn.LayerNorm(td + cdim), nn.Linear(td + cdim, 64), nn.ReLU(), nn.Linear(64, 2))
            if fmode == "moe_att":
                s.q = nn.Linear(td, 128); s.k = nn.Linear(cdim, 128)
            s.head = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2))
        elif fmode == "tex":
            s.head = nn.Sequential(nn.LayerNorm(td), nn.Linear(td, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2))
        elif fmode == "sem":
            s.head = nn.Sequential(nn.LayerNorm(cdim), nn.Linear(cdim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2))
        else:
            s.head = nn.Sequential(nn.LayerNorm(td + cdim), nn.Linear(td + cdim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2))
    def forward(s, x, cf):
        if s.fmode == "sem": return s.head(cf)
        t = s.tex(x)
        if s.fmode == "attention":
            att = torch.sigmoid((s.q(t) * s.k(cf)).sum(-1, keepdim=True) / np.sqrt(128))
            return s.head(att * s.v(cf))
        if s.fmode in ("moe", "moe_att"):
            import math
            w = torch.softmax(s.router(torch.cat([t, cf], 1)), 1)
            z = w[:, 0:1] * s.e1(t) + w[:, 1:2] * s.e2(cf)
            if s.fmode == "moe_att":
                att = torch.sigmoid((s.q(t) * s.k(cf)).sum(-1, keepdim=True) / math.sqrt(128))
                return s.head(att * z)
            return s.head(z)
        if s.fmode == "tex": return s.head(t)
        return s.head(torch.cat([t, cf], 1))


class DS(Dataset):
    def __init__(s, rows, feats, size):
        s.r, s.f, s.size = rows, feats, size
        s.tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    def __len__(s): return len(s.r)
    def __getitem__(s, i):
        try: im = Image.open(s.r[i]["path"]).convert("RGB")
        except Exception: im = Image.new("RGB", (s.size, s.size))
        if im.size != (s.size, s.size): im = im.resize((s.size, s.size), Image.BILINEAR)
        return s.tf(im), s.f[i]


def main():
    pv = [{"path": f"{EVAL}/pos/{f}", "src": "apricot_pos"} for f in sorted(os.listdir(f"{EVAL}/pos"))]
    nv = [{"path": f"{EVAL}/{s}/{f}", "src": s} for s in S3 + ["columbia_au"] for f in sorted(os.listdir(f"{EVAL}/{s}"))]
    cn = [{"path": r["path"], "src": "coco_train"} for r in json.load(open(NEGS_IDX))]
    rows = pv + nv + cn
    X = np.load(f"{CACHE}/lowfpr_sem_21138.npz")["X"]
    assert len(X) == len(rows), (len(X), len(rows))

    ck = torch.load(f"{TVC}/{a.ckpt}.pt", map_location="cpu", weights_only=False)
    m = Net(ck["cdim"], ck["fmode"]).cuda(); m.load_state_dict(ck["state_dict"]); m.eval()
    sc = []
    with torch.no_grad():
        dl = DataLoader(DS(rows, X, ck["size"]), batch_size=64, shuffle=False, num_workers=0)
        for x, cf in dl:
            sc.extend(torch.softmax(m(x.cuda(), cf.cuda()), 1)[:, 1].float().cpu().numpy().tolist())
    sc = np.array(sc)
    neg_idx = np.arange(len(pv), len(rows))
    order = neg_idx[np.argsort(-sc[neg_idx])][:a.top]

    os.makedirs(OUTDIR, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols, nrow = 8, (a.top + 7) // 8
    fig, axes = plt.subplots(nrow, cols, figsize=(cols * 2.1, nrow * 2.3))
    picks = []
    for i, idx in enumerate(order):
        ax = axes.flat[i]
        try:
            im = Image.open(rows[idx]["path"]).convert("RGB")
            im.thumbnail((256, 256))
            ax.imshow(im)
        except Exception:
            ax.text(.5, .5, "load fail", ha="center")
        ax.set_title(f"{sc[idx]:.3f} {rows[idx]['src']}", fontsize=7)
        ax.axis("off")
        picks.append({"path": rows[idx]["path"], "score": round(float(sc[idx]), 4), "src": rows[idx]["src"]})
    for j in range(len(order), nrow * cols):
        axes.flat[j].axis("off")
    plt.tight_layout()
    grid = f"{OUTDIR}/grid_{a.ckpt}.png"
    plt.savefig(grid, dpi=110)
    print(f"网格图 → {grid}", flush=True)

    # 分源统计：分数 > p99(全负样本) 的比例、以及"接近阈值"的分数分布
    thr99 = float(np.percentile(sc[neg_idx], 99))
    res = {"ckpt": a.ckpt, "thr_p99": round(thr99, 4),
           "neg_score_by_src": {}, "top_fp": picks, "grid": grid}
    for s_ in ["casia_au", "coco_val", "inria_noped", "columbia_au", "coco_train"]:
        v = np.array([sc[i] for i in neg_idx if rows[i]["src"] == s_])
        res["neg_score_by_src"][s_] = {"n": int(len(v)), "mean": round(float(v.mean()), 4),
                                       "p50": round(float(np.percentile(v, 50)), 4),
                                       "p95": round(float(np.percentile(v, 95)), 4),
                                       "p99": round(float(np.percentile(v, 99)), 4),
                                       "max": round(float(v.max()), 4),
                                       "frac_gt_thr99": round(float((v >= thr99).mean()), 4)}
        print(f"  {s_:12s} n={len(v):5d} mean={v.mean():.4f} p95={np.percentile(v,95):.4f} "
              f"p99={np.percentile(v,99):.4f} max={v.max():.4f}", flush=True)
    pos = np.array([sc[i] for i in range(len(pv))])
    res["pos"] = {"n": int(len(pos)), "mean": round(float(pos.mean()), 4),
                  "p10": round(float(np.percentile(pos, 10)), 4),
                  "p50": round(float(np.percentile(pos, 50)), 4),
                  "frac_gt_thr99": round(float((pos >= thr99).mean()), 4)}
    print(f"  apricot_pos  n={len(pos)} mean={pos.mean():.4f} p10={np.percentile(pos,10):.4f} "
          f"frac>thr99={(pos>=thr99).mean():.4f}", flush=True)
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"→ {a.out}")


if __name__ == "__main__":
    main()
