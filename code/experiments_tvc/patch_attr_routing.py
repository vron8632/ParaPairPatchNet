#!/usr/bin/env python3
"""
Exp-D｜路由权重 ↔ 样本属性（补丁面积 / 攻击目标 / 源数据集）的相关性

动机：Exp-C 的干预实验显示"路由的样本级条件化"有价值，但不知道它到底在看什么。
本脚本从 APDE test 的 parquet 里取补丁 mask（面积比例）、attack_goal、source_dataset，
与 moe_att 的路由权重（语义专家权重 w_sem）做相关性分析，并按面积分箱看性能分布。

回答三个问题：
  Q1 路由是否对补丁面积敏感（Spearman ρ）？
  Q2 路由是否对攻击目标（appearing / disappearing / …）有系统差异？
  Q3 检测性能是否随面积单调（模型在哪个尺度失效）？

输出：results/tvc/patch_attr_routing.json
"""
import os, json, math, glob, io, argparse
os.environ.setdefault("HF_HOME", "/home/oyp/.cache/huggingface")
WS = "/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard"
TVC = f"{WS}/results/tvc"; CACHE = f"{TVC}/feat_cache"
TEST_PARQUET = "/home/oyp/fastdata/apde_test"
TEST_JPG = "/home/oyp/fastdata/apde_test_jpg"

import numpy as np
import pandas as pd
import torch, torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from scipy.stats import spearmanr

ap = argparse.ArgumentParser()
ap.add_argument("--ckpts", default="mech_moeatt_s0,fab_moeatt_s2")
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--out", default=f"{TVC}/patch_attr_routing.json")
a = ap.parse_args()


def mask_meta():
    """(domain, sha16) → 面积比例 / attack_goal / source_dataset"""
    out = {}
    for f in sorted(glob.glob(f"{TEST_PARQUET}/*.parquet")):
        tag = os.path.basename(f).replace(".parquet", "")
        df = pd.read_parquet(f, columns=["image_sha256", "mask", "method", "attack_goal", "source_dataset"])
        for _, r in df.iterrows():
            m = r["mask"]
            area = None
            if isinstance(m, dict) and m.get("bytes"):
                try:
                    im = Image.open(io.BytesIO(m["bytes"]))
                    arr = np.asarray(im.convert("L"))
                    area = float((arr > 0).mean())
                except Exception:
                    area = None
            out[f"{tag}_{str(r['image_sha256'])[:16]}"] = {
                "area": area, "method": str(r["method"]),
                "goal": str(r["attack_goal"]), "src_dataset": str(r["source_dataset"])}
        print(f"  parquet {tag}: {len(df)}", flush=True)
    return out


class Net(nn.Module):
    def __init__(s, cdim):
        super().__init__()
        b = models.shufflenet_v2_x1_0(weights=None)
        td = b.fc.in_features; b.fc = nn.Identity(); s.tex = b
        s.e1 = nn.Sequential(nn.LayerNorm(td), nn.Linear(td, 256), nn.ReLU())
        s.e2 = nn.Sequential(nn.LayerNorm(cdim), nn.Linear(cdim, 256), nn.ReLU())
        s.router = nn.Sequential(nn.LayerNorm(td + cdim), nn.Linear(td + cdim, 64), nn.ReLU(), nn.Linear(64, 2))
        s.q = nn.Linear(td, 128); s.k = nn.Linear(cdim, 128)
        s.head = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2))
    def forward(s, x, cf):
        t = s.tex(x)
        w = torch.softmax(s.router(torch.cat([t, cf], 1)), 1)
        z = w[:, 0:1] * s.e1(t) + w[:, 1:2] * s.e2(cf)
        att = torch.sigmoid((s.q(t) * s.k(cf)).sum(-1, keepdim=True) / math.sqrt(128))
        return s.head(att * z), w[:, 1], att.squeeze(1)


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
    print("读取 parquet 元数据 ...", flush=True)
    meta = mask_meta()
    te = json.load(open(f"{TEST_JPG}/index.json"))
    Xte = np.load(f"{CACHE}/fab_apde_14000.npz")["X"]
    assert len(Xte) == len(te)

    # 对齐：jpg 名 = "{domain}_{sha16}.jpg"
    keys, areas, goals, srcs = [], [], [], []
    miss = 0
    for r in te:
        stem = os.path.basename(r["path"]).replace(".jpg", "")
        mm = meta.get(stem)
        if mm is None:
            miss += 1; keys.append(None); areas.append(np.nan); goals.append("unknown"); srcs.append("unknown")
        else:
            keys.append(stem); areas.append(mm["area"] if mm["area"] is not None else np.nan)
            goals.append(mm["goal"]); srcs.append(mm["src_dataset"])
    areas = np.array(areas, float)
    print(f"  对齐成功 {len(te)-miss}/{len(te)}（未匹配 {miss}）", flush=True)
    print(f"  面积比例：中位 {np.nanmedian(areas):.4f}，范围 {np.nanmin(areas):.4f}–{np.nanmax(areas):.4f}", flush=True)

    results = {"_dataset": {"n": len(te), "n_aligned": len(te) - miss,
                            "area_median": round(float(np.nanmedian(areas)), 5),
                            "goals": {g: int(sum(1 for x in goals if x == g)) for g in sorted(set(goals))}}}
    y = np.array([r["label"] for r in te])
    methods = np.array([r.get("method", "clean") for r in te])

    for name in [c.strip() for c in a.ckpts.split(",") if c.strip()]:
        p = f"{TVC}/{name}_moe_att.pt"
        if not os.path.exists(p):
            print(f"  ⚠️ 缺 {p}", flush=True); continue
        ck = torch.load(p, map_location="cpu", weights_only=False)
        m = Net(ck["cdim"]).cuda(); m.load_state_dict(ck["state_dict"]); m.eval()
        scores, wsems, atts = [], [], []
        with torch.no_grad():
            dl = DataLoader(DS(te, Xte, ck["size"]), batch_size=a.bs, shuffle=False, num_workers=0)
            for x, cf in dl:
                o, w1, at = m(x.cuda(), cf.cuda())
                scores.extend(torch.softmax(o, 1)[:, 1].cpu().numpy().tolist())
                wsems.extend(w1.cpu().numpy().tolist()); atts.extend(at.cpu().numpy().tolist())
        scores, wsems, atts = map(np.array, (scores, wsems, atts))
        pos = y == 1
        A = areas[pos]; wpos = wsems[pos]; sp = scores[pos]; sn = scores[~pos]
        ok = ~np.isnan(A)
        rho_area = spearmanr(np.log10(A[ok] + 1e-6), wpos[ok])
        thr5 = float(np.percentile(sn, 95))
        # 面积分位箱 → TPR / 平均语义权重
        qs = np.quantile(A[ok], [0, .2, .4, .6, .8, 1.0])
        bins = []
        for i in range(5):
            sel = (A >= qs[i]) & (A <= qs[i + 1]) & ok
            if sel.sum() < 10: continue
            bins.append({"area_lo": round(float(qs[i]), 5), "area_hi": round(float(qs[i + 1]), 5),
                         "n": int(sel.sum()), "tpr@5pct": round(float((sp[sel] >= thr5).mean()), 4),
                         "sem_w_mean": round(float(wpos[sel].mean()), 4)})
        # 按攻击目标 / 源数据集
        by_goal, by_src = {}, {}
        for g in sorted(set(np.array(goals)[pos])):
            sel = (np.array(goals)[pos] == g) & ok
            if sel.sum() >= 20:
                by_goal[g] = {"n": int(sel.sum()), "sem_w_mean": round(float(wpos[sel].mean()), 4),
                              "tpr@5pct": round(float((sp[sel] >= thr5).mean()), 4),
                              "area_median": round(float(np.median(A[sel])), 5)}
        for s_ in sorted(set(np.array(srcs)[pos])):
            sel = (np.array(srcs)[pos] == s_)
            if sel.sum() >= 20:
                by_src[s_] = {"n": int(sel.sum()), "sem_w_mean": round(float(wpos[sel].mean()), 4),
                              "tpr@5pct": round(float((sp[sel] >= thr5).mean()), 4)}
        results[name] = {
            "spearman_semw_vs_logarea": {"rho": round(float(rho_area.statistic), 4),
                                         "p": float(f"{rho_area.pvalue:.3g}"), "n": int(ok.sum())},
            "att_vs_logarea": (lambda rr: {"rho": round(float(rr.statistic), 4), "p": float(f"{rr.pvalue:.3g}")})(
                spearmanr(np.log10(A[ok] + 1e-6), atts[pos][ok])),
            "area_bins": bins, "by_goal": by_goal, "by_source_dataset": by_src,
            "sem_w_mean_pos": round(float(wpos.mean()), 4),
            "sem_w_std_pos": round(float(wpos.std()), 4),
        }
        print(f"  ✅ {name}: ρ(w_sem, log area)={rho_area.statistic:+.3f} (p={rho_area.pvalue:.3g})", flush=True)
        for b in bins:
            print(f"      area {b['area_lo']:.4f}–{b['area_hi']:.4f}: n={b['n']} "
                  f"TPR@5%={b['tpr@5pct']} w_sem={b['sem_w_mean']}", flush=True)
        for g, v in by_goal.items():
            print(f"      goal={g:14s} n={v['n']} TPR@5%={v['tpr@5pct']} w_sem={v['sem_w_mean']}", flush=True)
        del m
        torch.cuda.empty_cache()

    json.dump(results, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"\n→ {a.out}")


if __name__ == "__main__":
    main()
