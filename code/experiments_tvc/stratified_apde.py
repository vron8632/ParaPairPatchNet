#!/usr/bin/env python3
"""
分层主表（方法论第①③条）：APDE 测试集上的失效模式分布
  - 按补丁面积五分位（mask 来自 APDE parquet）
  - 按 attack_goal（appearing / hiding / …）
  - 按族（seen 8 / unseen 5）
  - 按源数据集
输出：results/tvc/stratified_apde.json
"""
import os, sys, json, glob, argparse, io, math
os.environ.setdefault("HF_HOME", "/home/oyp/.cache/huggingface")
WS = "/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard"
TVC = f"{WS}/results/tvc"; CACHE = f"{TVC}/feat_cache"
TEST_PARQUET = "/home/oyp/fastdata/apde_test"
TEST_JPG = "/home/oyp/fastdata/apde_test_jpg"
SEEN = {"advpatch", "DM-NAP", "GNAP", "TCA", "TCEGA", "tsea", "tsea-mim", "tsea-pgd"}
UNSEEN = {"AA", "AdvCloak", "AdvSticker", "AdvTshirt", "UPC"}

import numpy as np
import pandas as pd
import torch, torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

ap = argparse.ArgumentParser()
ap.add_argument("--ckpts", default="mix_moe_att_s0_moe_att,mix_moe_att_s1_moe_att,mix_moe_att_s2_moe_att,"
                                  "fab_moeatt_s2_moe_att,mech_moeatt_s0_moe_att,expA_sem_s0_sem,ctlX_moe_att_s0_moe_att")
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--out", default=f"{TVC}/stratified_apde.json")
a = ap.parse_args()


def mask_meta():
    out = {}
    for f in sorted(glob.glob(f"{TEST_PARQUET}/*.parquet")):
        tag = os.path.basename(f).replace(".parquet", "")
        df = pd.read_parquet(f, columns=["image_sha256", "mask", "attack_goal", "source_dataset"])
        for _, r in df.iterrows():
            area = None
            m = r["mask"]
            if isinstance(m, dict) and m.get("bytes"):
                try:
                    arr = np.asarray(Image.open(io.BytesIO(m["bytes"])).convert("L"))
                    area = float((arr > 0).mean())
                except Exception:
                    area = None
            out[f"{tag}_{str(r['image_sha256'])[:16]}"] = {
                "area": area, "goal": str(r["attack_goal"]), "src": str(r["source_dataset"])}
    return out


class Net(nn.Module):
    def __init__(s, cdim, fmode):
        super().__init__()
        b = models.shufflenet_v2_x1_0(weights=None)
        td = b.fc.in_features; b.fc = nn.Identity(); s.tex = b
        s.fmode = fmode; s.cdim = cdim
        if fmode in ("moe", "moe_att"):
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
    def __init__(s, rows, feats, size, tf, dummy=False):
        s.r, s.f, s.tf, s.size, s.dummy = rows, feats, tf, size, dummy
    def __len__(s): return len(s.r)
    def __getitem__(s, i):
        if s.dummy: return torch.zeros(3, 8, 8), s.f[i]
        try: im = Image.open(s.r[i]["path"]).convert("RGB")
        except Exception: im = Image.new("RGB", (s.size, s.size))
        if im.size != (s.size, s.size): im = im.resize((s.size, s.size), Image.BILINEAR)
        return s.tf(im), s.f[i]


def main():
    print("读取 parquet 元数据 ...", flush=True)
    meta = mask_meta()
    te = json.load(open(f"{TEST_JPG}/index.json"))
    Xte = np.load(f"{CACHE}/fab_apde_14000.npz")["X"]
    y = np.array([r["label"] for r in te])
    pos_m = np.array([r["method"] for r in te if r["label"] == 1])

    areas, goals, srcs = [], [], []
    for r in te:
        stem = os.path.basename(r["path"]).replace(".jpg", "")
        mm = meta.get(stem)
        areas.append(mm["area"] if mm and mm["area"] is not None else np.nan)
        goals.append(mm["goal"] if mm else "unknown")
        srcs.append(mm["src"] if mm else "unknown")
    areas = np.array(areas, float)
    pos_idx = np.where(y == 1)[0]; neg_idx = np.where(y == 0)[0]
    print(f"对齐 {len(te)}（pos 面积缺失 {int(np.isnan(areas[pos_idx]).sum())}）", flush=True)

    results = {}
    for name in [c.strip() for c in a.ckpts.split(",") if c.strip()]:
        p = f"{TVC}/{name}.pt"
        if not os.path.exists(p):
            print(f"  ⚠️ 缺 {p}", flush=True); continue
        ck = torch.load(p, map_location="cpu", weights_only=False)
        size = ck["size"]
        m = Net(ck["cdim"], ck["fmode"]).cuda(); m.load_state_dict(ck["state_dict"]); m.eval()
        tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        sc = []
        with torch.no_grad():
            dl = DataLoader(DS(te, Xte, size, tf, dummy=(ck["fmode"] == "sem")),
                            batch_size=a.bs, shuffle=False, num_workers=0)
            for x, cf in dl:
                sc.extend(torch.softmax(m(x.cuda(), cf.cuda()), 1)[:, 1].float().cpu().numpy().tolist())
        sc = np.array(sc)
        sp, sn = sc[pos_idx], sc[neg_idx]
        thr5 = float(np.percentile(sn, 95)); thr1 = float(np.percentile(sn, 99))
        A = areas[pos_idx]; ok = ~np.isnan(A)
        qs = np.quantile(A[ok], [0, .2, .4, .6, .8, 1.0])
        bins = []
        for i in range(5):
            sel = (A >= qs[i]) & (A <= qs[i + 1]) & ok
            if sel.sum() < 10: continue
            bins.append({"area_lo": round(float(qs[i]), 5), "area_hi": round(float(qs[i + 1]), 5),
                         "n": int(sel.sum()), "tpr@5pct": round(float((sp[sel] >= thr5).mean()), 4),
                         "tpr@1pct": round(float((sp[sel] >= thr1).mean()), 4)})
        fam = {f: round(float((sp[pos_m == f] >= thr5).mean()), 4) for f in sorted(set(pos_m))}
        by_goal = {}
        ga = np.array(goals)[pos_idx]
        for g_ in sorted(set(ga)):
            sel = (ga == g_) & ok
            if sel.sum() >= 20:
                by_goal[g_] = {"n": int(sel.sum()), "tpr@5pct": round(float((sp[sel] >= thr5).mean()), 4),
                               "area_median": round(float(np.median(A[sel])), 5)}
        by_src = {}
        sa = np.array(srcs)[pos_idx]
        for s_ in sorted(set(sa)):
            sel = sa == s_
            if sel.sum() >= 20:
                by_src[s_] = {"n": int(sel.sum()), "tpr@5pct": round(float((sp[sel] >= thr5).mean()), 4)}
        seen = [v for k, v in fam.items() if k in SEEN]; unseen = [v for k, v in fam.items() if k in UNSEEN]
        results[name] = {"fmode": ck["fmode"],
                         "apde@5pct": round(float((sp >= thr5).mean()), 4),
                         "seen_mean": round(float(np.mean(seen)), 4),
                         "unseen_mean": round(float(np.mean(unseen)), 4),
                         "UPC": fam.get("UPC"), "by_family": fam,
                         "area_bins": bins, "by_goal": by_goal, "by_source_dataset": by_src}
        print(f"  ✅ {name:26s} @5%={results[name]['apde@5pct']} 面积 bins:",
              " ".join(f"{b['tpr@5pct']}" for b in bins), flush=True)
        del m; torch.cuda.empty_cache()
    json.dump(results, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"\n→ {a.out}")


if __name__ == "__main__":
    main()
