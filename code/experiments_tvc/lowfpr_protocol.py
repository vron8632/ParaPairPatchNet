#!/usr/bin/env python3
"""
Exp-G｜低误报跨集评测协议升级（主线的测量基础）

背景：此前 TPR@1% 的阈值取 APRICOT 1000 个负样本的 p99 → 只有 ~10 个样本决定阈值，
      导致 seed 间 std（最高 0.237）远大于模式间差异，任何 @1% 结论都不可靠。

本脚本做三件事：
1) 评测负样本扩到 21,000（APRICOT 4 源 1,000 + COCO-train2017 20,000，后者**只评测不入训练**，
   因此不存在"扩负样本训练 → 学到真实=干净捷径"的风险）。21138 个样本足够按校准窗定律
   N ≥ 20/q* 支撑 @0.1% 的声明（20×1/0.001 = 20,000）。
2) 低误报指标改为两个抗阈值噪声的量：
   - pAUC：FPR ≤ 1% 区间的部分 AUC（阈值无关）
   - TPR@FPR：**交叉拟合阈值**（K 折：用其余折的负样本定阈值，在本折负样本上测实际 FPR，
     在正样本上测 TPR）→ 报告 K 折上的 mean±std，反映阈值不确定度
3) 报每个负样本源的实测 FPR（检查 columbia/casia/inria/coco_val/coco_train 的源异质性），
   以及"名义 FPR 被守住的比例"（覆盖率）。

用法：
  python lowfpr_protocol.py                       # 自动发现 results/tvc/*_<fmode>.pt
  python lowfpr_protocol.py --ckpts expA_sem_s0_sem,expA_tex_s0_tex
输出：results/tvc/lowfpr_protocol.json
"""
import os, json, math, glob, argparse
os.environ.setdefault("HF_HOME", "/home/oyp/.cache/huggingface")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
WS = "/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard"
TVC = f"{WS}/results/tvc"; EVAL = f"{TVC}/unified_eval_set"; CACHE = f"{TVC}/feat_cache"
NEGS_IDX = "/home/oyp/fastdata/apde_negs/index.json"

import numpy as np
import torch, torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

ap = argparse.ArgumentParser()
ap.add_argument("--ckpts", default="", help="逗号分隔的 ckpt 名（不含 .pt）；留空=自动发现")
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--folds", type=int, default=10)
ap.add_argument("--out", default=f"{TVC}/lowfpr_protocol.json")
ap.add_argument("--n-coco", type=int, default=20000)
ap.add_argument("--save-scores", action="store_true", help="把分数导出到 results/tvc/sota_scores_<ckpt>.npz")
ap.add_argument("--neg-long-side", type=int, default=0,
                    help="重采样对照：把所有负样本按长边缩放到该尺寸（匹配正样本的重采样历史），0=不缩放")
a = ap.parse_args()

S3 = ["casia_au", "coco_val", "inria_noped"]


# ───────────────────────── 数据 ─────────────────────────
def build_rows():
    pv = [{"path": f"{EVAL}/pos/{f}", "label": 1, "src": "apricot_pos"}
          for f in sorted(os.listdir(f"{EVAL}/pos"))]
    nv = [{"path": f"{EVAL}/{s}/{f}", "label": 0, "src": s}
          for s in S3 + ["columbia_au"] for f in sorted(os.listdir(f"{EVAL}/{s}"))]
    coco = json.load(open(NEGS_IDX))[:a.n_coco]
    cn = [{"path": r["path"], "label": 0, "src": "coco_train"} for r in coco]
    return pv + nv + cn


class DS(Dataset):
    def __init__(s, rows, feats, size, tf, dummy=False):
        s.r, s.f, s.tf, s.size, s.dummy = rows, feats, tf, size, dummy
    def __len__(s): return len(s.r)
    def __getitem__(s, i):
        if s.dummy: return torch.zeros(3, 8, 8), s.f[i]
        try: im = Image.open(s.r[i]["path"]).convert("RGB")
        except Exception: im = Image.new("RGB", (s.size, s.size))
        if a.neg_long_side and s.r[i]["label"] == 0:
            w, h = im.size; sc = a.neg_long_side / max(w, h)
            if abs(sc - 1) > 1e-6:
                im = im.resize((max(1, int(round(w*sc))), max(1, int(round(h*sc)))), Image.BILINEAR)
        if im.size != (s.size, s.size): im = im.resize((s.size, s.size), Image.BILINEAR)
        return s.tf(im), s.f[i]


def sem_features(rows):
    """MobileCLIP2-S0 语义特征（带缓存）"""
    tag = "rs" + str(a.neg_long_side) if a.neg_long_side else "native"
    f = f"{CACHE}/lowfpr_sem_{len(rows)}_{tag}.npz"
    if os.path.exists(f):
        z = np.load(f)
        if len(z["X"]) == len(rows):
            print(f"  语义特征缓存命中 {f}", flush=True)
            return z["X"]
    import open_clip
    mc, _, pre = open_clip.create_model_and_transforms(
        "MobileCLIP2-S0", pretrained="dfndr2b", cache_dir="/home/oyp/.cache/huggingface")
    mc = mc.cuda().eval()

    class ImgDS(Dataset):
        def __init__(s, rows): s.r = rows
        def __len__(s): return len(s.r)
        def __getitem__(s, i):
            try: im = Image.open(s.r[i]["path"]).convert("RGB")
            except Exception: im = Image.new("RGB", (256, 256))
            if a.neg_long_side and s.r[i]["label"] == 0:
                w, h = im.size; sc = a.neg_long_side / max(w, h)
                if abs(sc - 1) > 1e-6:
                    im = im.resize((max(1, int(round(w*sc))), max(1, int(round(h*sc)))), Image.BILINEAR)
            return pre(im)

    X = []
    with torch.no_grad():
        dl = DataLoader(ImgDS(rows), batch_size=64, shuffle=False, num_workers=8)
        for i, b in enumerate(dl):
            X.append(mc.encode_image(b.cuda()).float().cpu().numpy())
            if (i + 1) % 60 == 0: print(f"    语义特征 {min((i+1)*64, len(rows))}/{len(rows)}", flush=True)
    X = np.concatenate(X, 0)
    X = np.stack(X).astype(np.float32)
    np.savez_compressed(f, X=X)
    return X


# ───────────────────────── 模型（与 fusion_ablation_evt.py 一致）─────────────────────────
class Net(nn.Module):
    def __init__(s, cdim, fmode):
        super().__init__()
        b = models.shufflenet_v2_x1_0(weights=None)
        td = b.fc.in_features; b.fc = nn.Identity(); s.tex = b
        s.fmode = fmode; s.cdim = cdim
        if fmode == "gate":
            s.gate = nn.Sequential(nn.LayerNorm(td + cdim), nn.Linear(td + cdim, 128), nn.ReLU(), nn.Linear(128, 1))
            s.head = nn.Sequential(nn.LayerNorm(td + cdim), nn.Linear(td + cdim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2))
        elif fmode == "attention":
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
        else:  # concat
            s.head = nn.Sequential(nn.LayerNorm(td + cdim), nn.Linear(td + cdim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2))

    def forward(s, x, cf):
        if s.fmode == "sem": return s.head(cf)
        t = s.tex(x)
        if s.fmode == "gate":
            g = torch.sigmoid(s.gate(torch.cat([t, cf], 1)))
            return s.head(torch.cat([g * t, (1 - g) * cf], 1))
        if s.fmode == "attention":
            att = torch.sigmoid((s.q(t) * s.k(cf)).sum(-1, keepdim=True) / math.sqrt(128))
            return s.head(att * s.v(cf))
        if s.fmode in ("moe", "moe_att"):
            w = torch.softmax(s.router(torch.cat([t, cf], 1)), 1)
            z = w[:, 0:1] * s.e1(t) + w[:, 1:2] * s.e2(cf)
            if s.fmode == "moe_att":
                att = torch.sigmoid((s.q(t) * s.k(cf)).sum(-1, keepdim=True) / math.sqrt(128))
                return s.head(att * z)
            return s.head(z)
        if s.fmode == "tex": return s.head(t)
        return s.head(torch.cat([t, cf], 1))


# ───────────────────────── 指标 ─────────────────────────
def pauc(fpr_grid, tpr_grid):
    """部分 AUC：FPR ≤ max(fpr_grid) 区间内 TPR 对 FPR 的归一化面积（阈值无关）"""
    return float(np.trapz(tpr_grid, fpr_grid) / fpr_grid[-1])


def roc_curves(sp, sn):
    thr = np.unique(np.concatenate([sn, sp]))
    thr = np.sort(thr)[::max(1, len(thr) // 4000)]
    tpr = np.array([(sp >= t).mean() for t in thr])
    fpr = np.array([(sn >= t).mean() for t in thr])
    o = np.argsort(fpr)
    return fpr[o], tpr[o]


def crossfit(sp, sn, folds, targets=(0.05, 0.01, 0.005, 0.001)):
    """K 折交叉拟合：用其余折的负样本定阈值，在本折上测实际 FPR，在全部正样本上测 TPR"""
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(sn))
    parts = np.array_split(idx, folds)
    out = {}
    for q in targets:
        tprs, fprs = [], []
        for k in range(folds):
            te = parts[k]; tr = np.concatenate([parts[j] for j in range(folds) if j != k])
            thr = float(np.percentile(sn[tr], 100 * (1 - q)))
            tprs.append(float((sp >= thr).mean()))
            fprs.append(float((sn[te] >= thr).mean()))
        out[f"@{q*100:g}pct"] = {"tpr_mean": round(float(np.mean(tprs)), 4),
                                 "tpr_std": round(float(np.std(tprs, ddof=1)), 4),
                                 "fpr_obs_mean": round(float(np.mean(fprs)), 4),
                                 "fpr_obs_std": round(float(np.std(fprs, ddof=1)), 4),
                                 "onside_rate": round(float(np.mean([f <= q * 1.25 for f in fprs])), 3)}
    return out


def main():
    rows = build_rows()
    npos = sum(r["label"] == 1 for r in rows)
    print(f"评测集：pos {npos} + neg {len(rows)-npos}（含 coco_train {a.n_coco}）", flush=True)
    X = sem_features(rows)
    srcs = np.array([r["src"] for r in rows])
    y = np.array([r["label"] for r in rows])

    # 发现 ckpt
    if a.ckpts:
        names = [c.strip() for c in a.ckpts.split(",") if c.strip()]
    else:
        names = [os.path.basename(p)[:-3] for p in sorted(glob.glob(f"{TVC}/*_*.pt"))]
        names = [n for n in names if n.split("_")[-1] in ("tex", "sem", "concat", "gate", "attention", "moe", "moe_att")]
    print(f"待评测 ckpt：{len(names)} 个", flush=True)

    results = {}
    for name in names:
        p = f"{TVC}/{name}.pt"
        if not os.path.exists(p):
            print(f"  ⚠️ 缺 {p}", flush=True); continue
        ck = torch.load(p, map_location="cpu", weights_only=False)
        fmode, size = ck["fmode"], ck["size"]
        m = Net(ck["cdim"], fmode).cuda(); m.load_state_dict(ck["state_dict"]); m.eval()
        tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        sc = []
        with torch.no_grad():
            dl = DataLoader(DS(rows, X, size, tf, dummy=(fmode == "sem")), batch_size=a.bs,
                            shuffle=False, num_workers=0)
            for x, cf in dl:
                sc.extend(torch.softmax(m(x.cuda(), cf.cuda()), 1)[:, 1].float().cpu().numpy().tolist())
        sc = np.array(sc)
        sp, sn = sc[y == 1], sc[y == 0]
        fpr, tpr = roc_curves(sp, sn)
        r = {"fmode": fmode, "size": size, "lb_lambda": ck.get("lb_lambda", 0),
             "pAUC@1pct": round(pauc(fpr[fpr <= 0.01], tpr[fpr <= 0.01]), 4) if (fpr <= 0.01).sum() > 2 else None,
             "pAUC@5pct": round(pauc(fpr[fpr <= 0.05], tpr[fpr <= 0.05]), 4) if (fpr <= 0.05).sum() > 2 else None,
             "crossfit": crossfit(sp, sn, a.folds),
             "auc": round(float(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(y, sc)), 4),
             "fpr_by_source@1pct": {s: round(float((sn[srcs[y == 0] == s] >= np.percentile(sn, 99)).mean()), 4)
                                    for s in sorted(set(srcs[y == 0]))},
             "n_pos": int(len(sp)), "n_neg": int(len(sn)),
             "pos_score_mean": round(float(sp.mean()), 4), "neg_score_mean": round(float(sn.mean()), 4)}
        results[name] = r
        if a.save_scores:
            sfx = "_rs" + str(a.neg_long_side) if a.neg_long_side else ""
            np.savez_compressed(f"{TVC}/sota_scores_{name}{sfx}.npz", scores=sc.astype(np.float32),
                                src=srcs, label=y, path=np.array([q["path"] for q in rows]))
        cf1 = r["crossfit"]["@1pct"]
        print(f"  ✅ {name:26s} pAUC@1%={r['pAUC@1pct']}  TPR@1%={cf1['tpr_mean']}±{cf1['tpr_std']} "
              f"(实测FPR {cf1['fpr_obs_mean']}±{cf1['fpr_obs_std']})  AUC={r['auc']}", flush=True)
        del m
        torch.cuda.empty_cache()

    json.dump(results, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"\n→ {a.out}")


if __name__ == "__main__":
    main()
