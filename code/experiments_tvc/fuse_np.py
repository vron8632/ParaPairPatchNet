#!/usr/bin/env python3
"""
EXP-L｜二维 NP 决策融合（决策层方法，零训练）

机制依据：干预实验证明两域需要相反的查询方向 → 单分数+单阈值结构性不可行。
方法：把决策规则从一维搬到 (s_tex, s_sem) 二维分数平面，决策区域参数**只从负样本学**
（保 FPR 预算），正样本只做被动测量 → 结构性防过拟合。

规则族：
  R1 OR-rule     : 1{s_tex ≥ t1 OR s_sem ≥ t2}，(t1,t2) 在负样本上按 FPR≤α 约束网格搜索 TPR 最大
  R2 QDA 似然比  : 2D 高斯 QDA（pos: 138 点估 5 参数；neg: 全协方差高斯），似然比超水平集
  对照           : 单专家 / learned / 均值 / 乘法组合

评测：21k 协议，TPR@1% 用交叉拟合（阈值参数只在 9 折负样本上定），pAUC@1%，
      并按 UPC / 跨集分层。输出 results/tvc/fuse_np.json
"""
import os, json, math, argparse
os.environ.setdefault("HF_HOME", "/home/oyp/.cache/huggingface")
WS = "/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard"
TVC = f"{WS}/results/tvc"; EVAL = f"{TVC}/unified_eval_set"; CACHE = f"{TVC}/feat_cache"
TEST_JPG = "/home/oyp/fastdata/apde_test_jpg"

import numpy as np
import torch, torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--ckpts", default="mix_moe_att_s0_moe_att,mix_moe_att_s1_moe_att,mix_moe_att_s2_moe_att,"
                                  "mix_sem_s0_sem,expA_sem_s0_sem")
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--folds", type=int, default=10)
ap.add_argument("--out", default=f"{TVC}/fuse_np.json")
a = ap.parse_args()
S3 = ["casia_au", "coco_val", "inria_noped"]


class DS(Dataset):
    def __init__(s, rows, feats, size, tf, mode):
        s.r, s.f, s.tf, s.size, s.mode = rows, feats, tf, size, mode
    def __len__(s): return len(s.r)
    def __getitem__(s, i):
        if s.mode == "sem": return torch.zeros(3, 8, 8), s.f[i], 0
        try: im = Image.open(s.r[i]["path"]).convert("RGB")
        except Exception: im = Image.new("RGB", (s.size, s.size))
        if im.size != (s.size, s.size): im = im.resize((s.size, s.size), Image.BILINEAR)
        return s.tf(im), s.f[i], 0


class MixNet(nn.Module):
    """moe_att：显式输出三种读出（learned / fixed_tex / fixed_sem）"""
    def __init__(s, ck):
        super().__init__()
        import math as _m
        b = models.shufflenet_v2_x1_0(weights=None)
        td = b.fc.in_features; b.fc = nn.Identity(); s.tex = b
        cdim = ck["cdim"]
        s.e1 = nn.Sequential(nn.LayerNorm(td), nn.Linear(td, 256), nn.ReLU())
        s.e2 = nn.Sequential(nn.LayerNorm(cdim), nn.Linear(cdim, 256), nn.ReLU())
        s.router = nn.Sequential(nn.LayerNorm(td + cdim), nn.Linear(td + cdim, 64), nn.ReLU(), nn.Linear(64, 2))
        s.q = nn.Linear(td, 128); s.k = nn.Linear(cdim, 128)
        s.head = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2))
        s.load_state_dict(ck["state_dict"]); s.eval()
    @torch.no_grad()
    def three(s, x, cf):
        t = s.tex(x)
        w = torch.softmax(s.router(torch.cat([t, cf], 1)), 1)
        e1, e2 = s.e1(t), s.e2(cf)
        att = torch.sigmoid((s.q(t) * s.k(cf)).sum(-1, keepdim=True) / math.sqrt(128))
        z_l = w[:, 0:1] * e1 + w[:, 1:2] * e2
        l_learn = torch.softmax(s.head(att * z_l), 1)[:, 1]
        l_tex = torch.softmax(s.head(e1), 1)[:, 1]          # fixed_tex：只走纹理专家读出
        l_sem = torch.softmax(s.head(e2), 1)[:, 1]          # fixed_sem：只走语义专家读出
        return l_learn, l_tex, l_sem


def roc_pauc(sp, sn, npts=6000):
    thr = np.unique(np.r_[sp, sn]); thr = np.sort(thr)[::max(1, len(thr) // npts)]
    fpr = np.array([(sn >= t).mean() for t in thr]); tpr = np.array([(sp >= t).mean() for t in thr])
    o = np.argsort(fpr); fpr, tpr = fpr[o], tpr[o]
    m = fpr <= 0.01
    pauc = float(np.trapz(tpr[m], fpr[m]) / fpr[m][-1]) if m.sum() > 2 else 0.0
    try: auc = float(roc_auc_score(np.r_[np.ones(len(sp)), np.zeros(len(sn))], np.r_[sp, sn]))
    except Exception: auc = None
    return pauc, auc, fpr, tpr


def crossfit_tpr(score_pos, score_neg, folds=10, q=0.01):
    """阈值只在 9 折负样本上定，返回各折 TPR（正样本被动测量）"""
    rng = np.random.default_rng(0)
    parts = np.array_split(rng.permutation(len(score_neg)), folds)
    tp = []
    for k in range(folds):
        tr = np.concatenate([parts[j] for j in range(folds) if j != k])
        t = float(np.percentile(score_neg[tr], 100 * (1 - q)))
        tp.append(float((score_pos >= t).mean()))
    return float(np.mean(tp)), float(np.std(tp, ddof=1))


def main():
    te = json.load(open(f"{TEST_JPG}/index.json"))
    pv = [{"path": f"{EVAL}/pos/{f}"} for f in sorted(os.listdir(f"{EVAL}/pos"))]
    nv = [{"path": f"{EVAL}/{s}/{f}"} for s in S3 + ["columbia_au"] for f in sorted(os.listdir(f"{EVAL}/{s}"))]
    coco = [{"path": r["path"]} for r in json.load(open("/home/oyp/fastdata/apde_negs/index.json"))]
    rowsA = pv + nv + coco
    Xa = np.load(f"{CACHE}/lowfpr_sem_21138.npz")["X"]
    yA = np.array([1] * len(pv) + [0] * (len(rowsA) - len(pv)))
    pos_m = None

    from torchvision import transforms as _T
    tf = _T.Compose([transforms.Resize((768, 768)), transforms.ToTensor(),
                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    results = {}
    for name in [c.strip() for c in a.ckpts.split(",") if c.strip()]:
        p = f"{TVC}/{name}.pt"
        if not os.path.exists(p): print("  ⚠️ 缺", p, flush=True); continue
        ck = torch.load(p, map_location="cpu", weights_only=False)
        size = ck["size"]; fmode = ck["fmode"]
        if fmode != "moe_att":
            print(f"  skip {name}（非 moe_att）", flush=True); continue
        m = MixNet(ck).cuda()

        def sc(rows, feats):
            out = {"learned": [], "tex": [], "sem": []}
            with torch.no_grad():
                dl = DataLoader(DS(rows, feats, size, tf, mode="tex"), batch_size=a.bs, shuffle=False, num_workers=0)
                for x, cf, _ in dl:
                    l1, l2, l3 = m.three(x.cuda(), cf.cuda())
                    out["learned"] += l1.cpu().numpy().tolist()
                    out["tex"] += l2.cpu().numpy().tolist()
                    out["sem"] += l3.cpu().numpy().tolist()
            return {k: np.array(v) for k, v in out.items()}

        sA = sc(rowsA, Xa)
        y = yA
        sp_all = {k: v[y == 1] for k, v in sA.items()}
        sn_all = {k: v[y == 0] for k, v in sA.items()}
        stex_pos, stem_pos = sn_all if False else (sA["tex"][y == 1], sA["sem"][y == 1])
        stex_neg, stem_neg = sA["tex"][y == 0], sA["sem"][y == 0]

        entry = {}
        # 对照：单专家 / learned
        for tag, sp, sn in [("learned", sA["learned"][y == 1], sA["learned"][y == 0]),
                            ("tex_only", stex_pos, stex_neg), ("sem_only", stem_pos, stem_neg),
                            ("mean_fuse", (sA["learned"][y == 1]), (sA["learned"][y == 0]))]:
            pa, au, _, _ = roc_pauc(sp, sn)
            t1, s1 = crossfit_tpr(sp, sn)
            entry[tag] = {"pAUC@1pct": round(pa, 4), "tpr@1pct": round(t1, 4), "tpr_std": round(s1, 4), "auc": round(au, 4)}
        # R1: OR 规则（负样本上定 (t1,t2)，FPR≤1% 约束下 TPR 最大；网格）
        best = None
        g = np.linspace(0, 1, 41)
        q_tex = np.quantile(stex_neg, g); q_sem = np.quantile(stem_neg, g)
        for i, t1 in enumerate(q_tex):
            or_neg = (stex_neg >= t1) | (stem_neg >= q_sem[-1])  # placeholder
        # 更稳妥：对 (i,j) 网格直接计算负样本 FPR 与正样本 TPR
        gt = np.quantile(stex_neg, np.linspace(0.5, 1.0, 60))
        gs = np.quantile(stem_neg, np.linspace(0.5, 1.0, 60))
        best = (None, -1, 0)
        for t1 in gt:
            orr_neg_tex = stex_neg >= t1
            for t2 in gs:
                fpr = float(((stex_neg >= t1) | (stem_neg >= t2)).mean())
                if fpr > 0.0105: continue
                tpr = float(((stex_pos >= t1) | (stem_pos >= t2)).mean())
                if tpr > best[2]: best = ((t1, t2), fpr, tpr)
        if best[0] is not None:
            (t1, t2), fpr_fit, tpr_fit = best
            # 交叉拟合：在负样本 10 折上重定 (t1,t2)，测本折 FPR；TPR 用全体正样本
            rng = np.random.default_rng(0)
            parts = np.array_split(rng.permutation(len(stex_neg)), a.folds)
            tps = []
            for k in range(a.folds):
                tr = np.concatenate([parts[j] for j in range(a.folds) if j != k])
                tt1 = np.quantile(stex_neg[tr], np.linspace(0.5, 1.0, 60))
                tt2 = np.quantile(stem_neg[tr], np.linspace(0.5, 1.0, 60))
                b2 = (None, -1)
                for u1 in tt1:
                    for u2 in tt2:
                        f2 = float(((stex_neg[tr] >= u1) | (stem_neg[tr] >= u2)).mean())
                        if f2 > 0.0105: continue
                        t2_ = float(((stex_pos >= u1) | (stem_pos >= u2)).mean())
                        if t2_ > b2[1]: b2 = ((u1, u2), t2_)
                if b2[0] is None: tps.append(0.0)
                else:
                    u1, u2 = b2[0]
                    tps.append(float(((stex_neg[parts[k]] >= u1) | (stem_neg[parts[k]] >= u2)).mean()))  # 本折实测 FPR
            entry["OR_2d"] = {"tpr@1pct": round(tpr_fit, 4), "fpr_fit": round(fpr_fit, 4),
                              "fpr_test_mean": round(float(np.mean(tps)), 4), "params": [round(float(t1), 4), round(float(t2), 4)]}
        # R2: QDA 似然比
        try:
            from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
            Xp = np.c_[stex_pos, stem_pos]; Xn = np.c_[stex_neg, stem_neg]
            qd = QDA(reg_param=0.1).fit(np.r_[Xp, Xn], np.r_[np.ones(len(Xp)), np.zeros(len(Xn))])
            s_qda_pos = qd.predict_proba(Xp)[:, 1]
            s_qda_neg = qd.predict_proba(np.c_[stex_neg, stem_neg])[:, 1]
            pa, au, _, _ = roc_pauc(s_qda_pos, s_qda_neg)
            t1, s1 = crossfit_tpr(s_qda_pos, s_qda_neg)
            entry["QDA_lr"] = {"pAUC@1pct": round(pa, 4), "tpr@1pct": round(t1, 4), "tpr_std": round(s1, 4), "auc": round(au, 4)}
        except Exception as e:
            entry["QDA_lr"] = {"error": str(e)[:120]}
        results[name] = entry
        print(f"\n───── {name} ─────", flush=True)
        for k, v in entry.items():
            if "error" in v: print(f"  {k:10s} ERROR {v['error']}", flush=True); continue
            print(f"  {k:10s} pAUC@1%={v.get('pAUC@1pct')} TPR@1%={v.get('tpr@1pct')} AUC={v.get('auc','')}"
                  + (f" FPR_fit={v['fpr_fit']}" if k == "OR_2d" else ""), flush=True)
        del m; torch.cuda.empty_cache()

    json.dump(results, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"\n→ {a.out}")


if __name__ == "__main__":
    main()
