#!/usr/bin/env python3
"""
Exp-C｜路由干预实验（因果探针）

问题：moe_att 的收益来自"路由器按样本选专家"，还是来自"两个专家被共同训练 + 融合读出"？

做法：对已训练好的 ckpt，冻结全部参数，只在评测时替换路由权重 w（不改 head、不改 attention）：
  learned     原始学习到的 w
  fixed_tex   w=[1,0]（强制走纹理专家）
  fixed_sem   w=[0,1]（强制走语义专家）
  fixed_avg   w=[0.5,0.5]（强制均衡）
  shuffled    把 learned w 在样本间随机打乱（保持负载分布、破坏"样本↔专家"对应）

判据（事先声明）：
  - 若 learned ≫ shuffled 且 learned ≫ fixed_*  → 路由的样本级选择是收益来源（支持"自适应路由"）
  - 若 learned ≈ shuffled 且 fixed_avg 与 learned 相当 → 收益来自"专家共同训练 + 均衡读出"，而非路由（支持"负载均衡"解释，且与 LB 正则实验一致：λ 抹掉路由信息后性能下降）

输出：results/tvc/route_intervention.json
"""
import os, json, math, argparse
os.environ.setdefault("HF_HOME", "/home/oyp/.cache/huggingface")
WS = "/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard"
TVC = f"{WS}/results/tvc"; EVAL = f"{TVC}/unified_eval_set"; CACHE = f"{TVC}/feat_cache"
TEST_JPG = "/home/oyp/fastdata/apde_test_jpg"
SEEN = {"advpatch", "DM-NAP", "GNAP", "TCA", "TCEGA", "tsea", "tsea-mim", "tsea-pgd"}
UNSEEN = {"AA", "AdvCloak", "AdvSticker", "AdvTshirt", "UPC"}

import numpy as np
import torch, torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

ap = argparse.ArgumentParser()
ap.add_argument("--ckpts", default="mech_moeatt_s0,fab_moeatt_s2,lbreg_0.1_s0,lbreg_0.5_s0,lbreg_0.5_s1")
ap.add_argument("--out", default=f"{TVC}/route_intervention.json")
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--n-shuffle", type=int, default=5, help="shuffled 变体的随机重复次数")
a = ap.parse_args()


class Net(nn.Module):
    """与 fusion_ablation_evt.py 中 moe_att 完全一致的结构"""
    def __init__(s, cdim):
        super().__init__()
        b = models.shufflenet_v2_x1_0(weights=None)
        td = b.fc.in_features; b.fc = nn.Identity(); s.tex = b
        s.e1 = nn.Sequential(nn.LayerNorm(td), nn.Linear(td, 256), nn.ReLU())
        s.e2 = nn.Sequential(nn.LayerNorm(cdim), nn.Linear(cdim, 256), nn.ReLU())
        s.router = nn.Sequential(nn.LayerNorm(td + cdim), nn.Linear(td + cdim, 64), nn.ReLU(), nn.Linear(64, 2))
        s.q = nn.Linear(td, 128); s.k = nn.Linear(cdim, 128)
        s.head = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2))

    def parts(s, x, cf):
        t = s.tex(x)
        w = torch.softmax(s.router(torch.cat([t, cf], 1)), 1)
        e1, e2 = s.e1(t), s.e2(cf)
        att = torch.sigmoid((s.q(t) * s.k(cf)).sum(-1, keepdim=True) / math.sqrt(128))
        return att, e1, e2, w

    def logits(s, att, e1, e2, w):
        z = w[:, 0:1] * e1 + w[:, 1:2] * e2
        return s.head(att * z)


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


def metrics(sp, sn, pos_m, pA, nA):
    thr = float(np.percentile(sn, 95))
    fam = {f: round(float((sp[pos_m == f] >= thr).mean()), 4) for f in sorted(set(pos_m))}
    seen = [v for k, v in fam.items() if k in SEEN]
    unseen = [v for k, v in fam.items() if k in UNSEEN]
    out = {"apde@5pct": round(float((sp >= thr).mean()), 4),
           "seen_mean": round(float(np.mean(seen)), 4),
           "unseen_mean": round(float(np.mean(unseen)), 4),
           "UPC": fam.get("UPC"), "by_family": fam,
           "apricot@5pct": round(float((pA >= np.percentile(nA, 95)).mean()), 4),
           "apricot@1pct": round(float((pA >= np.percentile(nA, 99)).mean()), 4)}
    return out


def main():
    te = json.load(open(f"{TEST_JPG}/index.json"))
    pv = [{"path": f"{EVAL}/pos/{f}", "label": 1} for f in sorted(os.listdir(f"{EVAL}/pos"))]
    S3 = ["casia_au", "coco_val", "inria_noped"]
    nv = [{"path": f"{EVAL}/{s}/{f}", "label": 0} for s in S3 + ["columbia_au"]
          for f in sorted(os.listdir(f"{EVAL}/{s}"))]
    rowsA = pv + nv
    Xa = np.load(f"{CACHE}/fab_apricot_1138.npz")["X"]
    assert len(Xa) == len(rowsA)

    ckpt_names = [c.strip() for c in a.ckpts.split(",") if c.strip()]
    results = {}
    for name in ckpt_names:
        p = f"{TVC}/{name}_moe_att.pt"
        if not os.path.exists(p):
            print(f"  ⚠️ 缺少 ckpt: {p}", flush=True); continue
        ck = torch.load(p, map_location="cpu", weights_only=False)
        size = ck["size"]
        m = Net(ck["cdim"]).cuda(); m.load_state_dict(ck["state_dict"]); m.eval()
        print(f"\n───── {name} (size={size}, lb={ck.get('lb_lambda', 0)}) ─────", flush=True)

        # APDE：纹理前向一次，特征已缓存
        Xte = np.load(f"{CACHE}/fab_apde_14000.npz")["X"]
        assert len(Xte) == len(te)
        parts = []
        with torch.no_grad():
            dl = DataLoader(DS(te, Xte, size), batch_size=a.bs, shuffle=False, num_workers=0)
            for x, cf in dl:
                parts.append(m.parts(x.cuda(), cf.cuda()))
        att = torch.cat([q[0] for q in parts]); e1 = torch.cat([q[1] for q in parts])
        e2 = torch.cat([q[2] for q in parts]); w = torch.cat([q[3] for q in parts])

        partsA = []
        with torch.no_grad():
            dl = DataLoader(DS(rowsA, Xa, size), batch_size=a.bs, shuffle=False, num_workers=0)
            for x, cf in dl:
                partsA.append(m.parts(x.cuda(), cf.cuda()))
        attA = torch.cat([q[0] for q in partsA]); e1A = torch.cat([q[1] for q in partsA])
        e2A = torch.cat([q[2] for q in partsA]); wA = torch.cat([q[3] for q in partsA])

        pos_m = np.array([r["method"] for r in te if r["label"] == 1])
        pos_idx = torch.tensor([i for i, r in enumerate(te) if r["label"] == 1])
        neg_idx = torch.tensor([i for i, r in enumerate(te) if r["label"] == 0])

        variants = {"learned": (w, wA)}
        n = w.shape[0]
        dev = w.device
        variants["fixed_tex"] = (torch.tensor([[1., 0.]], device=dev).repeat(n, 1), torch.tensor([[1., 0.]], device=dev).repeat(wA.shape[0], 1))
        variants["fixed_sem"] = (torch.tensor([[0., 1.]], device=dev).repeat(n, 1), torch.tensor([[0., 1.]], device=dev).repeat(wA.shape[0], 1))
        variants["fixed_avg"] = (torch.tensor([[.5, .5]], device=dev).repeat(n, 1), torch.tensor([[.5, .5]], device=dev).repeat(wA.shape[0], 1))
        rng = np.random.default_rng(0)
        for k in range(a.n_shuffle):
            i1 = rng.permutation(n); i2 = rng.permutation(wA.shape[0])
            variants[f"shuffled{k}"] = (w[i1], wA[i2])

        res = {}
        with torch.no_grad():
            for vname, (wv, wvA) in variants.items():
                s = torch.softmax(m.logits(att, e1, e2, wv), 1)[:, 1].cpu().numpy()
                sA = torch.softmax(m.logits(attA, e1A, e2A, wvA), 1)[:, 1].cpu().numpy()
                r = metrics(s[pos_idx.numpy()], s[neg_idx.numpy()], pos_m, sA[:len(pv)], sA[len(pv):])
                res[vname] = r
        # 汇总 shuffled 的均值
        sh = [res[k] for k in res if k.startswith("shuffled")]
        res["_shuffled_mean"] = {k: round(float(np.mean([x[k] for x in sh])), 4)
                                 for k in ("apde@5pct", "seen_mean", "unseen_mean", "UPC", "apricot@5pct", "apricot@1pct")}
        res["_meta"] = {"size": size, "lb_lambda": ck.get("lb_lambda", 0),
                        "load_expert1": round(float(w[:, 0].mean()), 4),
                        "route_std_across_samples": round(float(w[:, 1].std()), 4)}
        results[name] = res
        print(f"  learned  : APDE@5%={res['learned']['apde@5pct']} unseen={res['learned']['unseen_mean']} "
              f"UPC={res['learned']['UPC']} APRICOT@5%={res['learned']['apricot@5pct']}", flush=True)
        for v in ("fixed_tex", "fixed_sem", "fixed_avg", "_shuffled_mean"):
            print(f"  {v:14s}: APDE@5%={res[v]['apde@5pct']} unseen={res[v]['unseen_mean']} "
                  f"UPC={res[v]['UPC']} APRICOT@5%={res[v]['apricot@5pct']}", flush=True)
        del m, att, e1, e2, w, attA, e1A, e2A, wA
        torch.cuda.empty_cache()

    json.dump(results, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"\n→ {a.out}")


if __name__ == "__main__":
    main()
