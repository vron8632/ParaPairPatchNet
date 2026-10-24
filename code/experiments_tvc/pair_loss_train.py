#!/usr/bin/env python3
"""
EXP-J｜背景对照对比损失（损失/训练方式层面的方法组件）

机制依据（EXP-H 成分分离）：只加补丁崩（0.26）、只加干净无效（0.66）、成对才有效（0.96）
→ 结构先验：「补丁性」只有在同背景对照下才可学习。本脚本把它形式化为损失：

    L = CE + λ · L_inv ,   L_inv = mean_b [ 1 − cos(Δ_b, Δ̄) ]
    Δ_b = f(x⁺_b) − f(x⁻_b)  （同背景合成对的 256 维表示差）

即「补丁效应的跨背景不变性」。与此前失败的表示层一致性不同：不跨域拉近分数，
只在**同背景内**建立对照方向，且不约束跨域分数相等。

训练数据：APDE 12k 子集 + lora 1500 + 合成同背景对 4000（正负都计入）
评测：APDE test 14000（族级 + UPC + seen/unseen），21k 协议由 lowfpr_protocol 另跑
"""
import os, sys, json, math, argparse, random
os.environ.setdefault("HF_HOME", "/home/oyp/.cache/huggingface")
WS = "/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard"
TVC = f"{WS}/results/tvc"; CACHE = f"{TVC}/feat_cache"
TRAIN_JPG = "/home/oyp/fastdata/apde_train_jpg"; TEST_JPG = "/home/oyp/fastdata/apde_test_jpg"
PSYN = "/home/oyp/fastdata/paired_syn"
SEEN = {"advpatch", "DM-NAP", "GNAP", "TCA", "TCEGA", "tsea", "tsea-mim", "tsea-pgd"}
UNSEEN = {"AA", "AdvCloak", "AdvSticker", "AdvTshirt", "UPC"}

import numpy as np
import torch, torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

ap = argparse.ArgumentParser()
ap.add_argument("--size", type=int, default=768)
ap.add_argument("--epochs", type=int, default=8)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--limit-train", type=int, default=12000)
ap.add_argument("--pair-batch", type=int, default=16)
ap.add_argument("--lambda-pair", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--fusions", default="sem,moe_att")
ap.add_argument("--num-workers", type=int, default=0)
ap.add_argument("--out", default=f"{TVC}/pairloss")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)


def build_rows(seed):
    tr = json.load(open(f"{TRAIN_JPG}/index.json"))
    pos = [r for r in tr if r["label"] == 1]; neg = [r for r in tr if r["label"] == 0]
    rng = random.Random(seed); rng.shuffle(pos)
    sub = pos[: a.limit_train // 2] + neg
    lora = [r for r in json.load(open(f"{TVC}/lora_data/index.json")) if r["split"] == "train"]
    sub += [{"path": r["path"], "label": r["label"]} for r in lora]
    pairs = json.load(open(f"{PSYN}/index.json"))
    for p in pairs:
        sub += [{"path": p["path_pos"], "label": 1}, {"path": p["path_neg"], "label": 0}]
    return sub, pairs


class DS(Dataset):
    def __init__(s, rows, feats, size, tf, dummy=False):
        s.r, s.f, s.y, s.tf, s.size, s.dummy = rows, feats, np.array([r["label"] for r in rows]), tf, size, dummy
    def __len__(s): return len(s.r)
    def __getitem__(s, i):
        if s.dummy: return torch.zeros(3, 8, 8), s.f[i], 0
        try: im = Image.open(s.r[i]["path"]).convert("RGB")
        except Exception: im = Image.new("RGB", (s.size, s.size))
        if im.size != (s.size, s.size): im = im.resize((s.size, s.size), Image.BILINEAR)
        return s.tf(im), s.f[i], int(s.y[i])


class PairDS(Dataset):
    """同背景合成对：返回 (x+, x−, cf+, cf−)"""
    def __init__(s, pairs, size, tf, F):
        s.p, s.size, s.tf, s.F = pairs, size, tf, F
    def __len__(s): return len(s.p)
    def __getitem__(s, i):
        p = s.p[i]; out = []
        for key in ("path_pos", "path_neg"):
            try: im = Image.open(p[key]).convert("RGB")
            except Exception: im = Image.new("RGB", (s.size, s.size))
            if im.size != (s.size, s.size): im = im.resize((s.size, s.size), Image.BILINEAR)
            out.append(s.tf(im))
        return out[0], out[1], s.F[2 * i], s.F[2 * i + 1]


def clip_feats(rows, tag):
    import hashlib
    sig = hashlib.md5("\n".join(sorted(r["path"] for r in rows)).encode()).hexdigest()[:8]
    f = f"{CACHE}/pl_{tag}_{len(rows)}_{sig}.npz"   # 内容指纹，避免不同 seed 的等长子集撞车
    if os.path.exists(f):
        z = np.load(f)
        if len(z["X"]) == len(rows): return z["X"]
    import open_clip
    mc, _, pre = open_clip.create_model_and_transforms("MobileCLIP2-S0", pretrained="dfndr2b",
                                                       cache_dir="/home/oyp/.cache/huggingface")
    mc = mc.cuda().eval()
    X = []
    with torch.no_grad():
        for i, r in enumerate(rows):
            try: im = Image.open(r["path"]).convert("RGB")
            except Exception: im = Image.new("RGB", (256, 256))
            X.append(mc.encode_image(pre(im).unsqueeze(0).cuda())[0].float().cpu().numpy())
            if (i + 1) % 4000 == 0: print(f"    特征[{tag}] {i+1}/{len(rows)}", flush=True)
    X = np.stack(X).astype(np.float32); np.savez_compressed(f, X=X); return X


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
            if fmode == "moe_att": s.q = nn.Linear(td, 128); s.k = nn.Linear(cdim, 128)
            s.head = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2))
        elif fmode == "tex":
            s.head = nn.Sequential(nn.LayerNorm(td), nn.Linear(td, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2))
        elif fmode == "sem":
            s.head = nn.Sequential(nn.LayerNorm(cdim), nn.Linear(cdim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2))
        else:
            s.head = nn.Sequential(nn.LayerNorm(td + cdim), nn.Linear(td + cdim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2))
    def feats_raw(s, x, cf):
        """进入 head 前的表示"""
        if s.fmode == "sem": return cf
        t = s.tex(x)
        if s.fmode in ("moe", "moe_att"):
            w = torch.softmax(s.router(torch.cat([t, cf], 1)), 1)
            z = w[:, 0:1] * s.e1(t) + w[:, 1:2] * s.e2(cf)
            if s.fmode == "moe_att":
                att = torch.sigmoid((s.q(t) * s.k(cf)).sum(-1, keepdim=True) / math.sqrt(128))
                z = att * z
            return z
        if s.fmode == "tex": return t
        return torch.cat([t, cf], 1)
    def pen(s, x, cf):
        """256 维倒数第二层表示（head 去掉末层，只应用一次）"""
        z = s.feats_raw(x, cf)
        for m in list(s.head.children())[:-1]: z = m(z)
        return z
    def forward(s, x, cf): return s.head(s.feats_raw(x, cf))


def main():
    sub, pairs = build_rows(a.seed)
    print(f"训练行 {len(sub)}（合成对 {len(pairs)} 对）", flush=True)
    X = clip_feats(sub, "main")
    Fp = clip_feats([{"path": p["path_pos"]} for p in pairs] + [{"path": p["path_neg"]} for p in pairs], "pairs")
    tf = transforms.Compose([transforms.Resize((a.size, a.size)), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    te = json.load(open(f"{TEST_JPG}/index.json"))
    Xte = np.load(f"{CACHE}/fab_apde_14000.npz")["X"]
    pos_m = np.array([r["method"] for r in te if r["label"] == 1])
    yte = np.array([r["label"] for r in te])

    for fmode in [x.strip() for x in a.fusions.split(",")]:
        torch.manual_seed(a.seed); np.random.seed(a.seed)
        m = Net(X.shape[1], fmode).cuda()
        opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
        ce = nn.CrossEntropyLoss()
        ds = DS(sub, X, a.size, tf, dummy=(fmode == "sem"))
        dl = DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=a.num_workers, drop_last=True)
        pdl = DataLoader(PairDS(pairs, a.size, tf, Fp), batch_size=a.pair_batch,
                         shuffle=True, num_workers=a.num_workers, drop_last=True)
        piter = iter(pdl)
        for ep in range(a.epochs):
            m.train(); tot = cnt = 0; inv_tot = 0
            for x, cf, y in dl:
                x, cf, y = x.cuda(non_blocking=True), cf.cuda(non_blocking=True), y.cuda(non_blocking=True)
                logits = m(x, cf)
                loss = ce(logits, y)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss); cnt += 1
                # pair step（背景对照）
                try: pxp, pxn, cfp, cfn = next(piter)
                except StopIteration:
                    piter = iter(pdl); pxp, pxn, cfp, cfn = next(piter)
                pxp, pxn, cfp, cfn = pxp.cuda(), pxn.cuda(), cfp.cuda(), cfn.cuda()
                dpos = m.pen(pxp, cfp); dneg = m.pen(pxn, cfn)
                D = dpos - dneg                                   # [B,256] 补丁效应方向
                Dbar = D.mean(0, keepdim=True)
                cosv = torch.nn.functional.cosine_similarity(D, Dbar.expand_as(D), dim=1)
                L_inv = (1 - cosv).mean()
                loss2 = ce(m(pxp, cfp), torch.ones(len(pxp), dtype=torch.long, device="cuda")) + \
                        ce(m(pxn, cfn), torch.zeros(len(pxn), dtype=torch.long, device="cuda"))
                total = loss2 + a.lambda_pair * L_inv
                opt.zero_grad(); total.backward(); opt.step()
                inv_tot += float(L_inv)
            sch.step()
            print(f"  [{fmode}] ep{ep+1}/{a.epochs} CE={tot/max(cnt,1):.4f} L_inv={inv_tot/max(cnt,1):.4f}", flush=True)
        # 评测 APDE
        m.eval(); sc = []
        dse = DS(te, Xte, size=a.size, tf=tf, dummy=(fmode == "sem"))
        with torch.no_grad():
            for x, cf, _ in DataLoader(dse, batch_size=64, shuffle=False, num_workers=0):
                sc.extend(torch.softmax(m(x.cuda(), cf.cuda()), 1)[:, 1].float().cpu().numpy().tolist())
        sc = np.array(sc)
        sp, sn = sc[yte == 1], sc[yte == 0]
        thr5 = float(np.percentile(sn, 95))
        fam = {f: round(float((sp[pos_m == f] >= thr5).mean()), 4) for f in sorted(set(pos_m))}
        seen = [v for k, v in fam.items() if k in SEEN]; unseen = [v for k, v in fam.items() if k in UNSEEN]
        res = {"fmode": fmode, "lambda_pair": a.lambda_pair, "seed": a.seed,
               "apde@5pct": round(float((sp >= thr5).mean()), 4),
               "seen_mean": round(float(np.mean(seen)), 4), "unseen_mean": round(float(np.mean(unseen)), 4),
               "UPC": fam.get("UPC"), "by_family": fam}
        out = f"{a.out}/pair_{fmode}_s{a.seed}"
        json.dump(res, open(out + ".json", "w"), indent=1, ensure_ascii=False)
        torch.save({"state_dict": m.state_dict(), "cdim": X.shape[1], "size": a.size, "fmode": fmode,
                    "lambda_pair": a.lambda_pair}, out + ".pt")
        print(f"  ✅ {fmode}: 未见={res['unseen_mean']} UPC={res['UPC']} APDE@5%={res['apde@5pct']}", flush=True)
        del m; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
