#!/usr/bin/env python3
"""
门控双分支 + 无标签族 DRO
==========================
A) 样本级门控：g = sigmoid(MLP([tex,clip]))；z = g*tex + (1-g)*clip
   - g 只看样本特征，不看数据集身份 → 通用
   - 记录 g 在 APDE / APRICOT 上的分布（验证"分工是模型自学的"）
B) 无标签族 DRO：Group DRO 需要组标签（Sagawa ICLR'20 的已知局限）
   - pseudo: 对 CLIP 特征做 KMeans → 伪族标签 → DRO
   - oracle: 用真族标签（对照，证明 pseudo 也有效）
评测：APDE 13 族（族级）+ APRICOT（跨集）+ 三个工作点
"""
import os, sys, json, argparse, math
os.environ.setdefault("HF_HOME","/home/oyp/.cache/huggingface")
os.environ.setdefault("HF_ENDPOINT","https://hf-mirror.com")
import numpy as np
from PIL import Image

WS="/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard"
TVC=f"{WS}/results/tvc"; EVAL=f"{TVC}/unified_eval_set"; CACHE=f"{TVC}/feat_cache"
TRAIN_JPG="/home/oyp/fastdata/apde_train_jpg"; TEST_JPG="/home/oyp/fastdata/apde_test_jpg"
SEEN={"advpatch","DM-NAP","GNAP","TCA","TCEGA","tsea","tsea-mim","tsea-pgd"}
UNSEEN={"AA","AdvCloak","AdvSticker","AdvTshirt","UPC"}

def wilson(k,n,z=1.96):
    if n==0: return (0.,0.)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (round(max(0.,c-h),4), round(min(1.,c+h),4))

def build_clip(name):
    import torch
    if name=="clip-b32":
        from transformers import CLIPModel
        m=CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda().eval()
        for p in m.parameters(): p.requires_grad=False
        def enc(pil):
            import torchvision.transforms as T
            x=T.Compose([T.Resize((224,224)),T.ToTensor(),
                         T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])(pil).unsqueeze(0).cuda()
            with torch.no_grad():
                o=m.vision_model(pixel_values=x)
                return m.visual_projection(o.pooler_output)[0].float().cpu().numpy()
        return enc,512
    import open_clip, torch
    mc,_,pre=open_clip.create_model_and_transforms("MobileCLIP2-S0",pretrained="dfndr2b",
                                                   cache_dir="/home/oyp/.cache/huggingface")
    mc=mc.cuda().eval()
    for p in mc.parameters(): p.requires_grad=False
    def enc(pil):
        with torch.no_grad(): f=mc.encode_image(pre(pil).unsqueeze(0).cuda())
        return f[0].float().cpu().numpy()
    return enc, int(enc(Image.new("RGB",(224,224))).shape[0])

def get_feats(rows,enc,key):
    f=f"{CACHE}/gl_{key}.npz"
    if os.path.exists(f):
        z=np.load(f)
        if len(z["X"])==len(rows):          # 样本数校验，防止错配（曾导致 F2 假结果）
            return z["X"],z["y"]
        print(f"    ⚠️ 缓存 {key} 长度不符({len(z['X'])} vs {len(rows)})，重新提取",flush=True)
    X=[];y=[]
    for i,r in enumerate(rows):
        try: X.append(enc(Image.open(r["path"]).convert("RGB")))
        except Exception: X.append(np.zeros_like(X[0]) if X else np.zeros(512,dtype=np.float32))
        y.append(r["label"])
        if (i+1)%4000==0: print(f"    feat {key} {i+1}/{len(rows)}",flush=True)
    X=np.stack(X).astype(np.float32);y=np.array(y);np.savez_compressed(f,X=X,y=y);return X,y

class WeightEMA:
    def __init__(self, model, decay=0.995):
        self.decay=decay
        self.shadow={k: v.detach().clone() for k,v in model.state_dict().items() if v.dtype.is_floating_point}
    def update(self, model):
        for k,v in model.state_dict().items():
            if k in self.shadow: self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1-self.decay)
    def apply(self, model):
        sd=model.state_dict(); sd.update({k:v.clone() for k,v in self.shadow.items()}); model.load_state_dict(sd)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--clip-model",default="mobileclip2-s0")
    ap.add_argument("--size",type=int,default=768)
    ap.add_argument("--epochs",type=int,default=8)
    ap.add_argument("--bs",type=int,default=32)
    ap.add_argument("--lr",type=float,default=5e-4)
    ap.add_argument("--limit-train",type=int,default=20000)
    ap.add_argument("--num-workers",type=int,default=8)
    ap.add_argument("--gating",action="store_true",help="样本级门控（否则硬拼接）")
    ap.add_argument("--dro-mode",default="none",choices=["none","oracle","pseudo"])
    ap.add_argument("--n-pseudo-groups",type=int,default=8)
    ap.add_argument("--dro-eta",type=float,default=2.0)
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--extra-negs", default="", help="额外负样本索引目录")
    ap.add_argument("--class-weight", action="store_true", help="CE 类别加权")
    ap.add_argument("--ema", action="store_true", help="权重 EMA（降 seed 方差）")
    ap.add_argument("--ema-decay", type=float, default=0.99)
    ap.add_argument("--val-split", type=float, default=0.0, help=">0 时按族分层划出验证集选最佳 epoch")
    ap.add_argument("--select-metric", default="worst_family", choices=["worst_family","mean","unseen"])
    ap.add_argument("--out",default=f"{TVC}/gated_labelfree.json")
    a=ap.parse_args()
    import torch, torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms, models
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    tr=json.load(open(f"{TRAIN_JPG}/index.json")); te=json.load(open(f"{TEST_JPG}/index.json"))
    pos=[r for r in tr if r["label"]==1]; neg=[r for r in tr if r["label"]==0]
    import random; rng=random.Random(a.seed); rng.shuffle(pos)
    sub=pos[:a.limit_train//2]+neg
    if a.extra_negs:
        en=json.load(open(f"{a.extra_negs}/index.json"))
        import random as _r; _r.Random(a.seed).shuffle(en)
        n_extra=min(len(en), max(0, a.limit_train - len(sub)))
        sub = sub + en[:n_extra]
        print(f"  + 额外负样本 {n_extra} 张 → 总计 {len(sub)}",flush=True)
    print(f"[gated] train={len(sub)} (pos={min(a.limit_train//2,len(pos))}, neg={len(neg)}) "
          f"gating={a.gating} dro={a.dro_mode}",flush=True)

    enc,cdim=build_clip(a.clip_model)
    Xtr,ytr=get_feats(sub,enc,"train"+("_xneg" if a.extra_negs else ""))

    # ── 组标签：oracle=真族 / pseudo=KMeans(CLIP特征) ──
    if a.dro_mode=="oracle":
        fams=sorted({r.get("method","clean") for r in sub}); f2i={f:i for i,f in enumerate(fams)}
        grp=np.array([f2i[r.get("method","clean")] for r in sub])
        print(f"  oracle 组 {len(fams)}: {fams}",flush=True)
    elif a.dro_mode=="pseudo":
        from sklearn.cluster import KMeans
        mu,sd=Xtr.mean(0),Xtr.std(0)+1e-6
        km=KMeans(n_clusters=a.n_pseudo_groups,n_init=10,random_state=0).fit((Xtr-mu)/sd)
        grp=km.labels_
        from collections import Counter
        print(f"  pseudo 组({a.n_pseudo_groups}) 分布: {dict(Counter(grp.tolist()))}",flush=True)
    else:
        grp=np.zeros(len(sub),dtype=int)

    tf_tr=transforms.Compose([transforms.RandomHorizontalFlip(),transforms.ColorJitter(0.2,0.2,0.2),
                              transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    tf_ev=transforms.Compose([transforms.Resize((a.size,a.size)),transforms.ToTensor(),
                              transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    def fit(im,s): return im if im.size==(s,s) else im.resize((s,s),Image.BILINEAR)

    class DS(Dataset):
        def __init__(s,rows,feats,labs,grp,tf): s.r,s.f,s.y,s.g,s.tf=rows,feats,labs,grp,tf
        def __len__(s): return len(s.r)
        def __getitem__(s,i):
            try: im=Image.open(s.r[i]["path"]).convert("RGB")
            except Exception: im=Image.new("RGB",(a.size,a.size))
            return s.tf(fit(im,a.size)), s.f[i], s.y[i], int(s.g[i])

    class Gated(nn.Module):
        def __init__(s,cdim,gating):
            super().__init__()
            b=models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
            td=b.fc.in_features; b.fc=nn.Identity(); s.tex=b
            s.gating=gating
            if gating:
                s.gate=nn.Sequential(nn.LayerNorm(td+cdim),nn.Linear(td+cdim,128),nn.ReLU(),nn.Linear(128,1))
                s.head=nn.Sequential(nn.LayerNorm(td+cdim),nn.Linear(td+cdim,256),nn.ReLU(),
                                     nn.Dropout(0.3),nn.Linear(256,2))
            else:
                s.head=nn.Sequential(nn.LayerNorm(td+cdim),nn.Linear(td+cdim,256),nn.ReLU(),
                                     nn.Dropout(0.3),nn.Linear(256,2))
        def forward(s,x,cf,return_gate=False):
            t=s.tex(x)                                   # [B,1024]
            if s.gating:
                # 门控：分别缩放两分支再拼接（保持各自维度，语义清晰）
                g=torch.sigmoid(s.gate(torch.cat([t,cf],1)))   # [B,1]
                out=s.head(torch.cat([g*t,(1.0-g)*cf],1))
                return (out,g.squeeze(1)) if return_gate else out
            return s.head(torch.cat([t,cf],1))

    # 分层验证集（用于选 epoch，降 seed 方差）
    val_idx=[]; tr_idx=list(range(len(sub)))
    if a.val_split>0 and len(grp)>0:
        import collections
        by_g=collections.defaultdict(list)
        for i,g_ in enumerate(grp): by_g[int(g_)].append(i)
        for g_,idxs in by_g.items():
            k=max(1,int(len(idxs)*a.val_split))
            val_idx.extend(idxs[:k])
        tr_idx=[i for i in range(len(sub)) if i not in set(val_idx)]
        print(f"  验证集 {len(val_idx)} / 训练 {len(tr_idx)}（按组分层的 {a.val_split:.0%}）",flush=True)

    m=Gated(cdim,a.gating).cuda()
    opt=torch.optim.AdamW(m.parameters(),lr=a.lr,weight_decay=1e-3)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs)
    if a.class_weight:
        npos=int(sum(1 for r in sub if r["label"]==1)); nneg=len(sub)-npos
        cw=torch.tensor([len(sub)/(2*max(nneg,1)), len(sub)/(2*max(npos,1))]).cuda()
        ce=nn.CrossEntropyLoss(weight=cw, reduction="none")
        print(f"  类别加权: w_neg={cw[0]:.2f} w_pos={cw[1]:.2f}",flush=True)
    else:
        ce=nn.CrossEntropyLoss(reduction="none")
    dl=DataLoader(DS(sub,Xtr,ytr,grp,tf_tr),batch_size=a.bs,shuffle=True,num_workers=a.num_workers,
                  pin_memory=False,drop_last=True,persistent_workers=(a.num_workers>0))
    ngrp=int(grp.max())+1
    gema=torch.zeros(ngrp); gate_vals=[]; wg_hist=[]
    ema=WeightEMA(m,a.ema_decay) if a.ema else None; best_sel=-1.0; best_sd=None

    for ep in range(a.epochs):
        m.train(); tot=0.
        for j,(x,cf,y,g_) in enumerate(dl):
            x,cf,y,g_=x.cuda(),cf.cuda(),y.cuda(),g_.cuda()
            out,gv=m.forward(x,cf,True) if a.gating else (m(x,cf),None)
            if gv is not None: gate_vals.append(gv.detach().cpu())
            per=ce(out,y)
            if a.dro_mode!="none":
                losses={}
                for gf in g_.unique():
                    losses[int(gf)]=per[g_==gf].mean()
                for gf,l in losses.items(): gema[gf]=0.9*gema[gf]+0.1*l.detach().cpu()
                w=torch.softmax(gema*a.dro_eta,dim=0)
                loss=sum(w[gf].cuda()*l for gf,l in losses.items())
                if (j+1)%300==0: wg_hist.append(w.detach().cpu().numpy().round(3).tolist())
            else: loss=per.mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot+=float(loss)
            if (j+1)%300==0: print(f"  ep{ep+1} {j+1}/{len(dl)} loss={tot/(j+1):.4f}",flush=True)
        sch.step()
        if a.ema and ep >= max(0, a.epochs-2): ema.update(m)   # 只在末段累积
        print(f"  [ep{ep+1}/{a.epochs}] loss={tot/max(len(dl),1):.4f}",flush=True)
        # 验证集选 epoch（按族算，取最差族/均值）
        if val_idx:
            m.eval(); vsc={}
            with torch.no_grad():
                for i in val_idx[::max(1,len(val_idx)//400)]:
                    try: im=Image.open(sub[i]["path"]).convert("RGB")
                    except Exception: continue
                    o,_=m(tf_ev(im).unsqueeze(0).cuda(),torch.from_numpy(Xtr[i]).unsqueeze(0).cuda())
                    vsc.setdefault(int(grp[i]),[]).append((float(torch.softmax(o,1)[0,1]),sub[i]["label"]))
            metrics={}
            for g_,pairs in vsc.items():
                arr=np.array(pairs); pos=arr[arr[:,1]==1][:,0]; neg=arr[arr[:,1]==0][:,0]
                if len(pos)>0 and len(neg)>0: metrics[g_]=float((pos>=np.percentile(neg,95)).mean())
            if metrics:
                sel={"worst_family":min(metrics.values()),"mean":float(np.mean(list(metrics.values()))),
                     "unseen":float(np.mean(list(metrics.values())))}[a.select_metric]
                if sel>best_sel:
                    best_sel=sel; best_sd={k:v.detach().clone() for k,v in m.state_dict().items()}
                    print(f"      val {a.select_metric}={sel:.4f} → 记录最佳 epoch {ep+1}",flush=True)
            m.train()

    if best_sd is not None:
        m.load_state_dict(best_sd); print(f"  使用验证集最佳权重（{a.select_metric}）",flush=True)
    elif ema is not None:
        ema.apply(m); print("  使用 EMA 权重",flush=True)
    m.eval()
    @torch.no_grad()
    def score(rows,feats,collect_gate=False):
        out=[];gs=[]
        dl2=DataLoader(DS(rows,feats,np.zeros(len(rows)),np.zeros(len(rows)),tf_ev),
                       batch_size=64,shuffle=False,num_workers=0)
        for x,cf,_,_ in dl2:
            if a.gating:
                o,gv=m(x.cuda(),cf.cuda(),True); gs.extend(gv.cpu().numpy().tolist())
            else: o=m(x.cuda(),cf.cuda())
            out.extend(torch.softmax(o,1)[:,1].float().cpu().numpy().tolist())
        return (np.array(out),np.array(gs)) if collect_gate else (np.array(out),None)

    # APDE 评测
    Xte,yte=get_feats(te,enc,"apde_test")
    s,_=score(te,Xte)
    pos_m=np.array([r["method"] for r in te if r["label"]==1])
    sp=s[[i for i,r in enumerate(te) if r["label"]==1]]; sn=s[[i for i,r in enumerate(te) if r["label"]==0]]
    wp={}
    for al in (0.05,0.01,0.001):
        t=float(np.percentile(sn,100*(1-al))); wp[f"@{al*100:g}pct"]=round(float((sp>=t).mean()),4)
    fam_tpr={f:round(float((sp[pos_m==f]>=np.percentile(sn,95)).mean()),4) for f in sorted(set(pos_m))}
    seen=[v for k,v in fam_tpr.items() if k in SEEN]; unseen=[v for k,v in fam_tpr.items() if k in UNSEEN]
    res={"gating":a.gating,"dro_mode":a.dro_mode,"n_train":len(sub),
         "params_M":round(sum(p.numel() for p in m.parameters())/1e6,2),
         "apde":{"workpoints":wp,"by_family":fam_tpr,"seen_mean":round(float(np.mean(seen)),4),
                 "unseen_mean":round(float(np.mean(unseen)),4)}}
    # APRICOT
    pv=[{"path":f"{EVAL}/pos/{f}","label":1} for f in sorted(os.listdir(f"{EVAL}/pos"))]
    S3=["casia_au","coco_val","inria_noped"]
    nv=[{"path":f"{EVAL}/{s_}/{f}","label":0} for s_ in S3+["columbia_au"] for f in sorted(os.listdir(f"{EVAL}/{s_}"))]
    XA,_=get_feats(pv+nv,enc,"apricot")
    sA,gA=score(pv+nv,XA,collect_gate=True)
    pA=sA[:len(pv)]; nA=sA[len(pv):]
    ap={}
    for al in (0.05,0.01,0.001):
        t=float(np.percentile(nA,100*(1-al))); ap[f"@{al*100:g}pct"]=round(float((pA>=t).mean()),4)
    res["apricot"]=ap
    # 门控分布（关键分析：APDE vs APRICOT 是否分化）
    if a.gating and gA is not None:
        _,gAPDE=score(te[:2000],Xte[:2000],collect_gate=True)
        res["gate_stats"]={"apricot_mean":float(np.mean(gA)),"apricot_p50":float(np.percentile(gA,50)),
                           "apde_mean":float(np.mean(gAPDE)) if gAPDE is not None else None,
                           "separation":float(abs(np.mean(gA)-(np.mean(gAPDE) if gAPDE is not None else 0)))}
    if wg_hist: res["group_weight_last"]=wg_hist[-1]
    print(f"\n  ✅ gating={a.gating} dro={a.dro_mode} ({res['params_M']}M)")
    print(f"     APDE: 未见5族={res['apde']['unseen_mean']}  UPC={fam_tpr.get('UPC')}  @0.1%={wp.get('@0.1pct')}")
    print(f"     APRICOT: {ap}")
    if "gate_stats" in res: print(f"     门控分布: {res['gate_stats']}")
    json.dump(res,open(a.out,"w"),indent=1,ensure_ascii=False)
    torch.save({"state_dict":m.state_dict(),"cdim":cdim,"size":a.size,"gating":a.gating},a.out.replace(".json",".pt"))
    print(f"→ {a.out}")

if __name__=="__main__": main()
