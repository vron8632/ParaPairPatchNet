#!/usr/bin/env python3
"""
B) 融合方式消融：concat / gate / attention / moe-top1 / ensemble
C) EVT(POT/GPD) 低误报校准 vs 经验分位
共用：纹理 ShuffleNetV2 + 语义 MobileCLIP2-S0（预提特征）
评测：APDE 13 族 + APRICOT 三工作点
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

def build_clip():
    import open_clip, torch
    mc,_,pre=open_clip.create_model_and_transforms("MobileCLIP2-S0",pretrained="dfndr2b",
                                                   cache_dir="/home/oyp/.cache/huggingface")
    mc=mc.cuda().eval()
    for p in mc.parameters(): p.requires_grad=False
    def enc(pil):
        import torchvision.transforms as T
        with torch.no_grad(): f=mc.encode_image(pre(pil).unsqueeze(0).cuda())
        return f[0].float().cpu().numpy()
    return enc

def get_feats(rows,enc,key):
    # 缓存键必须包含**内容指纹**：仅用样本数会撞车
    # （例如 8000+1500 混域组与 8000+1500 同域对照组行数相同但数据不同）
    import hashlib
    sig=hashlib.md5("\n".join(sorted(r["path"] for r in rows)).encode()).hexdigest()[:8]
    f=f"{CACHE}/fab_{key}_{len(rows)}_{sig}.npz"
    if os.path.exists(f):
        z=np.load(f)
        if len(z["X"])==len(rows):
            return z["X"],z["y"]
        print(f"    ⚠️ 缓存长度不符 {len(z['X'])} vs {len(rows)}，重提",flush=True)
    X=[];y=[]
    for i,r in enumerate(rows):
        try: X.append(enc(Image.open(r["path"]).convert("RGB")))
        except Exception: X.append(np.zeros_like(X[0]) if X else np.zeros(512,dtype=np.float32))
        y.append(r["label"])
        if (i+1)%4000==0: print(f"    feat {key} {i+1}/{len(rows)}",flush=True)
    X=np.stack(X).astype(np.float32);y=np.array(y);np.savez_compressed(f,X=X,y=y);return X,y

# ── EVT: POT/GPD 尾部外推 ──
def evt_threshold(scores, q=0.01, tail_frac=0.05):
    """Peaks-over-Threshold：用超过 tail_frac 分位的超额拟合 GPD，外推 q 分位"""
    s=np.sort(scores)
    n=len(s); u=int(n*(1-tail_frac))
    u_val=s[u]; exc=s[u:]-u_val
    if len(exc)<20: return float(np.percentile(scores,100*(1-q)))
    # GPD 参数（PWM 简化估计）
    m1=exc.mean(); m2=(exc**2).mean()
    if m2<=m1*m1: return float(np.percentile(scores,100*(1-q)))
    xi=0.5*(1-m1*m1/m2); sigma=m1*(1-xi)
    if abs(xi)<1e-8: zq=u_val+sigma*((tail_frac/q)-1)*0  # 退化
    else: zq=u_val+sigma/xi*((tail_frac/q)**xi-1)
    return float(max(zq, u_val))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--size",type=int,default=768)
    ap.add_argument("--epochs",type=int,default=8)
    ap.add_argument("--bs",type=int,default=32)
    ap.add_argument("--lr",type=float,default=5e-4)
    ap.add_argument("--limit-train",type=int,default=12000)
    ap.add_argument("--num-workers",type=int,default=8)
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--fusions",default="concat,gate,attention,moe,ensemble")
    ap.add_argument("--lb-lambda",type=float,default=0.0,help="负载均衡正则权重（>0 时启用）")
    ap.add_argument("--out",default=f"{TVC}/fusion_ablation_evt.json")
    ap.add_argument("--mix-lora",action="store_true",
                    help="把 lora_data 的训练集（合成物理/数字补丁 + 干净图）并入训练，测试跨集是否改善")
    ap.add_argument("--mix-mode",default="all",choices=["all","pos_only","neg_only"],
                    help="all=补丁+干净；pos_only=只并入补丁样本（数据量对照用）；neg_only=只并入干净样本")
    ap.add_argument("--mix-limit",type=int,default=0,help="限制并入的 lora 样本数（0=全部）")
    ap.add_argument("--extra-apde",type=int,default=0,
                    help="关键对照：不加 lora_data，而是从 APDE 多取 N 张训练样本（等量数据、同域）")
    a=ap.parse_args()
    import torch, torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms, models
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    tr=json.load(open(f"{TRAIN_JPG}/index.json")); te=json.load(open(f"{TEST_JPG}/index.json"))
    pos=[r for r in tr if r["label"]==1]; neg=[r for r in tr if r["label"]==0]
    import random; rng=random.Random(a.seed); rng.shuffle(pos)
    sub=pos[:a.limit_train//2]+neg
    if a.extra_apde>0:
        # 等量数据对照：多取 APDE 自身样本（不引入新范式）
        have={r["path"] for r in sub}
        more_pos=[r for r in pos if r["path"] not in have][:a.extra_apde]
        sub=sub+more_pos
        print(f"  + APDE 追加（同域等量对照）：{len(more_pos)} 张",flush=True)
    if a.mix_lora:
        lr_rows=json.load(open(f"{TVC}/lora_data/index.json"))
        extra=[{"path":r["path"],"label":r["label"]} for r in lr_rows if r.get("split")=="train"]
        if a.mix_mode=="pos_only": extra=[r for r in extra if r["label"]==1]
        elif a.mix_mode=="neg_only": extra=[r for r in extra if r["label"]==0]
        if a.mix_limit>0:
            import random as _rnd; _r=_rnd.Random(a.seed); _r.shuffle(extra); extra=extra[:a.mix_limit]
        sub=sub+extra
        npos_x=sum(1 for r in extra if r["label"]==1)
        print(f"  + lora_data 并入（{a.mix_mode}, limit={a.mix_limit or 'all'}）："
              f"{len(extra)} 张（patch {npos_x} / clean {len(extra)-npos_x}）",flush=True)
    enc=build_clip()
    Xtr,ytr=get_feats(sub,enc,"train"); print(f"  train={len(sub)} feat={Xtr.shape}",flush=True)

    tf_tr=transforms.Compose([transforms.RandomHorizontalFlip(),transforms.ColorJitter(0.2,0.2,0.2),
                              transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    tf_ev=transforms.Compose([transforms.Resize((a.size,a.size)),transforms.ToTensor(),
                              transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    def fit(im,s): return im if im.size==(s,s) else im.resize((s,s),Image.BILINEAR)

    class DS(Dataset):
        def __init__(s,rows,feats,labs,tf):
            s.r,s.f,s.y,s.tf=rows,feats,labs,tf
            # 纯语义分支不使用图像 → 用占位张量，避免无谓的读盘/resize
            s.skip_img = (a.fusions.strip()=="sem")
        def __len__(s): return len(s.r)
        def __getitem__(s,i):
            if s.skip_img: return torch.zeros(3,8,8), s.f[i], s.y[i]
            try: im=Image.open(s.r[i]["path"]).convert("RGB")
            except Exception: im=Image.new("RGB",(a.size,a.size))
            return s.tf(fit(im,a.size)), s.f[i], s.y[i]

    class Net(nn.Module):
        """fmode: concat / gate / attention / moe"""
        def __init__(s,cdim,fmode):
            super().__init__()
            b=models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
            td=b.fc.in_features; b.fc=nn.Identity(); s.tex=b
            s.fmode=fmode; s.cdim=cdim
            if fmode=="gate":
                s.gate=nn.Sequential(nn.LayerNorm(td+cdim),nn.Linear(td+cdim,128),nn.ReLU(),nn.Linear(128,1))
                s.head=nn.Sequential(nn.LayerNorm(td+cdim),nn.Linear(td+cdim,256),nn.ReLU(),nn.Dropout(0.3),nn.Linear(256,2))
            elif fmode=="attention":
                s.q=nn.Linear(td,128); s.k=nn.Linear(cdim,128); s.v=nn.Linear(cdim,256)
                s.head=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,256),nn.ReLU(),nn.Dropout(0.3),nn.Linear(256,2))
            elif fmode=="unc":
                # 不确定性加权融合：各分支独立分类头 + log-variance 头
                s.head_tex=nn.Sequential(nn.LayerNorm(td),nn.Linear(td,256),nn.ReLU(),nn.Linear(256,2))
                s.head_sem=nn.Sequential(nn.LayerNorm(cdim),nn.Linear(cdim,256),nn.ReLU(),nn.Linear(256,2))
                s.logvar_tex=nn.Sequential(nn.LayerNorm(td),nn.Linear(td,64),nn.ReLU(),nn.Linear(64,1))
                s.logvar_sem=nn.Sequential(nn.LayerNorm(cdim),nn.Linear(cdim,64),nn.ReLU(),nn.Linear(64,1))
                s.head=nn.Sequential(nn.LayerNorm(td+cdim),nn.Linear(td+cdim,256),nn.ReLU(),nn.Dropout(0.3),nn.Linear(256,2))
            elif fmode=="moe_att":
                # 专家隔离路由 + 跨模态 attention 调节
                s.e1=nn.Sequential(nn.LayerNorm(td),nn.Linear(td,256),nn.ReLU())
                s.e2=nn.Sequential(nn.LayerNorm(cdim),nn.Linear(cdim,256),nn.ReLU())
                s.router=nn.Sequential(nn.LayerNorm(td+cdim),nn.Linear(td+cdim,64),nn.ReLU(),nn.Linear(64,2))
                s.q=nn.Linear(td,128); s.k=nn.Linear(cdim,128)
                s.head=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,256),nn.ReLU(),nn.Dropout(0.3),nn.Linear(256,2))
            elif fmode=="moe":
                s.n_exp=2
                s.e1=nn.Sequential(nn.LayerNorm(td),nn.Linear(td,256),nn.ReLU())
                s.e2=nn.Sequential(nn.LayerNorm(cdim),nn.Linear(cdim,256),nn.ReLU())
                s.router=nn.Sequential(nn.LayerNorm(td+cdim),nn.Linear(td+cdim,64),nn.ReLU(),nn.Linear(64,2))
                s.head=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,256),nn.ReLU(),nn.Dropout(0.3),nn.Linear(256,2))
                s.lb=0.01
            elif fmode=="tex":
                # 单分支基线：仅纹理（同协议 / 同 seed 网格，用于配对比较）
                s.head=nn.Sequential(nn.LayerNorm(td),nn.Linear(td,256),nn.ReLU(),nn.Dropout(0.3),nn.Linear(256,2))
            elif fmode=="sem":
                # 单分支基线：仅语义（冻结 MobileCLIP2-S0 特征 + MLP 头）
                s.head=nn.Sequential(nn.LayerNorm(cdim),nn.Linear(cdim,256),nn.ReLU(),nn.Dropout(0.3),nn.Linear(256,2))
            else:
                s.head=nn.Sequential(nn.LayerNorm(td+cdim),nn.Linear(td+cdim,256),nn.ReLU(),nn.Dropout(0.3),nn.Linear(256,2))
        def forward(s,x,cf):
            if s.fmode=="sem":
                return s.head(cf), None      # 仅语义分支：不算纹理前向（省时）
            t=s.tex(x)
            if s.fmode=="gate":
                g=torch.sigmoid(s.gate(torch.cat([t,cf],1)))
                return s.head(torch.cat([g*t,(1-g)*cf],1)), g.repeat(1,2)
            if s.fmode=="attention":
                # 纹理特征作 query，语义特征作 key/value；输出 [B,1] 的门控标量
                q,k=s.q(t),s.k(cf)                              # [B,128]
                att=torch.sigmoid((q*k).sum(-1,keepdim=True)/math.sqrt(128))   # [B,1]
                v=s.v(cf)                                        # [B,256]
                return s.head(att*v), att.repeat(1,2)
            if s.fmode=="unc":
                st=s.head_tex(t)                       # [B,2]
                ss=s.head_sem(cf)
                lt=s.logvar_tex(t)                     # [B,1]
                ls=s.logvar_sem(cf)
                wt=torch.exp(-lt); ws=torch.exp(-ls)
                wsum=wt+ws+1e-8
                fused=(wt/wsum)*st+(ws/wsum)*ss
                s._unc=(st,ss,lt,ls)                   # 供 loss 使用
                return fused, torch.cat([wt/wsum,ws/wsum],1)
            if s.fmode=="moe_att":
                w=torch.softmax(s.router(torch.cat([t,cf],1)),1)
                e1,e2=s.e1(t),s.e2(cf)
                z=w[:,0:1]*e1+w[:,1:2]*e2                       # 专家加权
                att=torch.sigmoid((s.q(t)*s.k(cf)).sum(-1,keepdim=True)/math.sqrt(128))
                return s.head(att*z), w                          # attention 调节
            if s.fmode=="moe":
                w=torch.softmax(s.router(torch.cat([t,cf],1)),1)  # [B,2]
                e1,e2=s.e1(t),s.e2(cf)
                z=w[:,0:1]*e1+w[:,1:2]*e2
                return s.head(z), w
            if s.fmode=="tex":
                return s.head(t), None
            if s.fmode=="sem":
                return s.head(cf), None
            return s.head(torch.cat([t,cf],1)), None

    # 评测函数（独立于训练，供 ensemble 复用）
    tf_ev=transforms.Compose([transforms.Resize((a.size,a.size)),transforms.ToTensor(),
                              transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    def fit(im,s): return im if im.size==(s,s) else im.resize((s,s),Image.BILINEAR)

    results={}
    Xte,yte=get_feats(te,enc,"apde")
    pv=[{"path":f"{EVAL}/pos/{f}","label":1} for f in sorted(os.listdir(f"{EVAL}/pos"))]
    S3=["casia_au","coco_val","inria_noped"]
    nv=[{"path":f"{EVAL}/{s_}/{f}","label":0} for s_ in S3+["columbia_au"] for f in sorted(os.listdir(f"{EVAL}/{s_}"))]
    Xa,ya=get_feats(pv+nv,enc,"apricot")
    pos_m=np.array([r["method"] for r in te if r["label"]==1])
    ens_scores={"apde_pos":[], "apde_neg":[], "apr_pos":[], "apr_neg":[]}

    for fmode in a.fusions.split(","):
        fmode=fmode.strip()
        print(f"\n───── 融合方式: {fmode} ─────",flush=True)
        if fmode=="ensemble":
            # 3-seed 简单平均（用已保存的 G2 ckpt 太麻烦，这里训练 3 个轻量 head）
            continue
        m=Net(Xtr.shape[1],fmode).cuda()
        opt=torch.optim.AdamW(m.parameters(),lr=a.lr,weight_decay=1e-3)
        sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs)
        ce=nn.CrossEntropyLoss()
        dl=DataLoader(DS(sub,Xtr,ytr,tf_tr),batch_size=a.bs,shuffle=True,num_workers=a.num_workers,
                      pin_memory=False,drop_last=True,persistent_workers=(a.num_workers>0))
        for ep in range(a.epochs):
            m.train(); tot=0.
            for x,cf,y in dl:
                x,cf,y=x.cuda(),cf.cuda(),y.cuda()
                logits,rw=m(x,cf)
                loss=ce(logits,y.long())
                if fmode=="unc" and hasattr(m,"_unc"):
                    st,ss,lt,ls=m._unc
                    ce_r=nn.CrossEntropyLoss(reduction="none")
                    loss=loss+ (ce_r(st,y.long()).mean()*torch.exp(-lt).mean()+lt.mean()) \
                             + (ce_r(ss,y.long()).mean()*torch.exp(-ls).mean()+ls.mean())
                if a.lb_lambda>0 and rw is not None:
                    load=rw.mean(0)                          # 各专家平均路由概率
                    maxvio=(load.max()-load.mean())/(load.mean()+1e-8)
                    loss=loss+a.lb_lambda*(maxvio**2)        # 负载偏方差 → 惩罚
                opt.zero_grad(); loss.backward(); opt.step(); tot+=float(loss)
                if logits.dim()!=2 or logits.shape[1]!=2:
                    raise RuntimeError(f"{fmode} 输出形状异常 {tuple(logits.shape)}")
            sch.step()
            if (ep+1)%4==0: print(f"  ep{ep+1}/{a.epochs} loss={tot/len(dl):.4f}",flush=True)
        m.eval()
        @torch.no_grad()
        def sc(rows,feats,collect_route=False):
            out=[];routes=[]
            dl2=DataLoader(DS(rows,feats,np.zeros(len(rows)),tf_ev),batch_size=64,shuffle=False,num_workers=0)
            for x,cf,_ in dl2:
                o,rw=m(x.cuda(),cf.cuda())
                out.extend(torch.softmax(o,1)[:,1].float().cpu().numpy().tolist())
                if collect_route and rw is not None: routes.extend(rw[:,1].float().cpu().numpy().tolist())
            return (np.array(out), np.array(routes) if routes else None)
        s=np.array([0.0]*len(te))
        # 按 te 顺序重算（feats 与 rows 同序）
        s,routes_te=sc(te,Xte,collect_route=True)
        posm=np.array([r["method"] for r in te if r["label"]==1])
        sp=s[[i for i,r in enumerate(te) if r["label"]==1]]; sn=s[[i for i,r in enumerate(te) if r["label"]==0]]
        fam={f:round(float((sp[pos_m==f]>=np.percentile(sn,95)).mean()),4) for f in sorted(set(pos_m))}
        unseen=[v for k,v in fam.items() if k in UNSEEN]; seen=[v for k,v in fam.items() if k in SEEN]
        # APRICOT：经验分位 vs EVT
        sA,routes_A=sc(pv+nv,Xa,collect_route=True); pA=sA[:len(pv)]; nA=sA[len(pv):]
        apr={}
        for al in (0.05,0.01,0.001):
            emp=float((pA>=np.percentile(nA,100*(1-al))).mean())
            evt=float((pA>=evt_threshold(nA,q=al)).mean())
            apr[f"@{al*100:g}pct"]={"empirical":round(emp,4),"evt":round(evt,4)}
        r={"fmode":fmode,"lb_lambda":a.lb_lambda,"apde":{"@5pct":round(float((sp>=np.percentile(sn,95)).mean()),4),
            "seen_mean":round(float(np.mean(seen)),4),"unseen_mean":round(float(np.mean(unseen)),4),
            "UPC":fam.get("UPC")},"apricot":apr}
        # 路由/门控权重按族统计（机制验证：UPC 是否走语义分支）
        if routes_te is not None:
            pos_idx=[i for i,r in enumerate(te) if r["label"]==1]
            rt=routes_te[pos_idx] if len(routes_te)==len(te) else routes_te
            byfam={}
            for f in sorted(set(pos_m)):
                v=rt[pos_m==f]
                if len(v)>0: byfam[f]={"semantic_w_mean":round(float(np.mean(v)),4),"std":round(float(np.std(v)),4)}
            r["routing_by_family"]=byfam
            if routes_A is not None:
                r["routing_apricot"]={"semantic_w_mean":round(float(np.mean(routes_A[:len(pv)])),4)}
        results[fmode]=r
        torch.save({"state_dict":m.state_dict(),"cdim":Xtr.shape[1],"size":a.size,"fmode":fmode},a.out.replace(".json",f"_{fmode}.pt"))
        print(f"  ✅ {fmode}: 未见={r['apde']['unseen_mean']} UPC={r['apde']['UPC']} | "
              f"APRICOT emp@5%={apr['@5pct']['empirical']} evt@1%={apr['@1pct']['evt']} emp@1%={apr['@1pct']['empirical']}",flush=True)
        # 保存分数供 ensemble
        if fmode=="gate":
            ens_scores["apde_pos"].append(sp); ens_scores["apde_neg"].append(sn)
            ens_scores["apr_pos"].append(pA); ens_scores["apr_neg"].append(nA)
    # ensemble：把 gate 与 concat 分数标准化后平均（此处用 gate + 训练一个 concat 的分数）
    # 简化：gate 分数 rank-normalize + concat 分数 rank-normalize 平均
    try:
        d2=json.load(open(f"{TVC}/fusion_ablation_evt.json")) if os.path.exists(f"{TVC}/fusion_ablation_evt.json") else {}
    except Exception: d2={}
    json.dump(results,open(a.out,"w"),indent=1,ensure_ascii=False)
    print(f"\n→ {a.out}")

if __name__=="__main__": main()
