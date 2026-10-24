#!/usr/bin/env python3
"""
双分支（纹理+语义） + 族级 DRO
================================
纹理分支：ShuffleNetV2（可训练）
语义分支：CLIP / MobileCLIP 图像特征（冻结，预提取缓存）
融合：concat → 小 MLP
DRO：按攻击族分组，最差族加权（min_θ max_g E_g[loss]）

评测：APDE test 13 族（含 5 未见）+ APRICOT（跨集）
"""
import os, sys, json, argparse, math, io
os.environ.setdefault("HF_HOME", "/home/oyp/.cache/huggingface")
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

def fit(im,size):
    """统一到 size×size 正方形（避免不同长宽比导致 collate 失败）"""
    return im if im.size==(size,size) else im.resize((size,size),Image.BILINEAR)

# ── CLIP 特征提取（缓存）──
def build_clip(model_name):
    import torch
    if model_name == "clip-b32":
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
        return enc, 512
    else:
        import open_clip, torch
        name,tag=("MobileCLIP2-S0","dfndr2b") if "2-S0" in model_name else ("MobileCLIP-S1","datacompdr")
        m,_,pre=open_clip.create_model_and_transforms(name,pretrained=tag,cache_dir="/home/oyp/.cache/huggingface")
        m=m.cuda().eval()
        for p in m.parameters(): p.requires_grad=False
        def enc(pil):
            with torch.no_grad():
                f=m.encode_image(pre(pil).unsqueeze(0).cuda())
            return f[0].float().cpu().numpy()
        return enc, m.encode_image(torch.zeros(1,3,224,224).cuda()).shape[-1]

def get_feats(rows, enc, key):
    f=f"{CACHE}/fus_{key}.npz"
    if os.path.exists(f):
        z=np.load(f); return z["X"], z["y"]
    X=[]; y=[]
    for i,r in enumerate(rows):
        try: X.append(enc(Image.open(r["path"]).convert("RGB")))
        except Exception: X.append(np.zeros_like(X[0]) if X else np.zeros(512,dtype=np.float32))
        y.append(r["label"])
        if (i+1)%3000==0: print(f"    feat {key} {i+1}/{len(rows)}",flush=True)
    X=np.stack(X).astype(np.float32); y=np.array(y)
    np.savez_compressed(f,X=X,y=y); return X,y

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--clip-model",default="clip-b32",choices=["clip-b32","mobileclip2-s0","mobileclip-s1"])
    ap.add_argument("--size",type=int,default=768)
    ap.add_argument("--epochs",type=int,default=8)
    ap.add_argument("--bs",type=int,default=32)
    ap.add_argument("--lr",type=float,default=5e-4)
    ap.add_argument("--limit-train",type=int,default=20000)
    ap.add_argument("--num-workers",type=int,default=8)
    ap.add_argument("--dro",action="store_true")
    ap.add_argument("--dro-eta",type=float,default=2.0)
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--out",default=f"{TVC}/p1_fusion_dro.json")
    a=ap.parse_args()
    import torch, torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms, models
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    tr=json.load(open(f"{TRAIN_JPG}/index.json"))
    te=json.load(open(f"{TEST_JPG}/index.json"))
    pos=[r for r in tr if r["label"]==1]; neg=[r for r in tr if r["label"]==0]
    import random; rng=random.Random(0); rng.shuffle(pos)
    sub=pos[:a.limit_train//2]+neg
    print(f"[fusion] train={len(sub)} test={len(te)} clip={a.clip_model} dro={a.dro}",flush=True)

    enc,cdim=build_clip(a.clip_model)
    Xtr,ytr=get_feats(sub,enc,"apde_train")
    print(f"  CLIP 特征 {Xtr.shape}",flush=True)

    # 族标签（clean 记为 "clean"）
    fams=sorted({r.get("method","clean") for r in sub})
    fam2i={f:i for i,f in enumerate(fams)}
    fam_idx=np.array([fam2i[r.get("method","clean")] for r in sub])
    print(f"  训练族 {len(fams)}: {fams}",flush=True)

    tf_tr=transforms.Compose([transforms.RandomHorizontalFlip(),transforms.ColorJitter(0.2,0.2,0.2),
                              transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    tf_ev=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    class DS(Dataset):
        def __init__(s,rows,feats,labs,tf): s.rows,s.f,s.y,s.tf=rows,feats,labs,tf
        def __len__(s): return len(s.rows)
        def __getitem__(s,i):
            try: im=Image.open(s.rows[i]["path"]).convert("RGB")
            except Exception: im=Image.new("RGB",(a.size,a.size))
            return s.tf(fit(im,a.size)), s.f[i], s.y[i]

    class Fusion(nn.Module):
        def __init__(s,cdim):
            super().__init__()
            b=models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
            td=b.fc.in_features; b.fc=nn.Identity(); s.tex=b
            s.fuse=nn.Sequential(nn.LayerNorm(td+cdim),nn.Linear(td+cdim,256),nn.ReLU(),
                                 nn.Dropout(0.3),nn.Linear(256,2))
        def forward(s,x,cf):
            return s.fuse(torch.cat([s.tex(x),cf],1))

    head=nn.Linear(1024+cdim,512)
    m=Fusion(cdim).cuda()
    opt=torch.optim.AdamW(m.parameters(),lr=a.lr,weight_decay=1e-3)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs)
    ce=nn.CrossEntropyLoss(reduction="none")
    dl=DataLoader(DS(sub,Xtr,ytr,tf_tr),batch_size=a.bs,shuffle=True,num_workers=a.num_workers,
                  pin_memory=False,drop_last=True,persistent_workers=(a.num_workers>0))
    fam_ema=torch.zeros(len(fams))

    for ep in range(a.epochs):
        m.train(); tot=0.
        for j,(x,cf,y) in enumerate(dl):
            x,cf,y=x.cuda(),cf.cuda(),y.cuda()
            idx=torch.arange((j*a.bs)%len(sub),(j*a.bs)%len(sub)+len(y))%len(sub)
            fi=torch.tensor(fam_idx[idx.numpy()]).cuda()
            logits=m(x,cf); per=ce(logits,y)
            if a.dro:
                losses={}
                for f in fi.unique():
                    losses[int(f)]=per[fi==f].mean()
                for f,l in losses.items():
                    fam_ema[f]=0.9*fam_ema[f]+0.1*l.detach().cpu()
                w=torch.softmax(fam_ema*a.dro_eta,dim=0)
                loss=sum(w[f].cuda()*l for f,l in losses.items())
            else:
                loss=per.mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot+=float(loss)
            if (j+1)%300==0: print(f"  ep{ep+1} {j+1}/{len(dl)} loss={tot/(j+1):.4f}",flush=True)
        sch.step(); print(f"  [ep{ep+1}/{a.epochs}] loss={tot/max(len(dl),1):.4f}",flush=True)

    # 评测
    m.eval()
    @torch.no_grad()
    def score(rows,feats,tf):
        out=[]
        dl2=DataLoader(DS(rows,feats,np.zeros(len(rows)),tf),batch_size=64,shuffle=False,num_workers=0)
        for x,cf,_ in dl2:
            out.extend(torch.softmax(m(x.cuda(),cf.cuda()),1)[:,1].float().cpu().numpy().tolist())
        return np.array(out)

    Xte,yte=get_feats(te,enc,"apde_test")
    s=score(te,Xte,tf_ev)
    pos_m=np.array([r["method"] for r in te if r["label"]==1])
    sp=s[[i for i,r in enumerate(te) if r["label"]==1]]
    sn=s[[i for i,r in enumerate(te) if r["label"]==0]]
    thr=float(np.percentile(sn,95))
    fams_tpr={}
    for f in sorted(set(pos_m)):
        fams_tpr[f]=round(float((sp[pos_m==f]>=thr).mean()),4)
    seen=[v for k,v in fams_tpr.items() if k in SEEN]; unseen=[v for k,v in fams_tpr.items() if k in UNSEEN]
    res={"arch":"fusion","clip":a.clip_model,"dro":a.dro,"dro_eta":a.dro_eta,
         "n_train":len(sub),"params_M":round(sum(p.numel() for p in m.parameters())/1e6,2),
         "apde":{"@5pct_overall":round(float((sp>=thr).mean()),4),"by_family":fams_tpr,
                 "seen_mean":round(float(np.mean(seen)),4),"unseen_mean":round(float(np.mean(unseen)),4)}}
    # APRICOT（跨集）
    rowsA=[{"path":f"{EVAL}/pos/{f}","label":1} for f in sorted(os.listdir(f"{EVAL}/pos"))]
    S3=["casia_au","coco_val","inria_noped"]
    rowsN=[{"path":f"{EVAL}/{s_}/{f}","label":0} for s_ in S3+["columbia_au"] for f in sorted(os.listdir(f"{EVAL}/{s_}"))]
    XA,yA=get_feats(rowsA+rowsN,enc,"apricot")
    sA=score(rowsA+rowsN,XA,tf_ev)
    pA=sA[:len(rowsA)]; nA=sA[len(rowsA):]
    ap={}
    for al in (0.05,0.01):
        t=float(np.percentile(nA,100*(1-al))); ap[f"@{al*100:g}pct"]=round(float((pA>=t).mean()),4)
    res["apricot"]=ap
    print(f"\n  ✅ Fusion({res['params_M']}M, clip={a.clip_model}, dro={a.dro})")
    print(f"     APDE@5%={res['apde']['@5pct_overall']}  见过={res['apde']['seen_mean']}  **未见={res['apde']['unseen_mean']}**")
    print(f"     UPC={fams_tpr.get('UPC')}  |  APRICOT(跨集)={ap}")
    json.dump(res,open(a.out,"w"),indent=1,ensure_ascii=False)
    torch.save({"state_dict":m.state_dict(),"cdim":cdim,"size":a.size},a.out.replace(".json",".pt"))
    print(f"→ {a.out}")

if __name__=="__main__": main()
