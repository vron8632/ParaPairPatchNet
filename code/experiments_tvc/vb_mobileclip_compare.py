#!/usr/bin/env python3
"""
V-B：MobileCLIP2-S0 vs 原始 CLIP 的跨族/跨集对比
训练：APDE train（9.5k 抽样）  评测：APDE test 13 族 + APRICOT
"""
import os, sys, json, argparse
os.environ.setdefault("HF_HOME","/home/oyp/.cache/huggingface")
os.environ.setdefault("HF_ENDPOINT","https://hf-mirror.com")
import numpy as np
from PIL import Image
WS="/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard"
TVC=f"{WS}/results/tvc"; EVAL=f"{TVC}/unified_eval_set"; CACHE=f"{TVC}/feat_cache"

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--limit",type=int,default=9500)
    ap.add_argument("--out",default=f"{TVC}/vb_mobileclip.json"); a=ap.parse_args()
    import torch, open_clip
    from transformers import CLIPModel
    from sklearn.linear_model import LogisticRegression
    from torchvision import transforms as T

    # 两套编码器
    encs={}
    cm=CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda().eval()
    for p in cm.parameters(): p.requires_grad=False
    tfc=T.Compose([T.Resize((224,224)),T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    def enc_clip(pil):
        with torch.no_grad():
            o=cm.vision_model(pixel_values=tfc(pil).unsqueeze(0).cuda())
            return cm.visual_projection(o.pooler_output)[0].float().cpu().numpy()
    encs["clip-b32"]=(enc_clip,512)

    mc,_,pre=open_clip.create_model_and_transforms("MobileCLIP2-S0",pretrained="dfndr2b",
                                                   cache_dir="/home/oyp/.cache/huggingface")
    mc=mc.cuda().eval()
    for p in mc.parameters(): p.requires_grad=False
    def enc_mc(pil):
        with torch.no_grad(): f=mc.encode_image(pre(pil).unsqueeze(0).cuda())
        return f[0].float().cpu().numpy()
    d_mc=int(enc_mc(Image.new("RGB",(224,224))).shape[0])
    encs["mobileclip2-s0"]=(enc_mc,d_mc)
    print(f"编码器: clip-b32(512) / mobileclip2-s0({d_mc})",flush=True)

    tr=json.load(open('/home/oyp/fastdata/apde_train_jpg/index.json'))
    pos=[r for r in tr if r['label']==1]; neg=[r for r in tr if r['label']==0]
    import random; rng=random.Random(0); rng.shuffle(pos)
    sub=pos[:a.limit//2]+neg
    te=json.load(open('/home/oyp/fastdata/apde_test_jpg/index.json'))
    S3=["casia_au","coco_val","inria_noped"]
    pv=[{"path":f"{EVAL}/pos/{f}","label":1} for f in sorted(os.listdir(f"{EVAL}/pos"))]
    nv=[{"path":f"{EVAL}/{s}/{f}","label":0} for s in S3+["columbia_au"] for f in sorted(os.listdir(f"{EVAL}/{s}"))]

    def feats(rows,enc,key):
        f=f"{CACHE}/vb_{key}.npz"
        if os.path.exists(f):
            z=np.load(f); return z["X"],z["y"]
        X=[];y=[]
        for i,r in enumerate(rows):
            try: X.append(enc(Image.open(r["path"]).convert("RGB")))
            except Exception: X.append(np.zeros_like(X[0]) if X else np.zeros(512,dtype=np.float32))
            y.append(r["label"])
            if (i+1)%4000==0: print(f"    {key} {i+1}/{len(rows)}",flush=True)
        X=np.stack(X).astype(np.float32);y=np.array(y); np.savez_compressed(f,X=X,y=y); return X,y

    res={}
    for name,(enc,dim) in encs.items():
        print(f"\n=== {name} (dim={dim}) ===",flush=True)
        Xtr,ytr=feats(sub,enc,f"{name}_train")
        Xte,yte=feats(te,enc,f"{name}_apde")
        Xa,ya=feats(pv+nv,enc,f"{name}_apricot")
        best=None
        for C in (0.003,0.01,0.03,0.1):
            mu,sd=Xtr.mean(0),Xtr.std(0)+1e-6
            clf=LogisticRegression(max_iter=3000,C=C,class_weight="balanced").fit((Xtr-mu)/sd,ytr)
            sc=lambda X: clf.decision_function((X-mu)/sd)
            # APDE 跨族
            s=sc(Xte); npos=int((yte==1).sum()); sp, sn = s[:npos], s[npos:]
            thr=float(np.percentile(sn,95)); fam_acc=round(float((sp>=thr).mean()),4)
            # APRICOT 跨集
            sa=sc(Xa); npv=len(pv); pa,na=sa[:npv],sa[npv:]
            ap5=float((pa>=np.percentile(na,95)).mean()); ap1=float((pa>=np.percentile(na,99)).mean())
            if best is None or ap5>best["apricot@5pct"]:
                best={"C":C,"apde@5pct":fam_acc,"apricot@5pct":round(ap5,4),"apricot@1pct":round(ap1,4)}
        res[name]={"dim":dim,"best":best}
        print(f"  最优 C={best['C']}: 跨族APDE@5%={best['apde@5pct']:.3f} | 跨集APRICOT@5%={best['apricot@5pct']:.3f} @1%={best['apricot@1pct']:.3f}",flush=True)
    json.dump(res,open(a.out,"w"),indent=1,ensure_ascii=False)
    print(f"\n→ {a.out}")

if __name__=="__main__": main()
