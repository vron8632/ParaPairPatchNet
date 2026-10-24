#!/usr/bin/env python3
"""
V-A：CLIP 冻结特征的「跨族」与「跨集」能力对比
=================================================
训练特征来源：APDE train（8 族）
评测：
  - 跨族：APDE test 13 族（见过 8 + 未见 5）
  - 跨集：APRICOT（138 pos + 1000 neg）
对照：ShuffleNetV2 端到端（跨族 0.834 / 跨集 0.399）
判定：CLIP 跨集 >= 0.6 → 双分支方案成立
"""
import os, sys, json, argparse
import numpy as np
from PIL import Image
WS="/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard"
TVC=f"{WS}/results/tvc"; EVAL=f"{TVC}/unified_eval_set"
CACHE=f"{TVC}/feat_cache"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train-limit",type=int,default=15000)
    ap.add_argument("--out",default=f"{TVC}/va_clip_crossdomain.json")
    a=ap.parse_args()
    import torch
    from transformers import CLIPModel
    from torchvision import transforms as T
    from sklearn.linear_model import LogisticRegression
    m=CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda().eval()
    for p in m.parameters(): p.requires_grad=False
    tf=T.Compose([T.Resize((224,224)),T.ToTensor(),
                  T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    @torch.no_grad()
    def emb(paths):
        out=[]
        for i,p in enumerate(paths):
            p = p[0] if isinstance(p, tuple) else p
            x=tf(Image.open(p).convert("RGB")).unsqueeze(0).cuda()
            o=m.vision_model(pixel_values=x)
            out.append(m.visual_projection(o.pooler_output)[0].float().cpu().numpy())
            if (i+1)%3000==0: print(f"    {i+1}/{len(paths)}",flush=True)
        return np.stack(out)

    # 训练特征（APDE train，抽样）
    tr=json.load(open('/home/oyp/fastdata/apde_train_jpg/index.json'))
    pos=[r for r in tr if r['label']==1]; neg=[r for r in tr if r['label']==0]
    import random; rng=random.Random(0); rng.shuffle(pos)
    n_pos=a.train_limit//2
    sub=pos[:n_pos]+neg[:n_pos]
    cache_f=f"{CACHE}/clip_apde_train_{len(sub)}.npz"
    if os.path.exists(cache_f):
        z=np.load(cache_f); Xtr,ytr=z['X'],z['y']; print(f"  [train] 复用缓存 {Xtr.shape}",flush=True)
    else:
        print(f"  [train] 提取 {len(sub)} 张 CLIP 特征",flush=True)
        Xtr=emb([r['path'] for r in sub]); ytr=np.array([r['label'] for r in sub])
        np.savez_compressed(cache_f,X=Xtr,y=ytr)
    # 评测特征
    def eval_set(key,paths):
        f=f"{CACHE}/clip_{key}.npz"
        if os.path.exists(f):
            z=np.load(f); return z['X'],z['y']
        print(f"  [{key}] 提取 {len(paths)} 张",flush=True)
        X=emb(paths); y=np.array([lab for _,lab in paths])
        np.savez_compressed(f,X=X,y=y); return X,y

    pv=[(f"{EVAL}/pos/{f}",1) for f in sorted(os.listdir(f"{EVAL}/pos"))]
    S3=["casia_au","coco_val","inria_noped"]
    nv=[(f"{EVAL}/{s}/{f}",0) for s in S3+["columbia_au"] for f in sorted(os.listdir(f"{EVAL}/{s}"))]
    Xa,ya=eval_set("apricot_all",pv+nv)
    apde_pos=[(f"{TVC}/apde_imgs/pos/{f}",1) for f in sorted(os.listdir(f"{TVC}/apde_imgs/pos"))]
    apde_neg=[(f"{TVC}/apde_imgs/neg/{f}",0) for f in sorted(os.listdir(f"{TVC}/apde_imgs/neg"))]
    Xd,yd=eval_set("apde_all",apde_pos+apde_neg)

    mu,sd=Xtr.mean(0),Xtr.std(0)+1e-6
    res={"train_n":len(sub),"clip_dim":Xtr.shape[1]}
    for C in (0.01,0.1,1.0):
        clf=LogisticRegression(max_iter=3000,C=C,class_weight="balanced")
        clf.fit((Xtr-mu)/sd,ytr)
        sc=lambda X: clf.decision_function((X-mu)/sd)
        # 跨集 APRICOT
        sa=sc(Xa); npos=len(pv); p=sa[:npos]; n_all=sa[npos:]
        ap={}
        for al in (0.05,0.01,0.001):
            thr=float(np.percentile(n_all,100*(1-al))); ap[f"@{al*100:g}pct"]=round(float((p>=thr).mean()),4)
        # 跨族 APDE
        sd_=sc(Xd); npos2=len(apde_pos); pd=sd_[:npos2]; nd=sd_[npos2:]
        ad={}
        for al in (0.05,0.01):
            thr=float(np.percentile(nd,100*(1-al))); ad[f"@{al*100:g}pct"]=round(float((pd>=thr).mean()),4)
        res[f"C={C}"]={"cross_dataset_apricot":ap,"cross_family_apde":ad}
        print(f"  C={C}: 跨集 APRICOT@5%={ap['@5pct']:.3f} | 跨族 APDE@5%={ad['@5pct']:.3f}",flush=True)
    json.dump(res,open(a.out,"w"),indent=1,ensure_ascii=False)
    print(f"\n判定标准：CLIP 跨集 >= 0.6 → 双分支成立")
    print(f"→ {a.out}")

if __name__=="__main__": main()
