#!/usr/bin/env python3
"""评测 P1 checkpoint：同域(APRICOT) + 跨域(APDE 族级) × 三个工作点(5%/1%/0.1%)"""
import os, sys, json, argparse
import numpy as np
from PIL import Image
WS="/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard"
TVC=f"{WS}/results/tvc"; EVAL=f"{TVC}/unified_eval_set"; APDE=f"{TVC}/apde_imgs"

def fit(im,size):
    w,h=im.size; s=size/max(w,h)
    return im if abs(s-1)<1e-6 else im.resize((max(1,int(w*s)),max(1,int(h*s))),Image.BILINEAR)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--ckpt",required=True)
    ap.add_argument("--out",default=f"{TVC}/p1_domains.json")
    a=ap.parse_args()
    import torch, torch.nn as nn
    from torchvision import transforms, models
    ck=torch.load(a.ckpt, map_location="cuda", weights_only=False)
    size=ck.get("size",768)
    m=models.shufflenet_v2_x1_0(weights=None); m.fc=nn.Linear(m.fc.in_features,2)
    m.load_state_dict(ck["state_dict"]); m=m.cuda().eval()
    tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    @torch.no_grad()
    def sc(paths):
        out=[]
        for p in paths:
            x=tf(fit(Image.open(p).convert("RGB"),size)).unsqueeze(0).cuda()
            out.append(float(torch.softmax(m(x),1)[0,1]))
        return np.array(out)
    res={"ckpt":a.ckpt,"size":size,"n_train":ck.get("n_train")}
    S3=["casia_au","coco_val","inria_noped"]
    p=sc([f"{EVAL}/pos/{f}" for f in sorted(os.listdir(f"{EVAL}/pos"))])
    ns={s:sc([f"{EVAL}/{s}/{f}" for f in sorted(os.listdir(f"{EVAL}/{s}"))]) for s in S3+["columbia_au"]}
    for scope,keys in (("3src",S3),("4src",S3+["columbia_au"])):
        n_all=np.concatenate([ns[k] for k in keys]); rr={}
        for al in (0.05,0.01,0.001):
            thr=float(np.percentile(n_all,100*(1-al)))
            rr[f"@{al*100:g}pct"]={"thr":round(thr,4),"tpr":round(float((p>=thr).mean()),4)}
        res[f"apricot_{scope}"]=rr
    # APDE 分族
    pa=sc([f"{APDE}/pos/{f}" for f in sorted(os.listdir(f"{APDE}/pos"))])
    na=sc([f"{APDE}/neg/{f}" for f in sorted(os.listdir(f"{APDE}/neg"))])
    fams={}
    for f,x in zip(sorted(os.listdir(f"{APDE}/pos")),pa): fams.setdefault(f.split("_")[0],[]).append(x)
    apde={}
    for al in (0.05,0.01,0.001):
        thr=float(np.percentile(na,100*(1-al)))
        apde[f"@{al*100:g}pct"]={"thr":round(thr,4),"overall":round(float((pa>=thr).mean()),4),
                                 "by_family":{k:round(float((np.array(v)>=thr).mean()),4) for k,v in fams.items()}}
    res["apde"]=apde
    json.dump(res,open(a.out,"w"),indent=1,ensure_ascii=False)
    print("\n=== 同域 APRICOT ===")
    for k in ("apricot_3src","apricot_4src"):
        print(f"  {k}: "+" | ".join(f"{w}={v['tpr']:.3f}" for w,v in res[k].items()))
    print("\n=== 跨域 APDE（混合 13 族，注意与族级留一不同）===")
    for w,v in apde.items(): print(f"  {w}: overall={v['overall']:.3f}")
    print(f"\n→ {a.out}")

if __name__=="__main__": main()
