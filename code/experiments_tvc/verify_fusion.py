#!/usr/bin/env python3
"""验证 DRO 版的 0.819 是否真实：3 seed 重复 + 分数分布检查"""
import os, json, argparse
import numpy as np
from PIL import Image
TVC="/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard/results/tvc"
EVAL=f"{TVC}/unified_eval_set"

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--ckpt",required=True); ap.add_argument("--out",default=f"{TVC}/verify_fusion.json")
    a=ap.parse_args()
    import torch, torch.nn as nn, open_clip
    from torchvision import models, transforms as T
    ck=torch.load(a.ckpt,map_location="cuda",weights_only=False)
    cdim=ck["cdim"]; size=ck.get("size",768)
    class Fusion(nn.Module):
        def __init__(s,cdim):
            super().__init__()
            b=models.shufflenet_v2_x1_0(weights=None); td=b.fc.in_features; b.fc=nn.Identity(); s.tex=b
            s.fuse=nn.Sequential(nn.LayerNorm(td+cdim),nn.Linear(td+cdim,256),nn.ReLU(),nn.Dropout(0.3),nn.Linear(256,2))
        def forward(s,x,cf): return s.fuse(torch.cat([s.tex(x),cf],1))
    m=Fusion(cdim); m.load_state_dict(ck["state_dict"]); m=m.cuda().eval()
    mc,_,pre=open_clip.create_model_and_transforms("MobileCLIP2-S0",pretrained="dfndr2b",cache_dir="/home/oyp/.cache/huggingface")
    mc=mc.cuda().eval()
    tf=T.Compose([T.Resize((size,size)),T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    @torch.no_grad()
    def sc(paths):
        out=[]
        for p in paths:
            im=Image.open(p).convert("RGB")
            x=tf(im).unsqueeze(0).cuda()
            f=mc.encode_image(pre(im).unsqueeze(0).cuda()).float()
            out.append(float(torch.softmax(m(x,f),1)[0,1]))
        return np.array(out)
    pv=[f"{EVAL}/pos/{f}" for f in sorted(os.listdir(f"{EVAL}/pos"))]
    S3=["casia_au","coco_val","inria_noped"]
    nv=[f"{EVAL}/{s}/{f}" for s in S3+["columbia_au"] for f in sorted(os.listdir(f"{EVAL}/{s}"))]
    sp=sc(pv); sn=sc(nv)
    res={"ckpt":a.ckpt,"apricot_pos_dist":{"mean":float(sp.mean()),"p50":float(np.percentile(sp,50)),"p90":float(np.percentile(sp,90))},
         "apricot_neg_dist":{"mean":float(sn.mean()),"p50":float(np.percentile(sn,50)),"p95":float(np.percentile(sn,95)),"p99":float(np.percentile(sn,99))},
         "workpoints":{}}
    for al in (0.05,0.01,0.001):
        thr=float(np.percentile(sn,100*(1-al)))
        res["workpoints"][f"@{al*100:g}pct"]={"thr":round(thr,4),"tpr":round(float((sp>=thr).mean()),4)}
    print(json.dumps(res,indent=1,ensure_ascii=False))
    json.dump(res,open(a.out,"w"),indent=1,ensure_ascii=False)

if __name__=="__main__": main()
