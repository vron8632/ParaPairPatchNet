#!/usr/bin/env python3
"""
方差来源诊断：跨集 TPR 的 27% 标准差从哪来？
==============================================
对同一配置的多个 seed ckpt，分别统计：
  (a) APRICOT 负样本分数分布（均值/分位/标准差）
  (b) 由此得到的阈值
  (c) APRICOT 正样本分数分布
  (d) Wilson CI（138 正样本的统计不确定性下界）

判据：
  - 若 (a)(c) 抖动大 → 模型表征不稳 → 需 EMA/集成/降 lr
  - 若 (a)(c) 稳但 TPR 抖动 → 评测样本太少 → 需扩评测集/报告 CI
"""
import os, json, argparse, itertools
import numpy as np
from PIL import Image
TVC="/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard/results/tvc"
EVAL=f"{TVC}/unified_eval_set"

def wilson(k,n,z=1.96):
    if n==0: return (0.,0.)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (max(0.,c-h), min(1.,c+h))

def main():
    import math, torch, torch.nn as nn, open_clip, sys
    global math
    math=__import__("math")
    from torchvision import models, transforms as T
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--out", default=f"{TVC}/variance_diagnosis.json")
    a=ap.parse_args()
    assert len(a.ckpts)==len(a.labels)
    mc,_,pre=open_clip.create_model_and_transforms("MobileCLIP2-S0",pretrained="dfndr2b",
                                                   cache_dir="/home/oyp/.cache/huggingface")
    mc=mc.cuda().eval()
    tf=T.Compose([T.Resize((768,768)),T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    pv=[f"{EVAL}/pos/{f}" for f in sorted(os.listdir(f"{EVAL}/pos"))]
    S3=["casia_au","coco_val","inria_noped"]
    nv=[f"{EVAL}/{s}/{f}" for s in S3+["columbia_au"] for f in sorted(os.listdir(f"{EVAL}/{s}"))]
    # 预提特征（一次）
    cache=f"{TVC}/feat_cache/diag_apricot.npz"
    if os.path.exists(cache):
        z=np.load(cache); Fa,Fn=z["Fa"],z["Fn"]
    else:
        def emb(paths):
            out=[]
            for i,p in enumerate(paths):
                im=Image.open(p).convert("RGB")
                with torch.no_grad(): f=mc.encode_image(pre(im).unsqueeze(0).cuda())
                out.append(f[0].float().cpu().numpy())
            return np.stack(out).astype(np.float32)
        Fa=emb(pv); Fn=emb(nv); np.savez_compressed(cache,Fa=Fa,Fn=Fn)
    print(f"  特征: pos={Fa.shape} neg={Fn.shape}",flush=True)

    res={}
    for ck,lbl in zip(a.ckpts,a.labels):
        c=torch.load(ck,map_location="cuda",weights_only=False)
        cdim=c["cdim"]; size=c.get("size",768); gating=c.get("gating",False)
        class M(nn.Module):
            def __init__(s,cdim,gating):
                super().__init__()
                b=models.shufflenet_v2_x1_0(weights=None); td=b.fc.in_features; b.fc=nn.Identity(); s.tex=b
                s.gating=gating
                head=nn.Sequential(nn.LayerNorm(td+cdim),nn.Linear(td+cdim,256),nn.ReLU(),nn.Dropout(0.3),nn.Linear(256,2))
                # 两个名字指向同一模块：兼容 p1_fusion_dro(用 fuse) 与 p1_gated_labelfree(用 head) 存的 ckpt
                s.head=head
                if gating:
                    s.gate=nn.Sequential(nn.LayerNorm(td+cdim),nn.Linear(td+cdim,128),nn.ReLU(),nn.Linear(128,1))
            def forward(s,x,cf):
                t=s.tex(x)
                if s.gating:
                    g=torch.sigmoid(s.gate(torch.cat([t,cf],1)))
                    return s.head(torch.cat([g*t,(1-g)*cf],1)), g.squeeze(1)
                return s.head(torch.cat([t,cf],1)), None
        m=M(cdim,gating)
        sd=c["state_dict"]
        # 兼容 fuse/head 两种命名：把 ckpt 的键统一映射到当前模型
        if any(k.startswith("fuse.") for k in sd) and not any(k.startswith("head.") for k in sd):
            sd={k.replace("fuse.","head.") if k.startswith("fuse.") else k: v for k,v in sd.items()}
        m.load_state_dict(sd); m=m.cuda().eval()
        tf2=T.Compose([T.Resize((size,size)),T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        @torch.no_grad()
        def sc(paths,feats):
            out=[];gs=[]
            for p,f in zip(paths,feats):
                im=Image.open(p).convert("RGB")
                o,g=m(tf2(im).unsqueeze(0).cuda(),torch.from_numpy(f).unsqueeze(0).cuda())
                out.append(float(torch.softmax(o,1)[0,1]))
                if g is not None: gs.append(float(g[0]))
            return np.array(out), (np.array(gs) if gs else None)
        sp,gp=sc(pv,Fa); sn,gn=sc(nv,Fn)
        thr=float(np.percentile(sn,95)); k=int((sp>=thr).sum()); n=len(sp)
        lo,hi=wilson(k,n)
        res[lbl]={"neg_mean":round(float(sn.mean()),4),"neg_std":round(float(sn.std()),4),
                  "neg_p95":round(thr,4),"neg_p99":round(float(np.percentile(sn,99)),4),
                  "pos_mean":round(float(sp.mean()),4),"pos_std":round(float(sp.std()),4),
                  "thr":round(thr,4),"tpr@5pct":round(k/n,4),"wilson":[round(lo,4),round(hi,4)],
                  "gate_pos_mean":(round(float(gp.mean()),4) if gp is not None else None),
                  "gate_neg_mean":(round(float(gn.mean()),4) if gn is not None else None)}
        print(f"  {lbl:14s} neg均值={sn.mean():.3f} neg_std={sn.std():.3f} 阈值={thr:.3f} "
              f"pos均值={sp.mean():.3f} TPR={k/n:.3f} CI=[{lo:.3f},{hi:.3f}]",flush=True)
        del m; torch.cuda.empty_cache()

    # 方差分解
    keys=["neg_mean","neg_std","neg_p95","pos_mean","thr","tpr@5pct"]
    print("\n  === 方差来源分解（跨 seed 的标准差）===")
    for k in keys:
        vals=[v[k] for v in res.values() if v[k] is not None]
        if len(vals)>1: print(f"    {k:10s}: std={np.std(vals):.4f}  范围=[{min(vals):.3f},{max(vals):.3f}]")
    res["_summary"]={k:round(float(np.std([v[k] for v in res.values() if isinstance(v,dict) and v.get(k) is not None])),4) for k in keys}
    json.dump(res,open(a.out,"w"),indent=1,ensure_ascii=False)
    print(f"\n→ {a.out}")

if __name__=="__main__": main()
