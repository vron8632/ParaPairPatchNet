#!/usr/bin/env python3
"""
MoE 机制诊断（依据 Chi et al. NeurIPS'22 / 2505.00792 / DeepSeek 2408.15664）
============================================================================
对每个 ckpt 计算文献标准指标：
  ① 专家负载熵 ELE = -Σ p_i log p_i          (健康: 高, 最大 log2=0.693 时为 2 专家均衡)
  ② MaxVio = (max_load - mean_load)/mean_load (健康: 低)
  ③ per-token 路由熵 -Σ w log w               (健康: 低, 表示路由自信)
  ④ 路由决策的族间可分性（不同族的 argmax 专家分布是否不同）
  ⑤ 多 seed 路由一致性（同一样本在不同 seed 下 argmax 是否一致）
假设: moe 单用 ELE 低/MaxVio 高（负载坍塌）→ 路由不稳 → UPC 失败
      moe_att 的 attention 提高路由置信度、降低负载偏差
"""
import os, json, argparse, math
import numpy as np
from PIL import Image
WS="/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard"
TVC=f"{WS}/results/tvc"; EVAL=f"{TVC}/unified_eval_set"; CACHE=f"{TVC}/feat_cache"
TEST_JPG="/home/oyp/fastdata/apde_test_jpg"

def fit(im,s): return im if im.size==(s,s) else im.resize((s,s),Image.BILINEAR)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpts",nargs="+",required=True)
    ap.add_argument("--labels",nargs="+",required=True)
    ap.add_argument("--out",default=f"{TVC}/moe_mechanism.json")
    a=ap.parse_args()
    assert len(a.ckpts)==len(a.labels)
    import torch, torch.nn as nn, open_clip
    from torchvision import models, transforms as T
    mc,_,pre=open_clip.create_model_and_transforms("MobileCLIP2-S0",pretrained="dfndr2b",
                                                   cache_dir="/home/oyp/.cache/huggingface")
    mc=mc.cuda().eval()
    te=json.load(open(f"{TEST_JPG}/index.json"))
    pv=[{"path":f"{EVAL}/pos/{f}","label":1} for f in sorted(os.listdir(f"{EVAL}/pos"))]
    S3=["casia_au","coco_val","inria_noped"]
    nv=[{"path":f"{EVAL}/{s_}/{f}","label":0} for s_ in S3+["columbia_au"] for f in sorted(os.listdir(f"{EVAL}/{s_}"))]
    rows=te+pv+nv
    # 预提 CLIP 特征（复用缓存）
    cf_cache=f"{CACHE}/moe_diag_feats.npz"
    if os.path.exists(cf_cache):
        z=np.load(cf_cache); CF=z["CF"]
    else:
        out=[]
        for i,r in enumerate(rows):
            im=Image.open(r["path"]).convert("RGB")
            with torch.no_grad(): f=mc.encode_image(pre(im).unsqueeze(0).cuda())
            out.append(f[0].float().cpu().numpy())
            if (i+1)%3000==0: print(f"    feat {i+1}/{len(rows)}",flush=True)
        CF=np.stack(out).astype(np.float32); np.savez_compressed(cf_cache,CF=CF)
    print(f"  样本 {len(rows)}  CLIP 特征 {CF.shape}",flush=True)

    results={}
    route_matrix={}
    for ck,lbl in zip(a.ckpts,a.labels):
        c=torch.load(ck,map_location="cuda",weights_only=False)
        fmode=c.get("fmode","?"); cdim=c.get("cdim",512); size=c.get("size",768)
        sd=c["state_dict"]
        if "router.weight" not in sd and not any("router" in k for k in sd):
            print(f"  ⊘ {lbl}: 无 router（fmode={fmode}），跳过"); continue
        import torch.nn as nn
        from torchvision import models
        b=models.shufflenet_v2_x1_0(weights=None); td=b.fc.in_features; b.fc=nn.Identity()
        e1=nn.Sequential(nn.LayerNorm(td),nn.Linear(td,256),nn.ReLU())
        e2=nn.Sequential(nn.LayerNorm(cdim),nn.Linear(cdim,256),nn.ReLU())
        router=nn.Sequential(nn.LayerNorm(td+cdim),nn.Linear(td+cdim,64),nn.ReLU(),nn.Linear(64,2))
        for n_,mod in [("tex",b),("e1",e1),("e2",e2),("router",router)]:
            sub={k[len(n_)+1:]:v for k,v in sd.items() if k.startswith(n_+".")}
            if sub: mod.load_state_dict(sub)
        b=b.cuda().eval(); e1=e1.cuda().eval(); e2=e2.cuda().eval(); router=router.cuda().eval()
        tf=T.Compose([T.Resize((size,size)),T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        W=[]
        with torch.no_grad():
            for i,r in enumerate(rows):
                im=Image.open(r["path"]).convert("RGB")
                t=b(tf(im).unsqueeze(0).cuda())
                cf=torch.from_numpy(CF[i]).unsqueeze(0).cuda()
                w=torch.softmax(router(torch.cat([t,cf],1)),1)[0].cpu().numpy()
                W.append(w)
        W=np.stack(W)                                  # [N,2]
        # ① ELE（全样本负载熵）
        load=W.mean(0); ele=float(-(load*np.log(load+1e-12)).sum())
        # ② MaxVio
        maxvio=float((load.max()-load.mean())/load.mean())
        # ③ per-token 路由熵（均值）
        pte=float(np.mean(-(W*np.log(W+1e-12)).sum(1)))
        # ④ 各族 argmax 专家分布
        fam_arg={}
        n_te=len(te); n_pv=len(pv)
        idx_pos=[i for i,r in enumerate(te) if r["label"]==1]
        for f in sorted({r["method"] for r in te if r["label"]==1}):
            sel=[i for i in idx_pos if te[i]["method"]==f]
            if sel: fam_arg[f]={"expert1_frac":round(float((W[sel].argmax(1)==0).mean()),3)}
        res={"fmode":fmode,"n_samples":len(rows),"ELE":round(ele,4),"ELE_max":round(math.log(2),4),
             "MaxVio":round(maxvio,4),"per_token_entropy":round(pte,4),
             "load_expert1":round(float(load[0]),4),"load_expert2":round(float(load[1]),4),
             "family_expert1_frac":fam_arg}
        # APRICOT 子集
        ap_sel=range(n_te, n_te+n_pv)
        res["apricot_expert1_frac"]=round(float((W[list(ap_sel)].argmax(1)==0).mean()),3)
        results[lbl]=res; route_matrix[lbl]=(W.argmax(1))
        print(f"  {lbl:22s} ELE={ele:.4f}/{math.log(2):.4f}  MaxVio={maxvio:.4f}  "
              f"per-token熵={pte:.4f}  负载=[{load[0]:.3f},{load[1]:.3f}]",flush=True)
        del b,e1,e2,router; torch.cuda.empty_cache()

    # ⑤ 多 seed 路由一致性（同 fmode 的 ckpt 两两比较）
    print("\n  === 多 seed 路由一致性（argmax 一致率）===")
    cons={}
    labs=list(route_matrix.keys())
    for i in range(len(labs)):
        for j in range(i+1,len(labs)):
            A,B=route_matrix[labs[i]],route_matrix[labs[j]]
            if len(A)==len(B):
                r=float((A==B).mean()); cons[f"{labs[i]}|{labs[j]}"]=round(r,4)
                print(f"    {labs[i]:18s} vs {labs[j]:18s} {r:.4f}")
    results["_consistency"]=cons
    json.dump(results,open(a.out,"w"),indent=1,ensure_ascii=False)
    print(f"\n→ {a.out}")

if __name__=="__main__": main()
