#!/usr/bin/env python3
"""
A) SOTA 同协议对比：NutNet / SAC / AdvPatchXAI（有预训练权重）
在 APDE test 13 族（族级）+ APRICOT 上评测，与我们的 G2 直接可比。
每个方法的推理接口通过 adapter 函数适配。
"""
import os, sys, json, argparse, importlib
os.environ.setdefault("HF_HOME","/home/oyp/.cache/huggingface")
import numpy as np
from PIL import Image
WS="/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard"
TVC=f"{WS}/results/tvc"; EVAL=f"{TVC}/unified_eval_set"
TEST_JPG="/home/oyp/fastdata/apde_test_jpg"

def wilson(k,n,z=1.96):
    if n==0: return (0.,0.)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (round(max(0.,c-h),4), round(min(1.,c+h),4))
import math

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--methods",default="advpatchxai")
    ap.add_argument("--out",default=f"{TVC}/sota_same_protocol.json")
    a=ap.parse_args()
    sys.path.insert(0,f"{WS}/code")
    te=json.load(open(f"{TEST_JPG}/index.json"))
    pos_m=np.array([r["method"] for r in te if r["label"]==1])
    te_pos=[r["path"] for r in te if r["label"]==1]
    te_neg=[r["path"] for r in te if r["label"]==0]
    pv=[f"{EVAL}/pos/{f}" for f in sorted(os.listdir(f"{EVAL}/pos"))]
    S3=["casia_au","coco_val","inria_noped"]
    nv=[f"{EVAL}/{s}/{f}" for s in S3+["columbia_au"] for f in sorted(os.listdir(f"{EVAL}/{s}"))]
    res={}
    for name in a.methods.split(","):
        name=name.strip()
        try:
            if name=="advpatchxai":
                # AdvPatchXAI r50_v2（官方灰度 224 transform）
                sys.path.insert(0,f"{WS}/code/third_party/AdvPatchXAI")
                ck="/home/oyp/fastdata/advpatchxai_runs/r50_v2/checkpoints/net_trained"
                import torch
                from get_network import get_network
                net=get_network("ResNet").cuda().eval()
                sd=torch.load(ck,map_location="cuda",weights_only=False)
                sd=sd.get("model_state_dict",sd)
                sd={k.replace("module.",""):v for k,v in sd.items()}
                net.load_state_dict(sd)
                import torchvision.transforms as T
                tf=T.Compose([T.Grayscale(num_output_channels=1),T.Resize((224,224)),T.ToTensor()])
                @torch.no_grad()
                def infer(paths):
                    out=[]
                    for i,p in enumerate(paths):
                        x=tf(Image.open(p).convert("RGB")).unsqueeze(0).cuda()
                        if x.shape[1]==1: x=x.repeat(1,3,1,1)
                        lg=net(x)
                        pr=torch.softmax(lg,1)[0,1] if lg.shape[1]>1 else torch.sigmoid(lg)[0,0]
                        out.append(float(pr))
                    return np.array(out)
            else:
                print(f"  ⊘ {name}: adapter 未实现（NutNet/SAC 需各自接口），跳过"); continue
            sp=infer(te_pos); sn=infer(te_neg)
            thr=float(np.percentile(sn,95))
            fam={f:round(float((sp[pos_m==f]>=thr).mean()),4) for f in sorted(set(pos_m))}
            unseen=[v for k,v in fam.items() if k in {"AA","AdvCloak","AdvSticker","AdvTshirt","UPC"}]
            seen=[v for k,v in fam.items() if k in {"advpatch","DM-NAP","GNAP","TCA","TCEGA","tsea","tsea-mim","tsea-pgd"}]
            pA=infer(pv); nA=infer(nv)
            ap5=round(float((pA>=np.percentile(nA,95)).mean()),4)
            res[name]={"apde":{"@5pct":round(float((sp>=thr).mean()),4),
                               "seen_mean":round(float(np.mean(seen)),4),
                               "unseen_mean":round(float(np.mean(unseen)),4),"by_family":fam},
                       "apricot@5pct":ap5}
            print(f"  ✅ {name}: APDE未见={res[name]['apde']['unseen_mean']} UPC={fam.get('UPC')} | APRICOT@5%={ap5}",flush=True)
        except Exception as e:
            print(f"  ❌ {name}: {type(e).__name__}: {str(e)[:200]}",flush=True)
    json.dump(res,open(a.out,"w"),indent=1,ensure_ascii=False)
    print(f"→ {a.out}")

if __name__=="__main__": main()
