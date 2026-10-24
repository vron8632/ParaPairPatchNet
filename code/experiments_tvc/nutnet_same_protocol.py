#!/usr/bin/env python3
"""
NutNet (CCS'24) 同协议评测：APDE 13 族（族级）+ APRICOT
=========================================================
适配依据 code/third_party/NutNet/patchFilter_upstream.py:
  box_num=32 → AutoEncoder32 + ae_weights/n_13.pth（输入 13×13, 输出 13×13）
  流程: img∈[0,1] → *2-1 → 切 32×32 个 13×13 块 → AE 重建 → 每块 MSE
  图像级分数（两种）:
    (a) max_mse     : 所有块的最大重建误差
    (b) frac_gt     : 重建误差 >0.125 的块占比（官方阈值）
只评测"检测"能力（不含其 YOLO 抹除步骤），与我们的判据同口径。
"""
import os, sys, json, argparse
import numpy as np
from PIL import Image
WS="/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard"
TVC=f"{WS}/results/tvc"; EVAL=f"{TVC}/unified_eval_set"
NUT=f"{WS}/code/third_party/NutNet"
TEST_JPG="/home/oyp/fastdata/apde_test_jpg"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default=f"{TVC}/nutnet_same_protocol.json")
    ap.add_argument("--weight",default="n_13",choices=["n_13","n_26"],help="n_13=13x13 块(32 块), n_26=26x26 块(16 块)")
    a=ap.parse_args()
    import torch, torch.nn as nn
    # 内联 AutoEncoder32（避免引入 patchFilter_upstream 的 YOLO 依赖树）
    class AutoEncoder32(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder=nn.Sequential(
                nn.Conv2d(3,8,kernel_size=2,stride=1,padding=1), nn.ReLU(),
                nn.Conv2d(8,16,kernel_size=2,stride=2,padding=1), nn.ReLU(),
                nn.Conv2d(16,32,kernel_size=2,stride=2,padding=1), nn.ReLU())
            self.decoder=nn.Sequential(
                nn.ConvTranspose2d(32,16,kernel_size=2,stride=2,padding=1), nn.ReLU(),
                nn.ConvTranspose2d(16,8,kernel_size=2,stride=2,padding=1), nn.ReLU(),
                nn.ConvTranspose2d(8,3,kernel_size=2,stride=1,padding=1), nn.Tanh())
        def forward(self,x): return self.decoder(self.encoder(x))
    class AutoEncoder16(nn.Module):
        """box_num=16 → 块边长 26px；结构与上游 patchFilter_upstream.py 一致"""
        def __init__(self):
            super().__init__()
            self.encoder=nn.Sequential(
                nn.Conv2d(3,8,kernel_size=2,stride=2,padding=1), nn.ReLU(),
                nn.Conv2d(8,16,kernel_size=2,stride=2,padding=1), nn.ReLU(),
                nn.Conv2d(16,32,kernel_size=2,stride=2,padding=1), nn.ReLU())
            self.decoder=nn.Sequential(
                nn.ConvTranspose2d(32,16,kernel_size=2,stride=2,padding=1), nn.ReLU(),
                nn.ConvTranspose2d(16,8,kernel_size=2,stride=2,padding=1), nn.ReLU(),
                nn.ConvTranspose2d(8,3,kernel_size=2,stride=2,padding=1), nn.Tanh())
        def forward(self,x): return self.decoder(self.encoder(x))
    ae=(AutoEncoder32() if a.weight=="n_13" else AutoEncoder16()).cuda().eval()
    ae.load_state_dict(torch.load(f"{NUT}/ae_weights/{a.weight}.pth",map_location="cuda"))
    for p in ae.parameters(): p.requires_grad=False
    loss_fn=nn.MSELoss(reduction="none")
    INPUT=416
    BOX=32 if a.weight=="n_13" else 16      # 块数（32 或 16）
    BL=INPUT//BOX                            # 块边长：416/32=13 或 416/16=26

    @torch.no_grad()
    def score(paths):
        maxes=[]; fracs=[]
        for i,p in enumerate(paths):
            im=Image.open(p).convert("RGB").resize((INPUT,INPUT),Image.BILINEAR)
            x=torch.from_numpy(np.asarray(im)).float().permute(2,0,1).unsqueeze(0).cuda()/255.0
            x=x*2-1                                     # 官方约定 [-1,1]
            # 切块：[1,3,416,416] → [1,3,32,13,32,13]
            blk=x.unfold(2,BL,BL).unfold(3,BL,BL)
            blk=blk.permute(0,2,4,1,3,5).contiguous().view(-1,3,BL,BL)
            rec=ae(blk)
            mse=torch.mean(loss_fn(rec,blk),dim=(1,2,3))
            maxes.append(float(mse.max()))
            fracs.append(float((mse>0.125).float().mean()))
            if (i+1)%3000==0: print(f"    nutnet {i+1}/{len(paths)}",flush=True)
        return np.array(maxes), np.array(fracs)

    te=json.load(open(f"{TEST_JPG}/index.json"))
    te_pos=[r["path"] for r in te if r["label"]==1]
    te_neg=[r["path"] for r in te if r["label"]==0]
    pos_m=np.array([r["method"] for r in te if r["label"]==1])
    pv=[f"{EVAL}/pos/{f}" for f in sorted(os.listdir(f"{EVAL}/pos"))]
    S3=["casia_au","coco_val","inria_noped"]
    nv=[f"{EVAL}/{s}/{f}" for s in S3+["columbia_au"] for f in sorted(os.listdir(f"{EVAL}/{s}"))]

    res={}
    print("  [APDE 族级] ...",flush=True)
    mxp,fp=score(te_pos); mxn,fn=score(te_neg)
    print("  [APRICOT] ...",flush=True)
    mxA,fA=score(pv); mxN,fN=score(nv)
    seen={"advpatch","DM-NAP","GNAP","TCA","TCEGA","tsea","tsea-mim","tsea-pgd"}
    unseen={"AA","AdvCloak","AdvSticker","AdvTshirt","UPC"}
    for tag,sp,sn,pm,fam_mask in [("apde",mxp,mxn,pos_m,True),("apricot",mxA,mxN,None,False)]:
        thr=float(np.percentile(sn,95))
        if fam_mask:
            fam={f:round(float((sp[pm==f]>=thr).mean()),4) for f in sorted(set(pm))}
            se=[v for k,v in fam.items() if k in seen]; un=[v for k,v in fam.items() if k in unseen]
            res[tag]={"score":"max_mse","@5pct":round(float((sp>=thr).mean()),4),
                      "seen_mean":round(float(np.mean(se)),4),"unseen_mean":round(float(np.mean(un)),4),
                      "by_family":fam}
        else:
            res[tag]={"score":"max_mse","@5pct":round(float((sp>=thr).mean()),4),
                      "@1pct":round(float((sp>=np.percentile(sn,99)).mean()),4)}
    # 官方阈值口径（frac>0.125 的块占比）也报一版
    for tag,sp,sn in [("apde",fp,fn),("apricot",fA,fN)]:
        thr=float(np.percentile(sn,95))
        res[tag+"_frac"]={"@5pct":round(float((sp>=thr).mean()),4)}
    print(f"\n  ✅ NutNet max_mse:  APDE 未见={res['apde']['unseen_mean']} UPC={res['apde']['by_family'].get('UPC')} | "
          f"APRICOT@5%={res['apricot']['@5pct']} @1%={res['apricot']['@1pct']}")
    print(f"     NutNet frac>0.125: APDE@5%={res['apde_frac']['@5pct']} | APRICOT@5%={res['apricot_frac']['@5pct']}")
    json.dump(res,open(a.out,"w"),indent=1,ensure_ascii=False)
    print(f"→ {a.out}")

if __name__=="__main__": main()
