#!/usr/bin/env python3
"""
轻量 baseline 公平版 —— 修正两处不公平：
  1) 类别不平衡：构建集 train 中 patch:clean = 1800:700 → 加 class weight（与 VLM 的 FPR 约束等效的
     正则方向相反但都是"不偏向正类"的校正；VLM 靠 FPR 约束压正类，CNN 用 class weight）
  2) 评估分辨率：旧版 Resize(256)+CenterCrop(224) 会裁掉边缘补丁；改为等比 resize 至 448 全图（letterbox 填充）
用法: $PY code/experiments_tvc/light_baseline_fair.py resnet18
输出: results/tvc/light_fair_<model>.json
"""
import os, sys, json, time
import numpy as np

WS = "/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard"
TVC = f"{WS}/results/tvc"
BASE_IMG = f"{TVC}/baseline_imgs"
LORA_DATA = f"{TVC}/lora_data"
S3 = ["casia_au", "coco_val", "inria_noped"]


def train_fair(model_name, epochs=20, imgsz=448, bs=32):
    import torch, torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms, models
    from PIL import Image
    dev = "cuda"
    tf = transforms.Compose([
        transforms.Resize(imgsz), transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    rows = json.load(open(f"{LORA_DATA}/index.json"))
    tr = [(r["path"], r["label"]) for r in rows if r["split"] == "train"]
    # 类平衡权重（与 VLM 的 FPR 约束在"防止偏向正类"上同向）
    n1 = sum(1 for _, y in tr if y == 1); n0 = len(tr) - n1
    w = torch.tensor([n1 / len(tr), n0 / len(tr)]).to(dev)   # 正类少给权重？反一下：给多数类低权重
    w = torch.tensor([len(tr) / (2 * n0), len(tr) / (2 * n1)]).to(dev)
    print(f"  class weights: clean={w[0]:.3f} patch={w[1]:.3f} (n_clean={n0}, n_patch={n1})", flush=True)

    class DS(Dataset):
        def __init__(s): s.t = tf
        def __len__(s): return len(tr)
        def __getitem__(s, i):
            p, y = tr[i]
            return s.t(Image.open(p).convert("RGB")), y
    dl = DataLoader(DS(), batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)

    if model_name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, 2)
    else:
        m = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, 2)
    m = m.to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(weight=w)
    for ep in range(epochs):
        m.train(); tot = 0
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); loss = crit(m(x), y); loss.backward(); opt.step()
            tot += float(loss)
        sched.step()
        if (ep + 1) % 5 == 0:
            print(f"  [{model_name}-fair] ep{ep}: loss={tot/max(len(dl),1):.4f}", flush=True)
    torch.save(m.state_dict(), f"{TVC}/{model_name}_fair.pt")
    return m


def eval_fair(m, model_name, imgsz=448):
    """全图等比 resize（pad 到方形），不 center-crop"""
    import torch, torch.nn.functional as F
    from torchvision import transforms
    from PIL import Image
    dev = "cuda"
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def score_dir(d):
        files = sorted(os.listdir(d)); out = []
        for f in files:
            im = Image.open(os.path.join(d, f)).convert("RGB")
            w, h = im.size; s = imgsz / max(w, h)
            im = im.resize((max(1, int(w*s)), max(1, int(h*s))), Image.BILINEAR)
            canvas = Image.new("RGB", (imgsz, imgsz), (127, 127, 127))
            canvas.paste(im, ((imgsz - im.width)//2, (imgsz - im.height)//2))
            x = norm(transforms.ToTensor()(canvas)).unsqueeze(0).to(dev)
            with torch.no_grad():
                p = torch.softmax(m(x), 1)[0, 1].item()
            out.append(p)
        return out

    pos = score_dir(f"{BASE_IMG}/pos")
    neg = {s: score_dir(f"{BASE_IMG}/{s}") for s in S3}
    p = np.array(pos); nall = np.concatenate([np.array(v) for v in neg.values()])
    res = {"model": f"{model_name} fine-tuned (class-weighted, {imgsz}px full-image eval)",
           "n_pos": len(p), "n_neg": len(nall),
           "pos_mean": round(float(p.mean()), 4), "neg_mean": round(float(nall.mean()), 4),
           "pos_scores": [round(float(x), 4) for x in pos],
           "neg_scores_by_source": {k: [round(float(x), 4) for x in v] for k, v in neg.items()}}
    print(f"\n[{model_name}-fair] pos_mean={p.mean():.4f} neg_mean={nall.mean():.4f}")
    for a in (0.01, 0.05, 0.10, 0.20, 0.33):
        thr = float(np.percentile(nall, 100 * (1 - a)))
        tpr = float((p >= thr).mean())
        res[f"TPR@{int(a*100)}pct"] = {"threshold": round(thr, 4), "tpr": round(tpr, 4)}
        print(f"  目标 FPR={int(a*100):2d}%: TPR={tpr:.4f}", flush=True)
    json.dump(res, open(f"{TVC}/light_fair_{model_name}.json", "w"), indent=1, ensure_ascii=False)
    print(f"  → {TVC}/light_fair_{model_name}.json")


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "resnet18"
    m = train_fair(name)
    eval_fair(m, name)
    print(f"{name}-fair ALL DONE")
