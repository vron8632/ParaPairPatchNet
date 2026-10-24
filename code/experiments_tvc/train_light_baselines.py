#!/usr/bin/env python3
"""
轻量 baseline 训练与统一评估 —— 回应"为什么不用轻量检测器/分类器"
====================================================================
模型：
  A. ResNet18 (ImageNet 预训练, fc→2)          —— 经典 CNN 分类器
  B. ShuffleNetV2 x1.0 (同上)                  —— 极轻量移动端分类器
  C. CLIP ViT-B/32 冻结特征 + LogisticRegression —— 线性探针式微调
评估协议（与主表完全一致）：
  正：APRICOT 100 张真实打印补丁
  负：CASIA/COCO/INRIA 各 60 张真实照片（seed 42，同 confound 协议）
  阈值：负样本分位数 → TPR@FPR
输出：results/tvc/light_baselines.json
"""
import os, sys, json, time, shutil, math
import numpy as np

WS = "/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard"
LORA_DATA = f"{WS}/results/tvc/lora_data"
TVC = f"{WS}/results/tvc"
BASE_IMG = f"{WS}/results/tvc/baseline_imgs"
AP = "/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard/data/apricot/dev/data_mask_v2"
D = "/media/oyp/工作/Projects/042_image_forensic/DuetGuard/data"
S3 = ["casia_au", "coco_val", "inria_noped"]
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")

# ═══════════════ Part 1: 导出统一评估图像 ═══════════════
def export_eval_images(n_pos=100, n_neg_per=60, seed=42):
    import torch
    from PIL import Image
    os.makedirs(f"{BASE_IMG}/pos", exist_ok=True)
    for s in S3:
        os.makedirs(f"{BASE_IMG}/{s}", exist_ok=True)
    # 正：APRICOT
    files = sorted(os.listdir(AP), key=lambda f: int(f.split("_")[1].split(".")[0]))
    cnt = 0
    for f in files:
        if cnt >= n_pos: break
        out = f"{BASE_IMG}/pos/apricot_{cnt:03d}.jpg"
        if not os.path.exists(out):
            x = torch.load(os.path.join(AP, f), map_location="cpu", weights_only=False)
            img = np.clip(x["Image"][0].astype(np.float32)*255, 0, 255).astype(np.uint8)
            Image.fromarray(img).resize((512, int(512*img.shape[0]/img.shape[1])), Image.LANCZOS).save(out, quality=92)
        cnt += 1
    print(f"  pos 导出 {cnt}", flush=True)
    # 负：3 源（seed 42，与 confound 协议一致）
    rng = np.random.RandomState(seed)
    au = f"{D}/casia2/CASIA2.0_revised/Au"
    fs = sorted(f for f in os.listdir(au) if f.lower().endswith(IMG_EXT))
    for i in rng.permutation(len(fs))[:n_neg_per]:
        dst = f"{BASE_IMG}/casia_au/casia_{i:03d}.jpg"
        if not os.path.exists(dst): shutil.copy(os.path.join(au, fs[i]), dst)
    coco = f"{D}/coco/val2017"
    fs = sorted(f for f in os.listdir(coco) if f.lower().endswith(IMG_EXT))
    for i in rng.permutation(len(fs))[:n_neg_per]:
        dst = f"{BASE_IMG}/coco_val/coco_{i:03d}.jpg"
        if not os.path.exists(dst): shutil.copy(os.path.join(coco, fs[i]), dst)
    inria = f"{D}/inria/data_ped/no_pedestrians"
    fs = sorted(f for f in os.listdir(inria) if f.lower().endswith(IMG_EXT))
    for i in rng.permutation(len(fs))[:n_neg_per]:
        dst = f"{BASE_IMG}/inria_noped/inria_{i:03d}.jpg"
        if not os.path.exists(dst): shutil.copy(os.path.join(inria, fs[i]), dst)
    for s in S3:
        n = len(os.listdir(f"{BASE_IMG}/{s}"))
        print(f"  {s}: {n}", flush=True)

# ═══════════════ Part 2: 训练 CNN 分类器 ═══════════════
def train_cnn(model_name="resnet18", epochs=10, bs=64):
    import torch, torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms, models
    import torchvision
    dev = "cuda"
    tf_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    tf_eval = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    rows = json.load(open(f"{LORA_DATA}/index.json"))
    tr = [(r["path"], r["label"]) for r in rows if r["split"]=="train"]
    class DS(Dataset):
        def __init__(self, items, tf): self.items, self.tf = items, tf
        def __len__(self): return len(self.items)
        def __getitem__(self, i):
            from PIL import Image
            p, y = self.items[i]
            return self.tf(Image.open(p).convert("RGB")), y
    dl = DataLoader(DS(tr, tf_train), batch_size=bs, shuffle=True,
                    num_workers=4, pin_memory=True)

    if model_name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, 2)
    elif model_name == "shufflenetv2":
        m = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, 2)
    m = m.to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    for ep in range(epochs):
        m.train(); tot = 0
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); loss = crit(m(x), y)
            loss.backward(); opt.step(); tot += float(loss)
        sched.step()
        print(f"  [{model_name}] ep{ep}: loss={tot/max(len(dl),1):.4f}", flush=True)
    torch.save(m.state_dict(), f"{TVC}/{model_name}_patchcls.pt")
    return m, tf_eval

def cnn_scores(model, tf_eval, img_dir, device="cuda"):
    import torch
    import torch
    from PIL import Image
    scores = []
    files = sorted(os.listdir(img_dir))
    for f in files:
        im = Image.open(os.path.join(img_dir, f)).convert("RGB")
        x = tf_eval(im).unsqueeze(0).to(device)
        logits = model(x)
        p = torch.softmax(logits, 1)[0, 1].item()   # P(patch)
        scores.append(p)
    return scores

# ═══════════════ Part 3: CLIP 冻结特征 + LogisticRegression ═══════════════

def _clip_feat(clip, proc, ims):
    import torch
    x = proc(images=ims, return_tensors="pt").to("cuda")
    f = clip.get_image_features(pixel_values=x["pixel_values"])
    if not torch.is_tensor(f):   # transformers >= 5: BaseModelOutputWithPooling
        f = f.pooler_output
    return (f / f.norm(dim=-1, keepdim=True)).float()

def clip_linear(n_train_max=1500):
    import torch, os
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from PIL import Image
    from transformers import CLIPModel, CLIPImageProcessor
    from sklearn.linear_model import LogisticRegression
    dev = "cuda"
    proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(dev).eval()

    def feat(paths, bs=64):
        fs = []
        with torch.no_grad():
            for i in range(0, len(paths), bs):
                ims = [Image.open(p).convert("RGB") for p in paths[i:i+bs]]
                fs.append(_clip_feat(clip, proc, ims).cpu())
        return torch.cat(fs).numpy()

    rows = json.load(open(f"{LORA_DATA}/index.json"))
    tr = [(r["path"], r["label"]) for r in rows if r["split"]=="train"]
    Xtr = feat([p for p,_ in tr]); ytr = np.array([y for _,y in tr])
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr)
    print(f"  [clip-linear] train acc: {clf.score(Xtr, ytr):.3f}", flush=True)
    return clip, proc, clf

def clip_score_dir(clip, proc, clf, img_dir):
    import torch
    from PIL import Image
    import numpy as np
    files = sorted(os.listdir(img_dir))
    feats = []
    with torch.no_grad():
        for i in range(0, len(files), 64):
            ims = [Image.open(os.path.join(img_dir, f)).convert("RGB") for f in files[i:i+64]]
            feats.append(_clip_feat(clip, proc, ims).cpu())
    X = torch.cat(feats).numpy()
    return clf.predict_proba(X)[:, 1]

# ═══════════════ Part 4: 统一评估 ═══════════════
def evaluate(all_scores, out_path):
    pos = np.array(all_scores["NP-PatchLM_ours"]["_pos"]) if False else None
    res = {"protocol": {"pos": "APRICOT 100", "neg": "CASIA/COCO/INRIA 60 each (seed 42)"}}
    for name, dd in all_scores.items():
        pos = np.array(dd["pos"]); neg = np.concatenate([np.array(v) for v in dd["neg"].values()])
        row = {"pos_mean": round(float(pos.mean()),4), "neg_mean": round(float(neg.mean()),4)}
        for a in (0.01, 0.05, 0.10, 0.20):
            thr = float(np.percentile(neg, 100*(1-a)))
            row[f"TPR@{int(a*100)}pct"] = round(float((pos >= thr).mean()), 4)
        res[name] = row
        print(f"  {name:22s} @1%={row['TPR@1pct']} @5%={row['TPR@5pct']} @10%={row['TPR@10pct']} @20%={row['TPR@20pct']}")
    json.dump(res, open(out_path, "w"), indent=1, ensure_ascii=False)
    print(f"  → {out_path}")

# ═══════════════ main ═══════════════
if __name__ == "__main__":
    import torch
    print("=== Part 1: 导出统一评估图像 ===", flush=True)
    export_eval_images()
    print("=== Part 2: 训练 CNN 分类器 ===", flush=True)
    all_scores = {}
    for name in ("resnet18", "shufflenetv2"):
        m, tf = train_cnn(name)
        all_scores[name] = {
            "pos": cnn_scores(m, tf, f"{BASE_IMG}/pos"),
            "neg": {s: cnn_scores(m, tf, f"{BASE_IMG}/{s}") for s in S3}}
        del m; torch.cuda.empty_cache()
    print("=== Part 3: CLIP 冻结特征 + 线性头 ===", flush=True)
    try:
        clip, proc, clf = clip_linear()
        all_scores["clip_linear"] = {
            "pos": clip_score_dir(clip, proc, clf, f"{BASE_IMG}/pos"),
            "neg": {s: clip_score_dir(clip, proc, clf, f"{BASE_IMG}/{s}") for s in S3}}
        del clip; torch.cuda.empty_cache()
    except Exception as e:
        print(f"  [warn] CLIP 线性头失败（跳过）: {e}", flush=True)
    print("=== Part 4: 统一评估 ===", flush=True)
    # 只把 clip_linear 单独评估写入 clip_linear.json（CNN 已在 light_baselines.json）
    clip_only = {k: v for k, v in all_scores.items() if k == "clip_linear"}
    if clip_only:
        evaluate(clip_only, f"{TVC}/clip_linear.json")
    print("LIGHT BASELINES ALL DONE")
