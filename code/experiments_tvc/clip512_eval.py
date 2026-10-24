#!/usr/bin/env python3
"""
CLIP@512 对照（扩展实验）
================================
闭合"语义特征必要 → 微调语义模型最优"选型链中的分辨率混淆：
旧 CLIP 线性头用 224px，VLM 用 512px。本脚本用 512px 重提 CLIP 特征并重训线性头。
输出: results/tvc/clip_linear_512.json
"""
import os, sys, json
os.environ["HF_HOME"] = os.environ.get("HF_HOME") or "/home/oyp/.cache/huggingface"
os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT") or "https://hf-mirror.com"
import numpy as np

WS = "/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard"
TVC = f"{WS}/results/tvc"
BASE = f"{TVC}/baseline_imgs"
S3 = ["casia_au", "coco_val", "inria_noped"]
_IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
def _list_imgs(d):
    return sorted(f for f in os.listdir(d) if f.lower().endswith(_IMG_EXT))



def main():
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPImageProcessor
    from sklearn.linear_model import LogisticRegression
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda().eval()
    proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print("CLIP loaded; image_size =", proc.size)

    def feat(d, limit=None):
        files = _list_imgs(d)
        if limit: files = files[:limit]
        out = []
        for i in range(0, len(files), 32):
            ims = [Image.open(os.path.join(d, f)).convert("RGB") for f in files[i:i+32]]
            px = proc(images=ims, size={"height": 512, "width": 512},
                      crop_size={"height": 512, "width": 512},
                      return_tensors="pt").pixel_values.cuda()
            with torch.no_grad():
                # CLIP ViT-B/32 位置编码固定 224；处理 512 输入需插值位置编码
                po = m.vision_model(pixel_values=px, interpolate_pos_encoding=True).pooler_output
                out.append(m.visual_projection(po).cpu())
        return torch.cat(out).numpy()

    # 训练集：lora_data 的 train split（与轻量基线同源）
    idx = json.load(open(f"{TVC}/lora_data/index.json"))
    tr = [r for r in idx if r["split"] == "train"]
    Xtr, ytr = [], []
    for i in range(0, len(tr), 32):
        batch = tr[i:i+32]
        ims = [Image.open(r["path"]).convert("RGB") for r in batch]
        px = proc(images=ims, size={"height": 512, "width": 512},
                  crop_size={"height": 512, "width": 512},
                  return_tensors="pt").pixel_values.cuda()
        with torch.no_grad():
            po = m.vision_model(pixel_values=px, interpolate_pos_encoding=True).pooler_output
            Xtr.append(m.visual_projection(po).cpu())
        ytr += [r["label"] for r in batch]
    Xtr = torch.cat(Xtr).numpy(); ytr = np.array(ytr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
    print("linear head trained on", len(ytr), "samples")

    pos_dir = f"{BASE}/pos_all" if os.path.isdir(f"{BASE}/pos_all") else f"{BASE}/pos"
    pos = clf.predict_proba(feat(pos_dir))[:, 1]
    neg = {s: clf.predict_proba(feat(f"{BASE}/{s}"))[:, 1] for s in S3 + ["columbia_au"]}
    res = {"model": "CLIP ViT-B/32 @512px frozen + logistic head", "n_pos": len(pos),
           "pos_mean": round(float(pos.mean()), 4),
           "pos_scores": [round(float(x), 4) for x in pos],
           "neg_scores_by_source": {k: [round(float(x), 4) for x in v] for k, v in neg.items()}}
    for tag, keys in (("3src", S3), ("4src", S3 + ["columbia_au"])):
        n = np.concatenate([neg[k] for k in keys]); row = {}
        for a in (0.01, 0.05, 0.10, 0.20):
            thr = np.percentile(n, 100*(1-a))
            row[f"@{int(a*100)}pct"] = round(float((pos >= thr).mean()), 4)
        res[tag] = row
        print(f"[{tag}] " + "  ".join(f"{k}={v:.3f}" for k, v in row.items()))
    json.dump(res, open(f"{TVC}/clip_linear_512.json", "w"), indent=1, ensure_ascii=False)
    print(f"→ {TVC}/clip_linear_512.json")


if __name__ == "__main__":
    main()
