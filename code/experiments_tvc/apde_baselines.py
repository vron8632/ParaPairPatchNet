#!/usr/bin/env python3
"""
APDE 第二基准的多模型对比 —— 回答"APDE 上是否也对比了其它模型"
================================================================
在 APDE 测试子集（396 正 + 300 负，apde_imgs 已构建）上评测：
  1. NutNet (CCS'24) 官方 AE 判据      —— 补丁专用检测器
  2. ResNet18 / ShuffleNetV2 公平版     —— 已训轻量 CNN
  3. YOLO26n-cls                       —— 已训轻量检测器分类头
  4. CLIP ViT-B/32 冻结+线性头          —— 最强轻量
NP-PatchLM 已单独评测（apde_eval.json），此处汇总全部。
输出: results/tvc/apde_baselines.json
"""
import os, sys, json
import numpy as np

WS = "/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard"
TVC = f"{WS}/results/tvc"
IMG = f"{TVC}/apde_imgs"
S3_DIRS = None  # APDE neg 是单一目录


def tpr_at(pos, neg, a):
    thr = float(np.percentile(neg, 100 * (1 - a)))
    return round(float((pos >= thr).mean()), 4)


def main():
    res = {"benchmark": "APDE test sub (396 pos + 300 neg)", "models": {}}

    # ── 1. NutNet（AE 重建判据，小模型，GPU 占用极小）──
    import torch, torch.nn as nn
    from PIL import Image
    sys.path.insert(0, f"{WS}/code/experiments_tvc")
    from nutnet_baseline import build_ae32
    dev = "cuda"
    ae = build_ae32().to(dev)
    ae.load_state_dict(torch.load(f"{WS}/code/third_party/NutNet/ae_weights/n_13.pth", map_location=dev))
    ae.eval()
    bl = 416 // 32
    def nutnet_score(d):
        out = []
        for f in sorted(os.listdir(d)):
            im = Image.open(os.path.join(d, f)).convert("RGB").resize((416, 416), Image.BILINEAR)
            x = torch.from_numpy(np.asarray(im).copy()).float().permute(2, 0, 1).unsqueeze(0) / 255
            x = (x * 2 - 1).to(dev)
            with torch.no_grad():
                blk = x.unfold(2, bl, bl).unfold(3, bl, bl)
                blk = blk.permute(0, 2, 3, 4, 5, 1).contiguous().view(-1, bl, bl, 3).permute(0, 3, 1, 2)
                out.append(float(torch.mean((ae(blk) - blk) ** 2, dim=(1, 2, 3)).max()))
        return out
    print("[1/4] NutNet ...", flush=True)
    res["models"]["NutNet"] = {"pos": nutnet_score(f"{IMG}/pos"), "neg": nutnet_score(f"{IMG}/neg")}
    del ae; torch.cuda.empty_cache()

    # ── 2. CNN 公平版 ──
    from torchvision import transforms, models
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    def load_cnn(ckpt, name):
        if name == "resnet18":
            m = models.resnet18(weights=None); m.fc = nn.Linear(m.fc.in_features, 2)
        else:
            m = models.shufflenet_v2_x1_0(weights=None); m.fc = nn.Linear(m.fc.in_features, 2)
        m.load_state_dict(torch.load(ckpt, map_location=dev)); return m.to(dev).eval()
    def cnn_score(m, d, imgsz=448):
        out = []
        for f in sorted(os.listdir(d)):
            im = Image.open(os.path.join(d, f)).convert("RGB")
            w, h = im.size; s = imgsz / max(w, h)
            im2 = im.resize((max(1, int(w*s)), max(1, int(h*s))), Image.BILINEAR)
            cv = Image.new("RGB", (imgsz, imgsz), (127, 127, 127))
            cv.paste(im2, ((imgsz - im2.width)//2, (imgsz - im2.height)//2))
            x = norm(transforms.ToTensor()(cv)).unsqueeze(0).to(dev)
            with torch.no_grad():
                out.append(float(torch.softmax(m(x), 1)[0, 1]))
        return out
    for name in ("resnet18", "shufflenetv2"):
        print(f"[2/4] {name} fair ...", flush=True)
        m = load_cnn(f"{TVC}/{name}_fair.pt", name)
        res["models"][f"{name} (fair)"] = {"pos": cnn_score(m, f"{IMG}/pos"),
                                            "neg": cnn_score(m, f"{IMG}/neg")}
        del m; torch.cuda.empty_cache()

    # ── 3. YOLO26 ──
    from ultralytics import YOLO
    print("[3/4] YOLO26 ...", flush=True)
    ym = YOLO(f"{TVC}/yolo26_cls/weights/best.pt")
    def yolo_score(d):
        out = []
        for f in sorted(os.listdir(d)):
            im = Image.open(os.path.join(d, f)).convert("RGB")
            im.thumbnail((512, 512))
            with torch.no_grad():
                r = ym.predict(np.asarray(im).copy(), imgsz=512, verbose=False)[0]
            names = r.names
            pi = next(i for i, nm in names.items() if "patch" in nm.lower())
            out.append(float(r.probs.data.tolist()[pi]))
        return out
    res["models"]["YOLO26n-cls"] = {"pos": yolo_score(f"{IMG}/pos"), "neg": yolo_score(f"{IMG}/neg")}

    # ── 4. CLIP 线性头 ──
    print("[4/4] CLIP linear ...", flush=True)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from transformers import CLIPModel, CLIPImageProcessor
    from sklearn.linear_model import LogisticRegression
    from train_light_baselines import _clip_feat, clip_linear
    # 复用训练好的线性头：重训一次（快，特征缓存不在则重提）
    clip_m, proc, clf = clip_linear()
    def clip_score(d):
        out = []
        files = sorted(os.listdir(d))
        for i in range(0, len(files), 64):
            ims = [Image.open(os.path.join(d, f)).convert("RGB") for f in files[i:i+64]]
            with torch.no_grad():
                out.extend(clf.predict_proba(_clip_feat(clip_m, proc, ims).cpu().numpy())[:, 1].tolist())
        return out
    res["models"]["CLIP-linear"] = {"pos": clip_score(f"{IMG}/pos"), "neg": clip_score(f"{IMG}/neg")}

    # ── 汇总（并入 NP-PatchLM）──
    ae = json.load(open(f"{TVC}/apde_eval.json"))
    res["models"]["NP-PatchLM (ours)"] = {"pos": ae["pos_scores"], "neg": ae["neg_scores"]}
    table = {}
    for name, dd in res["models"].items():
        p, n = np.array(dd["pos"]), np.array(dd["neg"])
        row = {"pos_mean": round(float(p.mean()), 4), "neg_mean": round(float(n.mean()), 4)}
        for a in (0.01, 0.05, 0.10, 0.20):
            row[f"TPR@{int(a*100)}pct"] = tpr_at(p, n, a)
        table[name] = row
    print("\n=== APDE 第二基准汇总 ===")
    print(f"{'方法':24s} {'pos_mean':>9} {'neg_mean':>9} {'@1%':>7} {'@5%':>7} {'@10%':>7} {'@20%':>7}")
    for k, v in table.items():
        print(f"{k:24s} {v['pos_mean']:>9.4f} {v['neg_mean']:>9.4f} {v['TPR@1pct']:>7.3f} "
              f"{v['TPR@5pct']:>7.3f} {v['TPR@10pct']:>7.3f} {v['TPR@20pct']:>7.3f}")
    res["table"] = table
    json.dump(res, open(f"{TVC}/apde_baselines.json", "w"), indent=1, ensure_ascii=False)
    print(f"→ {TVC}/apde_baselines.json")


if __name__ == "__main__":
    main()
