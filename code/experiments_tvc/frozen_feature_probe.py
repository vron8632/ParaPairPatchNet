#!/usr/bin/env python3
"""
P2：冻结特征 + 线性分类器 —— 检验"骨干使用方式"是否比"骨干容量"更决定跨范式泛化
============================================================================
对比同一批骨干的三种使用方式：
  (A) 端到端微调（已有结果，直接引用）
  (B) 冻结全局特征 + LogisticRegression（本脚本）
  (C) 冻结全局特征 + LinearSVC（本脚本）
评测：APRICOT 3src/4src @5% + APDE @5%（与统一协议一致）
特征缓存到 results/tvc/feat_cache/<backbone>_<split>.npz，可复用。
"""
import os, sys, json, argparse, math
os.environ.setdefault("HF_HOME", "/home/oyp/.cache/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np

WS = "/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard"
TVC = f"{WS}/results/tvc"
EVAL = f"{TVC}/unified_eval_set"
APDE = f"{TVC}/apde_imgs"
CACHE = f"{TVC}/feat_cache"


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k/n; d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (round(max(0.0, c-h), 4), round(min(1.0, c+h), 4))


def fit_long(im, size):
    from PIL import Image
    w, h = im.size; s = size / max(w, h)
    return im if abs(s-1) < 1e-6 else im.resize((max(1, int(w*s)), max(1, int(h*s))), Image.BILINEAR)


def build(name):
    import torch, torch.nn as nn
    from torchvision import models
    if name == "shufflenetv2":
        m = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
        feat_dim = m.fc.in_features; m.fc = nn.Identity()
    elif name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        feat_dim = m.fc.in_features; m.fc = nn.Identity()
    elif name == "mobilenetv3":
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        feat_dim = m.classifier[-1].in_features; m.classifier[-1] = nn.Identity()
    else:
        raise SystemExit(f"未知骨干 {name}")
    m.eval()
    for p in m.parameters(): p.requires_grad = False
    return m.cuda(), feat_dim


def get_clip():
    import torch
    from transformers import CLIPModel
    m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda().eval()
    for p in m.parameters(): p.requires_grad = False
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", default="shufflenetv2,resnet18,mobilenetv3,clip")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--out", default=f"{TVC}/frozen_feature_probe.json")
    args = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)

    import torch
    from torchvision import transforms
    from PIL import Image
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC

    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])

    # 收集所有需要打分的图像（按 split 分组）
    idx = json.load(open("/home/oyp/fastdata/lora_data/index.json"))
    tr = [r for r in idx if r["split"] == "train"]
    S3 = ["casia_au", "coco_val", "inria_noped"]
    groups = {"train": [(r["path"], r["label"]) for r in tr]}
    groups["apricot_pos"] = [(f"{EVAL}/pos/{f}", 1) for f in sorted(os.listdir(f"{EVAL}/pos"))]
    for s in S3 + ["columbia_au"]:
        groups[s] = [(f"{EVAL}/{s}/{f}", 0) for f in sorted(os.listdir(f"{EVAL}/{s}"))]
    groups["apde_pos"] = [(f"{APDE}/pos/{f}", 1) for f in sorted(os.listdir(f"{APDE}/pos"))]
    groups["apde_neg"] = [(f"{APDE}/neg/{f}", 0) for f in sorted(os.listdir(f"{APDE}/neg"))]

    def extract(bb):
        cache_f = f"{CACHE}/{bb}_{args.size}.npz"
        if os.path.exists(cache_f):
            z = np.load(cache_f, allow_pickle=True)
            print(f"  [{bb}] 复用缓存", flush=True)
            return {k: z[k] for k in z.files}
        if bb == "clip":
            model = get_clip()
            def embed(pil):
                import torchvision.transforms as T
                x = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                               T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])(pil).unsqueeze(0).cuda()
                with torch.no_grad():
                    o = model.vision_model(pixel_values=x)
                    return model.visual_projection(o.pooler_output)[0].float().cpu().numpy()
        else:
            model, _ = build(bb)
            def embed(pil):
                x = tf(fit_long(pil, args.size)).unsqueeze(0).cuda()
                with torch.no_grad(): return model(x)[0].float().cpu().numpy()
        out = {}
        for g, items in groups.items():
            feats = []
            for i, (p, _) in enumerate(items):
                feats.append(embed(Image.open(p).convert("RGB")))
                if (i+1) % 500 == 0: print(f"    [{bb}] {g} {i+1}/{len(items)}", flush=True)
            out[g] = np.stack(feats); out[g+"_y"] = np.array([y for _, y in items])
            print(f"  [{bb}] {g}: {out[g].shape}", flush=True)
        np.savez_compressed(cache_f, **out)
        del model; torch.cuda.empty_cache()
        return out

    results = {}
    for bb in args.backbones.split(","):
        bb = bb.strip()
        print(f"\n{'='*80}\n骨干: {bb}\n{'='*80}", flush=True)
        try:
            F = extract(bb)
        except Exception as e:
            print(f"  ❌ 骨干 {bb} 失败（跳过）: {type(e).__name__}: {str(e)[:160]}", flush=True)
            import torch as _t; _t.cuda.empty_cache()
            continue
        Xtr, ytr = F["train"], F["train_y"]
        # 标准化（用训练集统计）
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        z = lambda A: (A - mu) / sd
        Xtr_z = z(Xtr)

        for clf_name, clf_fn in [
            ("LR_C1", lambda: LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")),
            ("LR_C0.01", lambda: LogisticRegression(max_iter=3000, C=0.01, class_weight="balanced")),
            ("SVC_C1", lambda: LinearSVC(C=1.0, class_weight="balanced", max_iter=5000)),
        ]:
            clf = clf_fn(); clf.fit(Xtr_z, ytr)
            score = lambda A: clf.decision_function(z(A)) if hasattr(clf, "decision_function") else clf.predict_proba(z(A))[:, 1]
            p = score(F["apricot_pos"])
            ns = {s: score(F[s]) for s in S3 + ["columbia_au"]}
            pa, na = score(F["apde_pos"]), score(F["apde_neg"])
            r = {"backbone": bb, "clf": clf_name, "apricot": {}, "apde": {}}
            for scope, keys in (("3src", S3), ("4src", S3 + ["columbia_au"])):
                n_all = np.concatenate([ns[k] for k in keys]); rr = {}
                for a in (0.05, 0.10):
                    thr = float(np.percentile(n_all, 100*(1-a)))
                    k_ = int((p >= thr).sum())
                    rr[f"@{int(a*100)}pct"] = {"tpr": round(k_/len(p), 4), "wilson": wilson(k_, len(p))}
                r["apricot"][scope] = rr
            thr = float(np.percentile(na, 95))
            fams = {}
            for f, sc in zip(sorted(os.listdir(f"{APDE}/pos")), pa):
                fams.setdefault(f.split("_")[0], []).append(sc)
            r["apde"]["@5pct"] = round(float((pa >= thr).mean()), 4)
            r["apde"]["by_family"] = {k: round(float((np.array(v) >= thr).mean()), 4) for k, v in fams.items()}
            r["apde"]["worst_family"] = min(r["apde"]["by_family"].items(), key=lambda x: x[1])
            print(f"  {clf_name:10s} APRICOT 3src@5%={r['apricot']['3src']['@5pct']['tpr']:.3f} "
                  f"| APDE@5%={r['apde']['@5pct']:.3f} | 最弱族={r['apde']['worst_family']}", flush=True)
            results[f"{bb}_{clf_name}"] = r
            json.dump(results, open(args.out, "w"), indent=1, ensure_ascii=False)

    json.dump(results, open(args.out, "w"), indent=1, ensure_ascii=False)
    print(f"\n→ {args.out}")
    print("\n=== 汇总（对照：端到端微调 ShuffleNetV2 0.957/0.093；VLM 0.717/0.712–1.000）===")
    for k, v in sorted(results.items(), key=lambda x: -x[1]["apde"]["@5pct"]):
        print(f"  {k:28s} APRICOT@5%={v['apricot']['3src']['@5pct']['tpr']:.3f}  APDE@5%={v['apde']['@5pct']:.3f}")


if __name__ == "__main__":
    main()
