#!/usr/bin/env python3
"""
机制研究 M1+M2：跨范式能力的层定位与表示相似性
==================================================
M1（层定位）：对每个模型、每一层表示，用 **APRICOT 训练**线性探针，
   在 **APDE**（未见范式）上测试 → 衡量"跨范式可迁移性"，定位能力所在层。
M2（表示相似性）：度量同一模型对 APRICOT 补丁 vs APDE 补丁的表示相似性（线性 CKA），
   检验"跨范式鲁棒 ≈ 表示跨范式一致"的假设。

模型：NP-PatchLM(VLM 微调) / VLM base / CLIP / ResNet18 / ShuffleNetV2
输出: results/tvc/mechanism_probe.json
"""
import os, sys, json, argparse
os.environ.setdefault("HF_HOME", "/home/oyp/.cache/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np

WS = "/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard"
TVC = f"{WS}/results/tvc"
EQ = f"{TVC}/unified_eval_set"
APDE = f"{TVC}/apde_imgs"
SIZE = 1024


def fit_long(im, size=SIZE):
    from PIL import Image
    w, h = im.size; s = size / max(w, h)
    return im if abs(s-1) < 1e-6 else im.resize((max(1, int(w*s)), max(1, int(h*s))), Image.BILINEAR)


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    try: return float(roc_auc_score(y, s))
    except Exception: return float("nan")


def linear_cka(X, Y):
    """线性 CKA（样本对齐，无需同维）"""
    X = X - X.mean(0, keepdims=True); Y = Y - Y.mean(0, keepdims=True)
    num = np.linalg.norm(X.T @ Y, "fro") ** 2
    den = np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro")
    return float(num / den) if den > 0 else 0.0


def probe_transfer(Xtr, ytr, Xte, yte):
    """APRICOT 训练 → APDE 测试的线性探针（返回 APDE 上的 AUC）"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=1000, C=1.0).fit(sc.transform(Xtr), ytr)
    return auc(yte, clf.predict_proba(sc.transform(Xte))[:, 1])


def get_vlm_hidden(ckpt=None):
    """提取 VLM 各层 hidden states（最后一 token 的表示）"""
    import torch
    sys.path.insert(0, f"{WS}/code/experiments_tvc")
    from lora_mllm import base_and_proc, PROMPT
    from peft import PeftModel
    model, proc = base_and_proc()
    if ckpt:
        model = PeftModel.from_pretrained(model, ckpt)
    model = model.cuda().eval()

    @torch.no_grad()
    def feats(im):
        im = fit_long(im)
        msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        t = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        enc = {k: (v.cuda() if torch.is_tensor(v) else v)
               for k, v in proc(text=[t], images=[im], return_tensors="pt").items()}
        out = model(**enc, output_hidden_states=True)
        hs = out.hidden_states          # tuple([layers][B,T,D])
        return [h[0, -1, :].float().cpu().numpy() for h in hs]
    return feats, model


def get_cnn_feats(ckpt, name):
    import torch, torch.nn as nn
    from torchvision import transforms, models
    dev = "cuda"
    if name == "resnet18":
        m = models.resnet18(weights=None); m.fc = nn.Linear(m.fc.in_features, 2)
    else:
        m = models.shufflenet_v2_x1_0(weights=None); m.fc = nn.Linear(m.fc.in_features, 2)
    m.load_state_dict(torch.load(ckpt, map_location=dev)); m = m.to(dev).eval()
    norm = transforms.Normalize([0.485,0.456,0.486], [0.229,0.224,0.225])
    feats_store = []
    def hook(mod, inp, out):
        feats_store.append(out.detach().float().mean(dim=(2,3)).cpu().numpy()[0])
    # 按层注册 hook
    layers = []
    if name == "resnet18":
        layers = [m.layer1, m.layer2, m.layer3, m.layer4, m.avgpool]
    else:
        layers = [m.conv5, m.stage2, m.stage3, m.stage4]
    hs = [l.register_forward_hook(hook) for l in layers]

    @torch.no_grad()
    def feats(im):
        feats_store.clear()
        x = norm(transforms.ToTensor()(fit_long(im))).unsqueeze(0).to(dev)
        m(x)
        return [np.asarray(f).ravel() for f in feats_store]
    return feats, m, hs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="vlm_ft,vlm_base,resnet18,shufflenetv2")
    ap.add_argument("--n", type=int, default=80, help="每类采样数（控制时间）")
    ap.add_argument("--out", default=f"{TVC}/mechanism_probe.json")
    args = ap.parse_args()
    from PIL import Image

    # 数据：APRICOT 正（源域）+ APDE 正（目标范式）+ 干净图（负）
    pos_apri = [Image.open(f"{EQ}/pos/{f}").convert("RGB") for f in sorted(os.listdir(f"{EQ}/pos"))[:args.n]]
    pos_apde = [Image.open(f"{APDE}/pos/{f}").convert("RGB") for f in sorted(os.listdir(f"{APDE}/pos"))[:args.n]]
    neg_files = (sorted(os.listdir(f"{EQ}/casia_au"))[:args.n//2] + sorted(os.listdir(f"{EQ}/coco_val"))[:args.n//2])
    neg = [Image.open(f"{EQ}/casia_au/{f}").convert("RGB") for f in sorted(os.listdir(f"{EQ}/casia_au"))[:args.n//2]]
    neg += [Image.open(f"{EQ}/coco_val/{f}").convert("RGB") for f in sorted(os.listdir(f"{EQ}/coco_val"))[:args.n//2]]
    print(f"  数据: APRICOT正 {len(pos_apri)} / APDE正 {len(pos_apde)} / 干净 {len(neg)}", flush=True)

    res = {"setup": {"n_apricot": len(pos_apri), "n_apde": len(pos_apde), "n_neg": len(neg),
                     "size": SIZE, "probe": "APRICOT(+) vs clean(-) 训练 → APDE(+) vs clean(-) 测试"}}

    for key in args.models.split(","):
        try:
            if key.startswith("vlm"):
                ckpt = f"{TVC}/lora_patch_detector" if key == "vlm_ft" else None
                feats, model = get_vlm_hidden(ckpt)
                cleanup = model
            else:
                ckpt = f"{TVC}/{key}_fair.pt"
                feats, model, hs = get_cnn_feats(ckpt, key)
                cleanup = None
            print(f"  [{key}] 提取表示…", flush=True)
            F_apri = [feats(im) for im in pos_apri]      # list[list[layer_vec]]
            F_apde = [feats(im) for im in pos_apde]
            F_neg = [feats(im) for im in neg]
            nlayers = len(F_apri[0])
            row = {"n_layers": nlayers, "per_layer": {}}
            for L in range(nlayers):
                Xa = np.stack([f[L] for f in F_apri]); Xp = np.stack([f[L] for f in F_apde])
                Xn = np.stack([f[L] for f in F_neg])
                # 源域可分性
                Xtr = np.vstack([Xa, Xn]); ytr = np.array([1]*len(Xa) + [0]*len(Xn))
                sc_src = auc(ytr, Xtr[:, 0]) if False else None
                # 用逻辑回归在源域做 5-fold 估源域可分性
                from sklearn.linear_model import LogisticRegression
                from sklearn.preprocessing import StandardScaler
                from sklearn.model_selection import cross_val_predict
                scl = StandardScaler().fit(Xtr)
                cv = cross_val_predict(LogisticRegression(max_iter=500), scl.transform(Xtr), ytr,
                                       cv=5, method="predict_proba")[:, 1]
                src_auc = auc(ytr, cv)
                # 跨范式迁移：APRICOT 训练 → APDE 测试
                Xte = np.vstack([Xp, Xn]); yte = np.array([1]*len(Xp) + [0]*len(Xn))
                tf_auc = probe_transfer(Xtr, ytr, Xte, yte)
                # 跨范式表示相似性（APRICOT 正 vs APDE 正的线性 CKA）
                cka = linear_cka(Xa[:min(len(Xa), len(Xp))], Xp[:min(len(Xa), len(Xp))])
                row["per_layer"][str(L)] = {"src_auc": round(src_auc, 4),
                                            "transfer_auc": round(tf_auc, 4),
                                            "cka_apricot_vs_apde": round(cka, 4)}
            res[key] = row
            best_t = max(row["per_layer"].items(), key=lambda kv: kv[1]["transfer_auc"])
            print(f"    [{key}] 层数={nlayers}  最佳迁移层 L{best_t[0]}: "
                  f"transfer_auc={best_t[1]['transfer_auc']:.3f}  src_auc={best_t[1]['src_auc']:.3f}  "
                  f"cka={best_t[1]['cka_apricot_vs_apde']:.3f}", flush=True)
            del feats
            import torch, gc; gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [{key}] 失败: {type(e).__name__}: {e}", flush=True)
            res[key] = {"error": f"{type(e).__name__}: {e}"}
    json.dump(res, open(args.out, "w"), indent=1, ensure_ascii=False)
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
