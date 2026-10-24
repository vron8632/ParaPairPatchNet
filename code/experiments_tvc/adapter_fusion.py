#!/usr/bin/env python3
"""
MLLM 创新 #2 — 可学习 Adapter 融合（MLLM 分数 × 视觉特征分数）
===========================================================
在**同一批样本**上分别得到两个分支的连续分数，再用轻量 Adapter 融合:
    视觉分支 s_v : clip+spn+dct 分类器（train on clean+digital 的 train split）
    MLLM 分支 s_m : LoRA-Qwen2-VL-2B（YES/NO logit 差）
    融合    h    : [s_m ; s_v ; s_m*s_v ; s_m - s_v] -> Linear -> GELU -> Linear -> sigmoid
                   （约 4*8+8+8+1 = 49 个参数，极轻量、可解释）

样本对齐: lora_data 由 patch_dataset_v2 的 tensor[i] 生成，文件名含 {dom}_{sp}_{i}，
          据此对齐到 features/{dom}_{sp}_features.pt 的第 i 个样本。

用法:
  $PY code/experiments_tvc/adapter_fusion.py --stage scores   # 计算两分支分数
  $PY code/experiments_tvc/adapter_fusion.py --stage fuse     # 拟合 Adapter 并评估
输出: results/tvc/adapter_fusion_scores.json / adapter_fusion.json
"""
import argparse, json, os, re, sys
import numpy as np

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)
WS = os.path.dirname(CODE_DIR)
OUT = os.path.join(WS, "results", "tvc")
SRC = "/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard"
FD = os.path.join(SRC, "results", "features")
TRAIN_DOMAINS = ("clean", "digital_patch")
EVAL_DOMAINS = ("clean", "physical_patch")   # 评估集: 干净 vs 物理补丁
SPLIT = "val"


def visual_scores(model, cic, fuse, device):
    """视觉分支: 用 train domain 训练 MLP，在 EVAL_DOMAINS 的 val 上打分。"""
    import torch
    import torch.nn as nn

    def load(dom, sp):
        return torch.load(os.path.join(FD, f"{dom}_{sp}_features.pt"),
                          map_location="cpu", weights_only=False)

    def feats(d):
        return torch.cat([fuse([d["spn_feats"][i]], [d["dct_feats"][i]],
                               [d["clip_feats"][i]], cic, device)
                          for i in range(len(d["labels"]))], 0)

    Xtr, ytr = [], []
    for dom in TRAIN_DOMAINS:
        d = load(dom, "train")
        Xtr.append(feats(d)); ytr.append(np.array(d["labels"]))
    Xtr = torch.cat(Xtr); ytr = torch.tensor(np.concatenate(ytr), dtype=torch.float32)

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd
    m = nn.Sequential(nn.Linear(Xtr.shape[1], 128), nn.ReLU(), nn.Linear(128, 1)).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    lf = nn.BCEWithLogitsLoss()
    Xtr, ytr = Xtr.to(device), ytr.to(device)
    for _ in range(400):
        m.train(); opt.zero_grad()
        lf(m(Xtr).squeeze(1), ytr).backward(); opt.step()
    m.eval()

    rows = []
    with torch.no_grad():
        for dom in EVAL_DOMAINS:
            d = load(dom, SPLIT)
            X = ((feats(d) - mu) / sd).to(device)
            p = torch.sigmoid(m(X).squeeze(1)).cpu().numpy()
            for i, s in enumerate(p):
                rows.append({"domain": dom, "idx": i, "score": float(s),
                             "label": 0 if dom == "clean" else 1})
    print(f"[visual] {len(rows)} 样本")
    return rows


def mllm_scores(args):
    """LoRA-MLLM 分数，按 {dom}_{sp}_{i} 对齐。"""
    import torch, numpy as np
    from PIL import Image
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from lora_mllm import QWEN_SNAP, PROMPT, load_lora_for_eval

    ckpt = os.path.join(OUT, "lora_patch_detector")
    model, proc = load_lora_for_eval(ckpt, 16)
    cfg = json.load(open(os.path.join(OUT, "lora_patch_detector", "train_config.json")))
    yes_id, no_id = cfg["yes_id"], cfg["no_id"]

    pat = re.compile(r"(clean|physical_patch|digital_patch)_(train|val|test)_(\d+)\.jpg$")
    rows, idx_map = [], []
    index = json.load(open(os.path.join(OUT, "lora_data", "index.json")))
    for r in index:
        m = pat.search(r["path"])
        if not m:
            continue
        dom, sp, i = m.group(1), m.group(2), int(m.group(3))
        if sp == SPLIT and dom in EVAL_DOMAINS:
            rows.append((dom, i, r["path"]))

    print(f"[mllm] {len(rows)} 样本待推理")
    out_rows = []
    with torch.no_grad():
        for k, (dom, i, path) in enumerate(rows):
            img = Image.open(path).convert("RGB")
            msgs = [{"role": "user",
                     "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
            t = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            enc = {kk: (v.cuda() if torch.is_tensor(v) else v)
                   for kk, v in proc(text=[t], images=[img], return_tensors="pt").items()}
            lg = model(**enc).logits[0, -1]
            s = float(torch.sigmoid(lg[yes_id] - lg[no_id]))
            out_rows.append({"domain": dom, "idx": i, "score": s,
                             "label": 0 if dom == "clean" else 1})
            if (k + 1) % 50 == 0:
                print(f"    {k+1}/{len(rows)}", flush=True)
    return out_rows


def stage_scores(args):
    from models.consistency import CrossInstanceConsistencyV2
    from models.fusion_classifier import FusionClassifierV2  # noqa
    from scripts.train import compute_fused_features
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cic = CrossInstanceConsistencyV2()
    v = visual_scores(None, cic, compute_fused_features, device)
    m = mllm_scores(args)
    data = {"visual": v, "mllm": m, "split": SPLIT, "eval_domains": list(EVAL_DOMAINS)}
    p = os.path.join(OUT, "adapter_fusion_scores.json")
    json.dump(data, open(p, "w"), indent=2)
    print(f"[scores] wrote {p}  (visual={len(v)}, mllm={len(m)})")


def auc_of(p, y):
    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return None
    order = np.argsort(np.concatenate([p[pos], p[neg]]))
    r = np.empty(len(order)); r[order] = np.arange(1, len(order) + 1)
    return float((r[:pos.sum()].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def report(p, y, tag):
    pos, neg = y == 1, y == 0
    th = np.unique(np.percentile(p[neg], np.linspace(0, 100, 201)))
    sweep = [{"tau": round(float(t), 4),
              "tpr": round(float(((p >= t) & pos).sum() / pos.sum()), 4),
              "fpr": round(float(((p >= t) & neg).sum() / neg.sum()), 4)} for t in th]
    rep = {"tag": tag, "n": int(len(y)), "n_pos": int(pos.sum()), "n_neg": int(neg.sum()),
           "auc": round(auc_of(p, y), 4), "sweep": sweep}
    for b in (0.01, 0.05, 0.10, 0.20):
        cand = [s for s in sweep if s["fpr"] <= b]
        rep[f"best_tpr_at_fpr<={int(b*100)}pct"] = max(cand, key=lambda s: s["tpr"]) if cand else None
    print(f"[{tag}] AUC={rep['auc']} n={rep['n']}(pos={rep['n_pos']},neg={rep['n_neg']})")
    for k in ("best_tpr_at_fpr<=1pct", "best_tpr_at_fpr<=5pct",
              "best_tpr_at_fpr<=10pct", "best_tpr_at_fpr<=20pct"):
        print(f"    {k}: {rep[k]}")
    return rep


def stage_fuse(args):
    import torch, torch.nn as nn
    data = json.load(open(os.path.join(OUT, "adapter_fusion_scores.json")))
    # 按 (domain, idx) 对齐
    m = {(r["domain"], r["idx"]): r for r in data["mllm"]}
    v = {(r["domain"], r["idx"]): r for r in data["visual"]}
    keys = sorted(set(m) & set(v))
    y = np.array([m[k]["label"] for k in keys])
    s_m = np.array([m[k]["score"] for k in keys], dtype=np.float32)
    s_v = np.array([v[k]["score"] for k in keys], dtype=np.float32)
    print(f"[fuse] 对齐 {len(keys)} 样本 (pos={(y==1).sum()}, neg={(y==0).sum()})")

    results = {"mllm_only": report(s_m, y, "mllm_only"),
               "visual_only": report(s_v, y, "visual_only")}

    # Adapter: [s_m, s_v, s_m*s_v, s_m-s_v] -> GELU MLP
    X = np.stack([s_m, s_v, s_m * s_v, s_m - s_v], 1).astype(np.float32)
    idx = np.random.RandomState(0).permutation(len(y))
    nva = len(idx) // 2
    va, te = idx[:nva], idx[nva:]
    mu, sd = X[va].mean(0), X[va].std(0) + 1e-6
    Xn = (X - mu) / sd
    torch.manual_seed(0)
    adapter = nn.Sequential(nn.Linear(4, args.hidden), nn.GELU(), nn.Linear(args.hidden, 1))
    opt = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=1e-3)
    lf = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(Xn[va]).cuda(); yt = torch.tensor(y[va], dtype=torch.float32).cuda()
    adapter = adapter.cuda()
    for _ in range(args.epochs):
        adapter.train(); opt.zero_grad()
        lf(adapter(Xt).squeeze(1), yt).backward(); opt.step()
    adapter.eval()
    with torch.no_grad():
        p = torch.sigmoid(adapter(torch.tensor(Xn[te]).cuda()).squeeze(1)).cpu().numpy()
    results["adapter_fusion"] = report(p, y[te], f"adapter_fusion(hidden={args.hidden})")
    n_params = sum(p.numel() for p in adapter.parameters())
    results["adapter_params"] = int(n_params)
    print(f"[fuse] adapter params = {n_params}")

    out = os.path.join(OUT, "adapter_fusion.json")
    json.dump(results, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"[fuse] wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["scores", "fuse", "all"], default="all")
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--epochs", type=int, default=500)
    args = ap.parse_args()
    if args.stage in ("scores", "all"):
        stage_scores(args)
    if args.stage in ("fuse", "all"):
        stage_fuse(args)


if __name__ == "__main__":
    main()
