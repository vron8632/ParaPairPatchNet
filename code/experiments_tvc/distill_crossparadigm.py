#!/usr/bin/env python3
"""
跨范式蒸馏：把 VLM 的跨范式不变性蒸馏到轻量模型
====================================================
教师：NP-PatchLM（VLM+LoRA，可选 ckpt）
学生：ShuffleNetV2 / ResNet18（2.3M / 11M）
关键：用**多范式配对数据**（digital/physical 同索引）+ 教师软标签，
      让学生学到"同一补丁不同范式 → 相同分数"的映射（跨范式不变性）。
训练目标：L = MSE(student, teacher) + CE(student, hard_label)
输出: results/tvc/distill_<student>.json（同域+跨域双评测）
"""
import os, sys, json, argparse, math
os.environ.setdefault("HF_HOME", "/home/oyp/.cache/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np

WS = "/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard"
TVC = f"{WS}/results/tvc"
EVAL = f"{TVC}/unified_eval_set"
APDE = f"{TVC}/apde_imgs"
sys.path.insert(0, f"{WS}/code/experiments_tvc")


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k/n; d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (round(max(0.0, c-h), 4), round(min(1.0, c+h), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default=f"{TVC}/full_lora_s1")
    ap.add_argument("--student", default="shufflenetv2", choices=["shufflenetv2", "resnet18"])
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--mse-weight", type=float, default=1.0)
    ap.add_argument("--cache", default=f"{TVC}/distill_teacher_scores.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"{TVC}/distill_{args.student}.json"

    import torch, torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms, models
    from PIL import Image
    from lora_mllm import load_lora_for_eval, PROMPT

    def fit(im, size):
        w, h = im.size; s = size / max(w, h)
        return im if abs(s-1) < 1e-6 else im.resize((max(1, int(w*s)), max(1, int(h*s))), Image.BILINEAR)

    idx = json.load(open(f"{TVC}/lora_data/index.json"))
    tr = [r for r in idx if r["split"] == "train"]

    # ── 1) 教师打分（缓存）──
    if os.path.exists(args.cache):
        ts = json.load(open(args.cache))
        print(f"[distill] 复用教师分数缓存: {len(ts)} 条", flush=True)
    else:
        model, proc = load_lora_for_eval(args.teacher, 16)
        cfg = json.load(open(f"{args.teacher}/train_config.json"))
        yi, ni = cfg["yes_id"], cfg["no_id"]

        @torch.no_grad()
        def tscore(im):
            im = fit(im, args.size)
            msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
            t = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            enc = {k: (v.cuda() if torch.is_tensor(v) else v)
                   for k, v in proc(text=[t], images=[im], return_tensors="pt").items()}
            lg = model(**enc).logits[0, -1]
            return float(torch.sigmoid(lg[yi] - lg[ni]))

        ts = {}
        for i, r in enumerate(tr):
            ts[r["path"]] = tscore(Image.open(r["path"]).convert("RGB"))
            if (i+1) % 200 == 0: print(f"  teacher {i+1}/{len(tr)}", flush=True)
        json.dump(ts, open(args.cache, "w"))
        del model; torch.cuda.empty_cache()
        print(f"[distill] 教师打分完成 {len(ts)} 条", flush=True)

    # ── 2) 学生训练（软标签 MSE + 硬标签 CE）──
    tf = transforms.Compose([transforms.RandomHorizontalFlip(),
                             transforms.ColorJitter(0.2, 0.2, 0.2),
                             transforms.ToTensor(),
                             transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])

    class DS(Dataset):
        def __len__(s): return len(tr)
        def __getitem__(s, i):
            r = tr[i]
            im = fit(Image.open(r["path"]).convert("RGB"), args.size)
            return tf(im), r["label"], ts.get(r["path"], 0.5)

    dl = DataLoader(DS(), batch_size=args.bs, shuffle=True, num_workers=0)
    if args.student == "shufflenetv2":
        m = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, 2)
    else:
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, 2)
    m = m.cuda()
    from collections import Counter
    c = Counter(r["label"] for r in tr)
    w = torch.tensor([len(tr)/(2*c[0]), len(tr)/(2*c[1])]).cuda()
    ce = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    for ep in range(args.epochs):
        m.train(); tot = 0
        for x, y, t in dl:
            x, y, t = x.cuda(), y.cuda(), t.float().cuda()
            logits = m(x)
            soft = torch.softmax(logits, 1)[:, 1]
            loss = ce(logits, y) + args.mse_weight * torch.mean((soft - t) ** 2)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss)
        sched.step()
        if (ep+1) % 5 == 0: print(f"  [{args.student}] ep{ep+1}/{args.epochs} loss={tot/max(len(dl),1):.4f}", flush=True)

    # ── 3) 双基准评测 ──
    m.eval()
    tf_ev = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
    @torch.no_grad()
    def score_dir(d):
        out = []
        for f in sorted(os.listdir(d)):
            im = fit(Image.open(os.path.join(d, f)).convert("RGB"), args.size)
            x = tf_ev(im).unsqueeze(0).cuda()
            out.append(float(torch.softmax(m(x), 1)[0, 1]))
        return np.array(out)

    S3 = ["casia_au", "coco_val", "inria_noped"]
    p = score_dir(f"{EVAL}/pos")
    ns = {s: score_dir(f"{EVAL}/{s}") for s in S3 + ["columbia_au"]}
    res = {"student": args.student, "teacher": os.path.basename(args.teacher),
           "mse_weight": args.mse_weight, "params_M": round(sum(p.numel() for p in m.parameters())/1e6, 1),
           "apricot": {}, "apde": {}}
    for scope, keys in (("3src", S3), ("4src", S3 + ["columbia_au"])):
        n_all = np.concatenate([ns[k] for k in keys]); rr = {}
        for a in (0.05, 0.10):
            thr = float(np.percentile(n_all, 100*(1-a))); k_ = int((p >= thr).sum())
            rr[f"@{int(a*100)}pct"] = {"tpr": round(k_/len(p), 4), "wilson": wilson(k_, len(p))}
        res["apricot"][scope] = rr
    pa = score_dir(f"{APDE}/pos"); na = score_dir(f"{APDE}/neg")
    fams = {}
    for f, sc in zip(sorted(os.listdir(f"{APDE}/pos")), pa):
        fams.setdefault(f.split("_")[0], []).append(sc)
    thr = float(np.percentile(na, 95))
    res["apde"]["@5pct"] = {"tpr": round(float((pa >= thr).mean()), 4)}
    res["apde"]["by_family"] = {k: round(float((np.array(v) >= thr).mean()), 4) for k, v in fams.items()}
    print(f"\n  ✅ {args.student}（{res['params_M']}M）: "
          f"APRICOT 3src@5%={res['apricot']['3src']['@5pct']['tpr']} | "
          f"APDE@5%={res['apde']['@5pct']['tpr']} | 分族={res['apde']['by_family']}", flush=True)
    json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)
    print(f"→ {out}")


if __name__ == "__main__":
    main()
