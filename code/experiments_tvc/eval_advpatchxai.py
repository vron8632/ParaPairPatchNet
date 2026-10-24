#!/usr/bin/env python3
"""
AdvPatchXAI 在统一低误报协议下的评测（第一步：验证现有跨范式 SOTA 在低 FPR 区的表现）
=========================================================================================
加载 AdvPatchXAI 训练好的 checkpoint，对：
  A) unified_eval_set（138 正 + 1000 负，四源/三源）—— 真实物理补丁
  B) apde_imgs（396 正 + 300 负）—— 对抗优化补丁（未见范式）
导出分数并按统一规则（负样本分位数）计算 TPR@{1,5,10,20}% + Wilson CI。
用法: $PY eval_advpatchxai.py --ckpt <state_dict_dir> --net resnet50
"""
import os, sys, json, argparse, math
import numpy as np

WS = "/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard"
TVC = f"{WS}/results/tvc"
AX = f"{WS}/code/third_party/AdvPatchXAI"
sys.path.insert(0, AX)


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k/n; d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (round(max(0.0, c-h), 4), round(min(1.0, c+h), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="AdvPatchXAI state_dict_dir_net 目录")
    ap.add_argument("--net", default="resnet50")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--out", default=f"{TVC}/advpatchxai_unified.json")
    ap.add_argument("--n-coco", type=int, default=20000, help="追加 COCO-train 评测负样本数（只评测不入训练）")
    ap.add_argument("--emit-npz", action="store_true", help="导出 21k 负样本协议下的分数到 sota_scores_advpatchxai.npz")
    args = ap.parse_args()

    import torch, torch.nn as nn, torch.nn.functional as F
    from PIL import Image
    from torchvision import transforms
    from advpatchxai.model import AdvPatchXAI, get_network
    # get_network 返回 (feature_net, add_on, pool, classification, num_prototypes)
    class _A: pass
    nargs = _A()
    nargs.net = args.net
    nargs.disable_pretrained = True      # 权重随后 load_state_dict 覆盖
    nargs.num_features = 0
    nargs.bias = True
    nargs.image_size = args.image_size
    feat, add_on, pool, cls, n_proto = get_network(2, nargs)
    net = AdvPatchXAI(num_classes=2, num_prototypes=n_proto, feature_net=feat,
                      args=nargs, add_on_layers=add_on, pool_layer=pool,
                      classification_layer=cls)
    # 加载 ckpt：可能是 {model_state_dict, ...} 包装，也可能是裸 state_dict（含/不含 module. 前缀）
    _p = args.ckpt
    if os.path.isdir(_p):
        for cand in ("net_trained_last", "net_trained", "net_pretrained"):
            if os.path.exists(os.path.join(_p, cand)):
                _p = os.path.join(_p, cand); break
    ck = torch.load(_p, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) and "model_state_dict" in ck else ck
    if any(k.startswith("module.") for k in list(sd.keys())[:5]):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    print(f"[advpatchxai] 加载 {os.path.basename(_p)}: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if len(missing) > 20:
        print(f"  ⚠️ 大量 key 缺失，示例: {missing[:3]}", flush=True)
    dev = "cuda"
    net = net.to(dev).eval()
    print(f"[advpatchxai] 加载完成: {args.ckpt}", flush=True)

    # ⚠️ 严格复用官方预处理：Resize → Grayscale(3) → ToTensor → Normalize(_MEAN,_STD)
    # （AdvPatchXAI 论文 Sec.3.2 使用灰度输入；此前用 RGB 导致分数语义错误）
    try:
        from util.data import _transform_no_augment
        tf = _transform_no_augment(args.image_size, grayscale=True)
        print("[advpatchxai] 使用官方灰度预处理 ✅", flush=True)
    except Exception as e:
        print(f"[advpatchxai] 官方 transform 导入失败({e})，回退手写", flush=True)
        tf = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                                 transforms.Grayscale(3), transforms.ToTensor(),
                                 transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])

    @torch.no_grad()
    def score_dir(d):
        out = []
        fs = sorted(os.listdir(d))
        for i, f in enumerate(fs):
            im = Image.open(os.path.join(d, f)).convert("RGB")
            x = tf(im).unsqueeze(0).to(dev)
            _, _, _, logits = net(x, inference=True)
            p = torch.softmax(logits, 1)[0, 1].item()
            out.append(p)
            if (i+1) % 200 == 0: print(f"    {os.path.basename(d)} {i+1}/{len(fs)}", flush=True)
        return np.array(out)

    @torch.no_grad()
    def score_paths(paths):
        out = []
        for i, p_ in enumerate(paths):
            try:
                im = Image.open(p_).convert("RGB")
                x = tf(im).unsqueeze(0).to(dev)
                _, _, _, logits = net(x, inference=True)
                out.append(torch.softmax(logits, 1)[0, 1].item())
            except Exception:
                out.append(0.0)
            if (i+1) % 2000 == 0: print(f"    coco {i+1}/{len(paths)}", flush=True)
        return np.array(out)

    EQ = f"{TVC}/unified_eval_set"
    S3 = ["casia_au", "coco_val", "inria_noped"]
    res = {"model": f"AdvPatchXAI ({args.net}, input {args.image_size})", "protocol": "unified (low-FPR)"}

    # A) 真实物理补丁（APRICOT）
    p = score_dir(f"{EQ}/pos")
    ns = {s: score_dir(f"{EQ}/{s}") for s in S3 + ["columbia_au"]}
    row = {"n_pos": len(p), "pos_mean": round(float(p.mean()), 4),
           "neg_mean": {k: round(float(v.mean()), 4) for k, v in ns.items()}, "tpr": {}}
    for scope, keys in (("3src", S3), ("4src", S3 + ["columbia_au"])):
        n_all = np.concatenate([ns[k] for k in keys]); rr = {}
        for a in (0.01, 0.05, 0.10, 0.20):
            thr = float(np.percentile(n_all, 100*(1-a))); k_ = int((p >= thr).sum())
            rr[f"@{int(a*100)}pct"] = {"tpr": round(k_/len(p), 4), "wilson": wilson(k_, len(p)),
                                        "thr": round(thr, 4)}
        row["tpr"][scope] = rr
    res["apricot"] = row
    print(f"  [APRICOT] 3src@5%={row['tpr']['3src']['@5pct']['tpr']}  "
          f"4src@5%={row['tpr']['4src']['@5pct']['tpr']}", flush=True)

    # B) 对抗优化补丁（APDE，未见范式）
    ap_de = f"{TVC}/apde_imgs"
    pp = score_dir(f"{ap_de}/pos"); nn_ = score_dir(f"{ap_de}/neg")
    row2 = {"n_pos": len(pp), "pos_mean": round(float(pp.mean()), 4),
            "neg_mean": round(float(nn_.mean()), 4), "tpr": {}}
    for a in (0.01, 0.05, 0.10, 0.20):
        thr = float(np.percentile(nn_, 100*(1-a))); k_ = int((pp >= thr).sum())
        row2["tpr"][f"@{int(a*100)}pct"] = {"tpr": round(k_/len(pp), 4),
                                            "wilson": wilson(k_, len(pp)), "thr": round(thr, 4)}
    res["apde"] = row2
    print(f"  [APDE] @5%={row2['tpr']['@5pct']['tpr']}  @10%={row2['tpr']['@10pct']['tpr']}", flush=True)

    # C) COCO-train 追加负样本（低误报协议：负样本从 1,000 扩到 21,000）
    if args.n_coco and args.n_coco > 0:
        coco_paths = [r["path"] for r in json.load(open("/home/oyp/fastdata/apde_negs/index.json"))[:args.n_coco]]
        print(f"  [COCO-train] 追加 {len(coco_paths)} 张负样本…", flush=True)
        cs = score_paths(coco_paths)
        neg_all = np.concatenate([np.concatenate([ns[k] for k in S3 + ["columbia_au"]]), cs])
        r2 = {}
        for q in (0.05, 0.01, 0.001):
            thr = float(np.percentile(neg_all, 100*(1-q)))
            r2[f"@{q*100:g}pct"] = {"tpr": round(float((p >= thr).mean()), 5),
                                    "thr": round(thr, 4),
                                    "wilson": wilson(int((p >= thr).sum()), len(p))}
        res["apricot_21k"] = {"n_pos": len(p), "n_neg": len(neg_all), "tpr": r2,
                              "neg_mean_by_src": {**{k: round(float(np.mean(ns[k])), 4) for k in ns},
                                                  "coco_train": round(float(cs.mean()), 4)}}
        print(f"  [APRICOT 21k 负样本] @5%={r2['@5pct']['tpr']}  @1%={r2['@1pct']['tpr']}  @0.1%={r2['@0.1pct']['tpr']}", flush=True)
        if args.emit_npz:
            order4 = S3 + ["columbia_au"]
            scores = np.concatenate([p] + [ns[k] for k in order4] + [cs])
            src = (["apricot_pos"] * len(p) + sum([[k] * len(ns[k]) for k in order4], [])
                   + ["coco_train"] * len(cs))
            label = np.concatenate([np.ones(len(p)), np.zeros(len(neg_all))]).astype(np.int64)
            np.savez_compressed(f"{TVC}/sota_scores_advpatchxai.npz", scores=scores.astype(np.float32),
                                src=np.array(src), label=label)
            print(f"  → {TVC}/sota_scores_advpatchxai.npz", flush=True)

    json.dump(res, open(args.out, "w"), indent=1, ensure_ascii=False)
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
