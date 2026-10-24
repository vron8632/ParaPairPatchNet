#!/usr/bin/env python3
"""
APDE 第二测试基准 —— 跨基准泛化评测（零样本，不重训）
================================================================
APDE (ICCV 2025, Gandolfczjh/APDE): 94 种补丁、94,000 图，416x416 PNG，
由与 APRICOT 完全不同的 13 种攻击方法生成。若 NP-PatchLM 在 APDE 上
保持高 TPR@FPR，即为真实的跨基准泛化证据。

数据: /tmp/apde_test/*.parquet (正样本, 4 种攻击) + /tmp/apde_probe/negative.parquet (1000 负)
输出: results/tvc/apde_eval.json
"""
import os, json, io
import numpy as np

WS = "/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard"
TVC = f"{WS}/results/tvc"
APDE_POS = "/tmp/apde_test"
APDE_NEG = "/tmp/apde_probe/negative.parquet"
OUT_IMG = f"{TVC}/apde_imgs"


def build(n_pos_total=400, n_neg=300):
    import pandas as pd
    from PIL import Image
    os.makedirs(f"{OUT_IMG}/pos", exist_ok=True)
    os.makedirs(f"{OUT_IMG}/neg", exist_ok=True)

    def dump_rows(df, dst_dir, tag, limit):
        n = 0
        for _, row in df.iterrows():
            if n >= limit: break
            fp = f"{dst_dir}/{tag}_{n:04d}.png"
            if not os.path.exists(fp):
                Image.open(io.BytesIO(row["image"]["bytes"])).save(fp)
            n += 1
        return n

    # 正：4 个已下载攻击类型，均匀分摊
    pos_files = sorted(f for f in os.listdir(APDE_POS)
                       if f.endswith(".parquet") and os.path.getsize(f"{APDE_POS}/{f}") > 1_000_000)
    per = n_pos_total // max(len(pos_files), 1)
    total_pos = 0
    for f in pos_files:
        df = pd.read_parquet(f"{APDE_POS}/{f}")
        tag = f.split("__")[0]
        total_pos += dump_rows(df, f"{OUT_IMG}/pos", tag, per)
    # 负
    dn = pd.read_parquet(APDE_NEG)
    total_neg = dump_rows(dn, f"{OUT_IMG}/neg", "clean", n_neg)
    print(f"[apde] pos={total_pos} (from {len(pos_files)} attacks)  neg={total_neg}", flush=True)
    return total_pos, total_neg


def main():
    import torch
    from PIL import Image
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from lora_mllm import load_lora_for_eval, PROMPT

    print("=== Part 1: 构建 APDE 图像集 ===", flush=True)
    build()

    print("=== Part 2: NP-PatchLM 零样本评测 ===", flush=True)
    model, proc = load_lora_for_eval(f"{TVC}/lora_patch_detector", 16)
    cfg = json.load(open(f"{TVC}/lora_patch_detector/train_config.json"))
    yes_id, no_id = cfg["yes_id"], cfg["no_id"]

    def score_dir(d):
        out = []
        files = sorted(os.listdir(d))
        for i, f in enumerate(files):
            im = Image.open(os.path.join(d, f)).convert("RGB")
            im.thumbnail((512, 512))
            msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
            t = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            enc = {k: (v.cuda() if torch.is_tensor(v) else v)
                   for k, v in proc(text=[t], images=[im], return_tensors="pt").items()}
            with torch.no_grad():
                lg = model(**enc).logits[0, -1]
            out.append(float(torch.sigmoid(lg[yes_id] - lg[no_id])))
            if (i + 1) % 50 == 0:
                print(f"  {d.split('/')[-1]} {i+1}/{len(files)}", flush=True)
        return out

    pos = np.array(score_dir(f"{OUT_IMG}/pos"))
    neg = np.array(score_dir(f"{OUT_IMG}/neg"))
    res = {"benchmark": "APDE test (ICCV 2025) — zero-shot, no retraining",
           "n_pos": len(pos), "n_neg": len(neg),
           "pos_mean": round(float(pos.mean()), 4), "neg_mean": round(float(neg.mean()), 4),
           "pos_scores": [round(float(x), 4) for x in pos],
           "neg_scores": [round(float(x), 4) for x in neg]}
    print(f"\nAPDE: pos_mean={pos.mean():.4f} neg_mean={neg.mean():.4f}")
    for a in (0.01, 0.05, 0.10, 0.20, 0.33):
        thr = float(np.percentile(neg, 100 * (1 - a)))
        tpr = float((pos >= thr).mean())
        res[f"TPR@{int(a*100)}pct"] = {"threshold": round(thr, 4), "tpr": round(tpr, 4)}
        print(f"  目标 FPR={int(a*100):2d}%: TPR={tpr:.4f}", flush=True)
    json.dump(res, open(f"{TVC}/apde_eval.json", "w"), indent=1, ensure_ascii=False)
    print(f"[apde] wrote {TVC}/apde_eval.json")


if __name__ == "__main__":
    main()
