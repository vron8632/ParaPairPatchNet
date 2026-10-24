#!/usr/bin/env python3
"""
粗定位实验（扩展实验）
================================
图像级报警 → 粗定位：用**遮挡法**（occlusion sensitivity）——把图像划分为网格，
逐格用灰块遮挡后重测 VLM 补丁分数；分数下降最大的格子即为补丁所在区域。
优点：不重训、直接复用本文模型，且可用 APRICOT 掩码真值做定量评估（命中率）。

输出:
  results/tvc/patch_localization.json   （命中率等定量指标）
  results/tvc/loc_viz/                  （定性可视化图）
用法: $PY patch_localization.py --n 20 --grid 6
"""
import os, sys, json, argparse
import numpy as np

WS = "/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard"
TVC = f"{WS}/results/tvc"
BASE = f"{TVC}/baseline_imgs"
APRICOT_D = "/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard/data/apricot/dev/data_mask_v2"
sys.path.insert(0, f"{WS}/code/experiments_tvc")


def load_mask_for(fname):
    """从 APRICOT .pt 读取像素级 mask（真值），返回 (mask_bool, 原图尺寸)"""
    import torch
    p = os.path.join(APRICOT_D, fname.replace(".png", ".pt"))
    if not os.path.exists(p): return None
    x = torch.load(p, map_location="cpu", weights_only=False)
    m = x["Mask"]
    # 注意：APRICOT .pt 中 Mask/Image 存的是 numpy 数组（非 tensor），需兼容两种类型
    m = m[0] if getattr(m, "ndim", 2) == 3 else m
    m_np = m.numpy() if hasattr(m, "numpy") else np.asarray(m)
    img = x["Image"]
    return (m_np > 0), tuple(img.shape[1:3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=f"{TVC}/lora_patch_detector")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--grid", type=int, default=6)
    ap.add_argument("--out", default=f"{TVC}/patch_localization.json")
    args = ap.parse_args()

    import torch
    from PIL import Image, ImageDraw
    from lora_mllm import load_lora_for_eval, PROMPT

    model, proc = load_lora_for_eval(args.ckpt, 16)
    cfg = json.load(open(f"{args.ckpt}/train_config.json")); yes_id, no_id = cfg["yes_id"], cfg["no_id"]

    def score(im):
        msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        t = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        enc = {k: (v.cuda() if torch.is_tensor(v) else v)
               for k, v in proc(text=[t], images=[im], return_tensors="pt").items()}
        with torch.no_grad():
            lg = model(**enc).logits[0, -1]
        return float(torch.sigmoid(lg[yes_id] - lg[no_id]))

    pos_dir = f"{BASE}/pos_all"
    files = sorted(os.listdir(pos_dir))[:args.n]
    viz_dir = f"{TVC}/loc_viz"; os.makedirs(viz_dir, exist_ok=True)
    g = args.grid
    hits, n_valid, rows = 0, 0, []

    for fi, f in enumerate(files):
        im0 = Image.open(os.path.join(pos_dir, f)).convert("RGB")
        im0 = im0.resize((512, int(512 * im0.height / im0.width))) if max(im0.size) > 512 else im0
        W, H = im0.size
        s0 = score(im0)
        heat = np.zeros((g, g), dtype=np.float32)
        for r in range(g):
            for c in range(g):
                im = im0.copy(); d = ImageDraw.Draw(im)
                x0, y0 = int(c * W / g), int(r * H / g)
                x1, y1 = int((c + 1) * W / g), int((r + 1) * H / g)
                d.rectangle([x0, y0, x1, y1], fill=(128, 128, 128))
                heat[r, c] = s0 - score(im)          # 遮挡后分数下降幅度
        best = np.unravel_index(int(heat.argmax()), heat.shape)
        # 真值命中判定：预测格中心是否落在 mask 内
        mask = load_mask_for(f)
        hit = None
        if mask is not None:
            m, (Hh, Ww) = mask
            cx = int((best[1] + 0.5) * Ww / g); cy = int((best[0] + 0.5) * Hh / g)
            if 0 <= cy < Hh and 0 <= cx < Ww:
                hit = bool(m[cy, cx]); n_valid += 1
                hits += int(hit)
        rows.append({"file": f, "score_clean": round(s0, 4),
                     "max_drop": round(float(heat.max()), 4),
                     "pred_cell": [int(best[0]), int(best[1])], "hit": hit})
        # 可视化
        viz = im0.copy(); d = ImageDraw.Draw(viz)
        hn = (heat - heat.min()) / max(heat.max() - heat.min(), 1e-6)
        for r in range(g):
            for c in range(g):
                if hn[r, c] > 0.5:
                    a = int(120 * hn[r, c])
                    d.rectangle([int(c*W/g), int(r*H/g), int((c+1)*W/g), int((r+1)*H/g)],
                                outline=(255, 60, 0), width=3)
        viz.save(f"{viz_dir}/{f.replace('.png','_loc.png')}")
        if (fi+1) % 5 == 0:
            print(f"  {fi+1}/{len(files)}  当前命中 {hits}/{n_valid}", flush=True)

    res = {"ckpt": os.path.basename(args.ckpt), "n": len(files), "grid": g,
           "hit_rate": round(hits / n_valid, 4) if n_valid else None,
           "n_evaluated": n_valid, "per_image": rows}
    print(f"\n[loc] 粗定位命中率 {hits}/{n_valid} = {res['hit_rate']}")
    json.dump(res, open(args.out, "w"), indent=1, ensure_ascii=False)
    print(f"→ {args.out}；可视化 → {viz_dir}/")


if __name__ == "__main__":
    main()
