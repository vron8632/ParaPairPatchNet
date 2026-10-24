#!/usr/bin/env python3
"""
Prompt 变体鲁棒性（扩展实验 的一部分）
================================================
主模型对 4 种 prompt 措辞的敏感性（不重训，仅推理）。
输出: results/tvc/prompt_robustness.json
"""
import os, sys, json
import numpy as np

WS = "/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard"
TVC = f"{WS}/results/tvc"
BASE = f"{TVC}/baseline_imgs"
S3 = ["casia_au", "coco_val", "inria_noped"]

PROMPTS = {
 "P1_original": "Is there a suspicious printed sticker or adversarial patch in the image? Answer YES or NO.",
 "P2_short": "Does this image contain an adversarial patch? YES or NO.",
 "P3_forensic": "Is the image free of any printed stickers or patches? Answer YES or NO.",
 "P4_cot": "Carefully examine the image for any printed sticker-like object that could be an adversarial patch. Is such a patch present? YES or NO.",
}


def main():
    import torch
    from PIL import Image
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from lora_mllm import load_lora_for_eval
    model, proc = load_lora_for_eval(f"{TVC}/lora_patch_detector", 16)
    cfg = json.load(open(f"{TVC}/lora_patch_detector/train_config.json"))
    yes_id, no_id = cfg["yes_id"], cfg["no_id"]
    tok = proc.tokenizer

    pos_dir = f"{BASE}/pos_all" if os.path.isdir(f"{BASE}/pos_all") else f"{BASE}/pos"
    all_files = [(f"{pos_dir}/{f}", 1) for f in sorted(os.listdir(pos_dir))]
    for s in S3 + ["columbia_au"]:
        all_files += [(f"{BASE}/{s}/{f}", 0) for f in sorted(os.listdir(f"{BASE}/{s}"))]

    # 注意 P3 语义反转（问"是否无补丁"）→ 取其 YES 概率的补概率
    res = {}
    for name, prompt in PROMPTS.items():
        scores, labels = [], []
        for i, (fp, y) in enumerate(all_files):
            im = Image.open(fp).convert("RGB"); im.thumbnail((512, 512))
            msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            t = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            enc = {k: (v.cuda() if torch.is_tensor(v) else v)
                   for k, v in proc(text=[t], images=[im], return_tensors="pt").items()}
            with torch.no_grad():
                lg = model(**enc).logits[0, -1]
            s = float(torch.sigmoid(lg[yes_id] - lg[no_id]))
            if "free of" in prompt: s = 1.0 - s      # 语义反转校正
            scores.append(s); labels.append(y)
            if (i+1) % 150 == 0: print(f"  {name} {i+1}/{len(all_files)}", flush=True)
        sc, lb = np.array(scores), np.array(labels)
        pos_s, neg_s = sc[lb == 1], sc[lb == 0]
        row = {"n_pos": int((lb == 1).sum()), "n_neg": int((lb == 0).sum()),
               "pos_mean": round(float(pos_s.mean()), 4), "neg_mean": round(float(neg_s.mean()), 4)}
        for a in (0.05, 0.10):
            thr = np.percentile(neg_s, 100*(1-a))
            row[f"TPR@{int(a*100)}pct"] = round(float((pos_s >= thr).mean()), 4)
        res[name] = row
        print(f"  {name}: pos_mean={row['pos_mean']} neg_mean={row['neg_mean']} "
              f"@5%={row['TPR@5pct']} @10%={row['TPR@10pct']}", flush=True)
    json.dump(res, open(f"{TVC}/prompt_robustness.json", "w"), indent=1, ensure_ascii=False)
    print(f"→ {TVC}/prompt_robustness.json")


if __name__ == "__main__":
    main()
