#!/usr/bin/env python3
"""
为 AdvPatchXAI 准备 ImageFolder 格式数据
============================================
其数据约定：<root>/{real/, patched/}（train）与 <root>/test/{real,patched}
我们把：
  训练  : lora_data（label 1→patched, 0→real）
  评测A : unified_eval_set（pos→patched, 四源 neg→real）
  评测B : apde_imgs（pos→patched, neg→real）
输出到 /home/oyp/fastdata/advpatchxai_data/{train,apricot,apde}
"""
import os, json, shutil
import numpy as np

WS = "/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard"
TVC = f"{WS}/results/tvc"
OUT = "/home/oyp/fastdata/advpatchxai_data"


def link_or_copy(src, dst_dir, name=None):
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, name or os.path.basename(src))
    if not os.path.exists(dst):
        try: os.link(src, dst)          # 硬链接，省空间
        except Exception: shutil.copy2(src, dst)


def main():
    idx = json.load(open(f"{TVC}/lora_data/index.json"))
    tr = [r for r in idx if r["split"] == "train"]
    for r in tr:
        sub = "patched" if r["label"] == 1 else "real"
        link_or_copy(r["path"], f"{OUT}/train/{sub}")
    print(f"train: real={len(os.listdir(f'{OUT}/train/real'))} "
          f"patched={len(os.listdir(f'{OUT}/train/patched'))}")

    EQ = f"{TVC}/unified_eval_set"
    for f in sorted(os.listdir(f"{EQ}/pos")):
        link_or_copy(f"{EQ}/pos/{f}", f"{OUT}/apricot/patched", f"pos_{f}")
    for s in ("casia_au", "coco_val", "inria_noped", "columbia_au"):
        for f in sorted(os.listdir(f"{EQ}/{s}")):
            link_or_copy(f"{EQ}/{s}/{f}", f"{OUT}/apricot/real", f"{s}_{f}")
    print(f"apricot: patched={len(os.listdir(f'{OUT}/apricot/patched'))} "
          f"real={len(os.listdir(f'{OUT}/apricot/real'))}")

    AP = f"{TVC}/apde_imgs"
    for f in sorted(os.listdir(f"{AP}/pos")):
        link_or_copy(f"{AP}/pos/{f}", f"{OUT}/apde/patched", f"pos_{f}")
    for f in sorted(os.listdir(f"{AP}/neg")):
        link_or_copy(f"{AP}/neg/{f}", f"{OUT}/apde/real", f"neg_{f}")
    print(f"apde: patched={len(os.listdir(f'{OUT}/apde/patched'))} "
          f"real={len(os.listdir(f'{OUT}/apde/real'))}")


if __name__ == "__main__":
    main()
