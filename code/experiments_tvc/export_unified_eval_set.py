#!/usr/bin/env python3
"""
落盘统一评测集（供所有按目录读数据的方法共用）
================================================
把统一协议的正/负样本导出为目录结构：
  results/tvc/unified_eval_set/{pos,casia_au,coco_val,inria_noped,columbia_au}
正样本 = APRICOT 本地全部 138 张；负样本 = 4 源各 250 张（authentic columbia），
抽取逻辑与 unified_main_1024.collect_neg 完全一致（同 seed 与顺序）。
"""
import os, json, shutil
import numpy as np
from PIL import Image

WS = "/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard"
TVC = f"{WS}/results/tvc"
D = "/media/oyp/工作/Projects/042_image_forensic/DuetGuard/data"
OUT = f"{TVC}/unified_eval_set"
IMG_EXT = (".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff")
N_PER = 250


def main():
    os.makedirs(f"{OUT}/pos", exist_ok=True)
    # 正样本：复用已落盘的 pos_all（138 张）
    src_pos = f"{TVC}/baseline_imgs/pos_all"
    for f in sorted(os.listdir(src_pos)):
        dst = f"{OUT}/pos/{f}"
        if not os.path.exists(dst): shutil.copy(src_pos + "/" + f, dst)
    print(f"pos: {len(os.listdir(f'{OUT}/pos'))}")

    rng = np.random.RandomState(42)     # 与 collect_neg 一致
    manifest = {"pos": {"n": 138, "src": "APRICOT dev split 全部带掩码样本"}, "neg": {}}

    def dump(path, tag, filt=None):
        os.makedirs(f"{OUT}/{tag}", exist_ok=True)
        fs = sorted(f for f in os.listdir(path)
                    if f.lower().endswith(IMG_EXT) and (filt is None or filt(f)))
        pick = [fs[i] for i in rng.permutation(len(fs))[:N_PER]]
        for f in pick:
            dst = f"{OUT}/{tag}/{f}"
            if os.path.exists(dst): continue
            try:
                Image.open(os.path.join(path, f)).convert("RGB").save(dst)
            except Exception: pass
        n = len(os.listdir(f"{OUT}/{tag}"))
        manifest["neg"][tag] = {"n": n, "source_dir": path}
        print(f"  {tag}: {n}")

    dump(f"{D}/casia2/CASIA2.0_revised/Au", "casia_au")
    dump(f"{D}/coco/val2017", "coco_val")
    dump(f"{D}/inria/data_ped/no_pedestrians", "inria_noped")
    # columbia：只取 Au-*（authentic），与 collect_neg 同序
    col = f"{D}/columbia/ImSpliceDataset"
    items = []
    for sub in sorted(os.listdir(col)):
        d2 = os.path.join(col, sub)
        if not os.path.isdir(d2) or not sub.startswith("Au"): continue
        for f in sorted(os.listdir(d2)):
            if f.lower().endswith(IMG_EXT): items.append((sub, f))
    rng.shuffle(items)
    os.makedirs(f"{OUT}/columbia_au", exist_ok=True)
    for sub, f in items[:N_PER]:
        dst = f"{OUT}/columbia_au/{sub}__{f}"
        if not os.path.exists(dst):
            try: Image.open(os.path.join(col, sub, f)).convert("RGB").save(dst)
            except Exception: pass
    manifest["neg"]["columbia_au"] = {"n": len(os.listdir(f"{OUT}/columbia_au")),
                                      "source_dir": col, "note": "authentic only (Au-*)"}
    print(f"  columbia_au: {len(os.listdir(f'{OUT}/columbia_au'))} (authentic)")
    json.dump(manifest, open(f"{OUT}/manifest.json", "w"), indent=1, ensure_ascii=False)
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
