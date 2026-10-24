#!/usr/bin/env python3
"""
COCO 补丁数据集构建（对齐 papers 的主数据集协议）
==================================================
papers/ 里 AdvPatchXAI (CVPR25)、APEX (USENIX26)、AntiStyler (CVPR26) 都以 MS COCO
为主评测集（AntiStyler 还加 INRIA/Superstore/APRICOT）。当前项目主实验只用 CASIA
（篡改数据集），补丁任务上属于错配。本脚本用本地 COCO val2017 构建 clean / patch
两域，供补丁检测评测，摆脱"只依赖 APRICOT"。

补丁生成复用已验证的 scripts/patch_attack.py (AdversarialPatchGeneratorV2 + PCSimulation)。

用法:
  $PY code/experiments_tvc/build_coco_patches.py --n 600 --out <SRC>/results/features/coco_patch
输出: <out>/{clean,physical_patch,digital_patch}.pt  与 dinov2 特征可复用
"""
import argparse, json, os, random, sys, time

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)

DEFAULT_IMG_DIR = "/media/oyp/工作/Projects/042_image_forensic/DuetGuard/data/coco/val2017"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def collect_images(root):
    """递归收集图像（INRIA 等含子目录），返回绝对路径列表。"""
    out = []
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                out.append(os.path.join(dirpath, f))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600, help="base image 数")
    ap.add_argument("--img-dir", default=DEFAULT_IMG_DIR, help="base 图像目录（可含子目录）")
    ap.add_argument("--tag", default="coco_patch", help="输出子目录名")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--patch-size", type=int, default=128)
    ap.add_argument("--severity", default="strong")
    args = ap.parse_args()

    import torch
    import torchvision.transforms as T
    from PIL import Image
    from scripts.patch_attack import AdversarialPatchGeneratorV2
    from scripts.pc_simulation import PCSimulation

    out = args.out or os.path.join(os.path.dirname(CODE_DIR), "results", "features", args.tag)
    os.makedirs(out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    imgs = collect_images(args.img_dir)
    rng = random.Random(args.seed)
    rng.shuffle(imgs)
    imgs = imgs[:args.n]
    print(f"[{args.tag}] 选定 {len(imgs)} 张 base image <- {args.img_dir}  -> {out}")

    gen = AdversarialPatchGeneratorV2(patch_size=args.patch_size, device=device)
    pc = PCSimulation(severity=args.severity)
    tf = T.Compose([T.Resize((512, 512)), T.ToTensor()])

    res = {"clean": [], "digital_patch": [], "physical_patch": []}
    t0 = time.time()
    for i, path in enumerate(imgs):
        fn = os.path.basename(path)
        try:
            img = tf(Image.open(path).convert("RGB"))
        except Exception as e:
            print(f"  skip {fn}: {e}"); continue
        try:
            mask, content = gen.generate_strong_pgd_patch(img, loss_type="color", strategy="random")
            img_d = gen.apply_patch(img, mask, content)
            img_p = gen.apply_patch(img, mask, pc.simulate(content))
        except Exception as e:
            print(f"  attack fail {fn}: {e}"); continue
        res["clean"].append((img, 0, fn))
        res["digital_patch"].append((img_d, 1, fn))
        res["physical_patch"].append((img_p, 1, fn))
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  [{i+1}/{len(imgs)}] {el:.0f}s ETA {el/(i+1)*(len(imgs)-i-1):.0f}s", flush=True)

    # 60/20/20 split，**按 base image 分组**（同一 base 的三版本同 split，避免 R#4-3 泄漏）
    keys = sorted({fn for _, _, fn in res["clean"]})
    rng2 = random.Random(args.seed)
    rng2.shuffle(keys)
    n = len(keys)
    n_tr, n_va = int(n * 0.6), int(n * 0.2)
    assign = {}
    for j, k in enumerate(keys):
        assign[k] = "train" if j < n_tr else ("val" if j < n_tr + n_va else "test")

    for dom, items in res.items():
        buckets = {"train": [], "val": [], "test": []}
        for img, lab, fn in items:
            buckets[assign[fn]].append((img, lab, fn))
        for sp, lst in buckets.items():
            p = os.path.join(out, f"{dom}_{sp}.pt")
            torch.save({"tensors": [x[0] for x in lst],
                        "labels": [x[1] for x in lst],
                        "base_ids": [x[2] for x in lst]}, p)
            print(f"  {dom}/{sp}: {len(lst)} -> {p}")

    json.dump({"n_base": len(keys), "split": {k: assign[k] for k in keys},
               "seed": args.seed, "grouped_split": True,
               "source": args.img_dir}, open(os.path.join(out, "meta.json"), "w"), indent=2)
    print(f"[{args.tag}] done ({time.time()-t0:.0f}s), grouped by base image")


if __name__ == "__main__":
    main()
