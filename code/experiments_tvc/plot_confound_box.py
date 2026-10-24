#!/usr/bin/env python3
"""
画 confound 分来源分数箱线图（支撑"跨来源 FPR 系统性差异"这一局限）
用法: $PY code/experiments_tvc/plot_confound_box.py \
        --json results/tvc/confound_main_withscores.json \
        --out figures/confound_per_source_box.pdf
输入 json 需含 pos_scores / neg_scores_by_source（confound_check.py 新增字段）。
"""
import argparse, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    d = json.load(open(args.json, encoding="utf-8"))
    neg = d.get("neg_scores_by_source")
    pos = d.get("pos_scores")
    if not neg or not pos:
        raise SystemExit("json 缺少 pos_scores / neg_scores_by_source 字段，请用新版 confound_check.py 重跑")

    labels, data = [], []
    for src in sorted(neg):
        if len(neg[src]):
            labels.append(src); data.append(np.array(neg[src]))
    labels.append("APRICOT\n(patch, pos)"); data.append(np.array(pos))

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    bp = ax.boxplot(data, labels=labels, showmeans=True, widths=0.6,
                    patch_artist=True, medianprops=dict(color="black", lw=1.4))
    n = len(data)
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor("#d94b4b" if i == n - 1 else "#5b8dd6")
        box.set_alpha(0.75)

    ax.set_ylabel("P(YES)  (patch probability)", fontsize=10)
    ax.set_ylim(-0.03, 1.03)
    ax.axhline(0.5, ls="--", c="gray", lw=0.8)
    ax.grid(axis="y", ls=":", alpha=0.4)
    plt.setp(ax.get_xticklabels(), fontsize=8.5)
    if args.title:
        ax.set_title(args.title, fontsize=10)
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=200)
    # 同时给一份 png 方便预览
    png = os.path.splitext(args.out)[0] + ".png"
    fig.savefig(png, dpi=150)
    print(f"wrote {args.out}\nwrote {png}")

    # 打印关键统计，便于写正文
    print("\n=== 各来源 P(YES) 分布 ===")
    for lab, arr in zip(labels, data):
        print(f"  {lab:24s} n={len(arr):4d} median={np.median(arr):.4f} "
              f"mean={arr.mean():.4f} p90={np.percentile(arr,90):.4f} max={arr.max():.4f}")


if __name__ == "__main__":
    main()
