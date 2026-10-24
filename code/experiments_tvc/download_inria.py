#!/usr/bin/env python3
"""
INRIA Person 下载（走 hf-mirror，避开 datasets 库对 huggingface.co 的直连）
============================================================================
背景: `datasets` 库即使设了 HF_ENDPOINT 仍会请求 huggingface.co（本机不可达），
因此改用 hf-mirror 的 REST API 列文件 + 并行下载原图。

数据集结构: data_ped/no_pedestrians/*.png (无行人) 与 data_ped/pedestrians/*.png (有行人)
用途: 作为**真实自然图像**来源 —— (a) 负样本（无补丁）；(b) 生成 INRIA 补丁集（补齐 AntiStyler/APEX
      使用的 INRIA 评测集，摆脱"只用 CASIA/COCO"）。

用法:
  $PY code/experiments_tvc/download_inria.py --max 800
输出: <DUETGUARD>/data/inria/{no_pedestrians,pedestrians}/*.png
"""
import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import urllib.request

REPO = "marcelarosalesj/inria-person"
EP = "https://hf-mirror.com"
OUT = "/media/oyp/工作/Projects/042_image_forensic/DuetGuard/data/inria"


def _curl_json(url):
    import subprocess
    r = subprocess.run(["curl", "-sL", "--max-time", "60",
                        "-H", "User-Agent: Mozilla/5.0", url],
                       capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip():
        raise RuntimeError(f"curl failed ({r.returncode}) for {url}")
    return json.loads(r.stdout)


def list_tree(path="data_ped"):
    return _curl_json(f"{EP}/api/datasets/{REPO}/tree/main/{path}?recursive=true")


def _valid_image(path):
    """校验下载结果确实是可解码的图片（hf-mirror 有时返回错误页，大小非 0 但内容无效）。"""
    try:
        from PIL import Image
        with Image.open(path) as im:
            im.verify()
        return os.path.getsize(path) > 1000
    except Exception:
        return False


def download(rel):
    """用 curl 下载 + 有效性校验（实测 urllib 对 hf-mirror 文件端点会超时）。"""
    import subprocess
    dst = os.path.join(OUT, rel)
    if os.path.exists(dst) and _valid_image(dst):
        return ("skip", rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    url = f"{EP}/datasets/{REPO}/resolve/main/{rel}"
    for attempt in range(4):
        r = subprocess.run(["curl", "-sL", "--max-time", "120", "-o", dst, url],
                           capture_output=True)
        if r.returncode == 0 and _valid_image(dst):
            return ("ok", rel)
        time.sleep(1 + attempt)
    return ("fail", rel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=800, help="每类最多下载张数")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    entries = list_tree("data_ped")
    files = [e["path"] for e in entries if e.get("path", "").endswith(".png")]
    neg = [f for f in files if "/no_pedestrians/" in f]
    pos = [f for f in files if "/pedestrians/" in f]
    print(f"[inria] 可用: no_pedestrians={len(neg)} pedestrians={len(pos)}")
    sel = neg[:args.max] + pos[:args.max]
    print(f"[inria] 计划下载 {len(sel)} 张 -> {OUT}")

    t0 = time.time()
    stats = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, (st, rel) in enumerate(ex.map(download, sel)):
            key = st.split(":")[0]
            stats[key] = stats.get(key, 0) + 1
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(sel)} ({time.time()-t0:.0f}s) {stats}", flush=True)
    print(f"[inria] done: {stats}  ({time.time()-t0:.0f}s)")
    json.dump({"repo": REPO, "n_neg": len(neg), "n_pos": len(pos), "downloaded": stats},
              open(os.path.join(OUT, "meta.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
