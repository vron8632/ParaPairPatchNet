#!/usr/bin/env python3
"""
APRICOT 含负样本评估（negative-control protocol）— ForensicGuard
=========================================================================
目的: 针对对 APRICOT 零样本评估的三条质疑:
  1. 只有正样本、无 clean 对照 -> 引入 COCO / CASIA-Au 作为负样本, 报告 FPR/AUC;
  2. prob 全部 1.0、无分数分布    -> 输出正/负样本概率分布(分位数+直方图)与 ROC-AUC;
  3. 标签口径含糊                -> 用 --patch_threshold_frac 显式控制 APRICOT 补丁真值阈值,
                                    并在结果中记录每张图的 mask 占比供核对。

用法:
    python3 code/run_apricot_neg_eval.py --help
    python3 code/run_apricot_neg_eval.py                          # 默认: APRICOT dev vs COCO(200)
    python3 code/run_apricot_neg_eval.py --neg both --max_neg 500 # COCO + CASIA-Au 各采样
    python3 code/run_apricot_neg_eval.py --segmenter sam          # 与 run_sam_apricot 对比
    python3 code/run_apricot_neg_eval.py --pos both               # 正样本含 test split(大, 慢)

输出:
    results/neg_eval/<脚本名>_<主标记>_<版本>_<时间戳>.json — 完整结果
        例: run_apricot_neg_eval_yolo26-seg_v1_20260801_003000.json
        例: run_apricot_neg_eval_sam_v2_20260801_004500.json
    results/neg_eval/<脚本名>_<主标记>_<版本>_<时间戳>.log  — 终端镜像日志
    results/neg_eval/run_apricot_neg_eval_latest.json/.log  — 快捷指针(非正式结果)
"""
import argparse, os, sys, time, json, glob, zipfile, random
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

BASE = '/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard'
sys.path.insert(0, os.path.join(BASE, 'code'))

from models.consistency import CrossInstanceConsistencyV2
from models.fusion_classifier import FusionClassifierV2
from scripts.train import compute_fused_features
from features.spn_extractor import get_spn_extractor
from features.dct_profile import DCTProfileExtractor
from features.clip_encoder import get_clip_extractor

OUT_DIR = os.path.join(BASE, 'results/neg_eval')
LOG_DIR = os.path.join(BASE, 'results/logs')
SCRIPT = os.path.splitext(os.path.basename(__file__))[0]  # run_apricot_neg_eval
APRICOT_ZIPS = {'dev': os.path.join(BASE, 'data/apricot/dev.zip'),
                'test': os.path.join(BASE, 'data/apricot/test.zip')}
NEG_DIRS = {'coco': os.path.join(BASE, 'data/coco/val2017'),
            'casia': os.path.join(BASE, 'data/casia2/CASIA2.0_revised/Au')}

transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
MIN_MASK_PIX = 64 * 64          # 与 run_sam_apricot 一致的最小实例过滤
TAUS_DEFAULT = [0.3, 0.5, 0.7, 0.8]


def fmt_eta(secs):
    secs = int(max(0, secs)); h, r = divmod(secs, 3600); m, s = divmod(r, 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


def log(msg, fh=None):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    if fh is not None:
        fh.write(line + '\n'); fh.flush()


def load_pipeline(device, segmenter_kind, model_path, log_fn=None):
    def info(msg):
        if log_fn:
            log_fn(msg)
    cic = CrossInstanceConsistencyV2()
    model = FusionClassifierV2(input_dim=698)
    info(f'加载融合模型 {model_path} ...')
    model.load_state_dict(torch.load(
        os.path.join(BASE, 'results/models', model_path),
        map_location='cpu', weights_only=True))
    model = model.to(device).eval()
    info('加载 SPN 提取器 ...')
    spn = get_spn_extractor(mode='learned', device=device)
    info('加载 DCT 提取器 ...')
    dct = DCTProfileExtractor()
    info('加载 CLIP 编码器 ...')
    clip_img = get_clip_extractor(device=device)

    if segmenter_kind == 'yolo26-seg':
        from features.segmenter import InstanceSegmenter
        info('加载分割器 YOLO26-seg ...')
        seg_model = InstanceSegmenter(device=device)
    elif segmenter_kind == 'sam':
        cands = ['sam_b.pt', os.path.expanduser('~/weights/sam_b.pt'),
                 os.path.join(BASE, 'weights/sam_b.pt')]
        found = next((c for c in cands if os.path.exists(c)), None)
        info('加载分割器 SAM ...')
        if found:
            seg_model = __import__('ultralytics').SAM(found)
        else:
            seg_model = __import__('ultralytics').SAM('sam_b.pt')  # 自动下载
    else:
        raise ValueError(segmenter_kind)
    return cic, model, spn, dct, clip_img, seg_model


def get_masks(img_orig, seg_model, segmenter_kind, device):
    """返回实例 mask 列表（512x512 bool ndarray）。yolo26-seg 走论文管线，sam 复刻 run_sam_apricot。"""
    if segmenter_kind == 'yolo26-seg':
        out = seg_model.segment(img_orig.to(device))
        return [m for m, _ in out if m.sum() > MIN_MASK_PIX]
    # SAM 分支
    img_sam = (img_orig.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    res = seg_model(img_sam, verbose=False)
    if res is None or len(res) == 0 or res[0].masks is None:
        return []
    masks = []
    for m in res[0].masks.data.cpu().numpy():   # (N, H_sam, W_sam)
        m_pil = Image.fromarray((m * 255).astype(np.uint8)).resize((512, 512))
        m512 = np.array(m_pil) > 0
        if m512.sum() > MIN_MASK_PIX:
            masks.append(m512)
    return masks


@torch.no_grad()
def predict_one(img_orig, masks, cic, model, spn, dct, clip_img, device):
    """与 run_sam_apricot.py 完全一致的推理路径，返回 (K, prob)。"""
    K = len(masks)
    if K < 1:
        spn_f = spn.extract(img_orig.to(device)).unsqueeze(0)
        dct_f = dct.extract_profile(img_orig.cpu()).unsqueeze(0)
        clip_f = clip_img.encode_image(img_orig.cpu()).unsqueeze(0)
    else:
        sl, dl, cl = [], [], []
        for m in masks:
            mt = torch.from_numpy(m).float().to(device)
            sl.append(spn.extract_instance(img_orig.to(device), mt))
            dl.append(dct.extract_profile(img_orig.cpu(), m))
            cl.append(clip_img.encode_image(img_orig.cpu(), m))
        spn_f, dct_f, clip_f = torch.stack(sl), torch.stack(dl), torch.stack(cl)
    spn_f, dct_f, clip_f = spn_f.to(device), dct_f.to(device), clip_f.to(device)
    fused = compute_fused_features([spn_f], [dct_f], [clip_f], cic, device)
    prob = torch.sigmoid(model(fused)).item()
    return K, prob


def load_image(path_or_img):
    """统一预处理: 任意来源 -> img_orig ([0,1] 反归一化 tensor, 3,512,512)。"""
    if isinstance(path_or_img, str):
        img_pil = Image.open(path_or_img).convert('RGB')
    elif isinstance(path_or_img, Image.Image):
        img_pil = path_or_img.convert('RGB')
    else:  # 已是 tensor (3,512,512)
        return path_or_img * std + mean
    return transform(img_pil) * std + mean


def collect_apricot(pos_splits, max_pos, seed, progress=None):
    """读取 APRICOT zip 的 .pt 文件, 返回任务列表 [(source, key, 图片tensor, label, mask_frac)]。
    label 由补丁真值 mask 占比 > patch_threshold_frac 决定（运行时传入阈值, 这里先留原始信息）。
    progress: 可选回调 progress(split, i, total, last_file_secs), 用于打印读取进度/ETA。"""
    tasks = []
    for split in pos_splits:
        zip_path = APRICOT_ZIPS[split]
        with zipfile.ZipFile(zip_path, 'r') as z:
            pt_files = sorted([f for f in z.namelist() if f.endswith('.pt')])
            total = len(pt_files)
            for i, pt_name in enumerate(pt_files, 1):
                t0f = time.time()
                with z.open(pt_name) as f:
                    data = torch.load(f, map_location='cpu', weights_only=False)
                img_np = data['Image'][0]                 # (H,W,3) uint8 或 float[0,1]
                mask = np.asarray(data['Mask'], dtype=bool)  # (H,W) 补丁真值
                # 修复: float[0,1] 直接 astype(uint8) 会被截断成全黑, 需先 *255
                if img_np.dtype != np.uint8:
                    img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
                img_pil = Image.fromarray(img_np)
                img_orig = load_image(img_pil)
                frac = float(mask.sum()) / float(mask.size)
                tasks.append({'source': f'apricot-{split}', 'key': pt_name,
                              'img': img_orig, 'patch_frac': frac})
                if progress:
                    progress(split, i, total, time.time() - t0f)
    rng = random.Random(seed)
    if max_pos and len(tasks) > max_pos:
        rng.shuffle(tasks)
        tasks = tasks[:max_pos]
    return tasks


def collect_negatives(neg_names, max_neg, seed):
    tasks = []
    for name in neg_names:
        d = NEG_DIRS[name]
        exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')
        files = [f for e in exts for f in glob.glob(os.path.join(d, e))]
        files = sorted(files)
        rng = random.Random(seed + hash(name) % 10000)
        if max_neg and len(files) > max_neg:
            rng.shuffle(files)
            files = files[:max_neg]
        for fp in files:
            tasks.append({'source': f'clean-{name}', 'key': os.path.basename(fp), 'path': fp})
    return tasks


def compute_auc(labels, scores):
    from scipy.stats import rankdata
    labels = np.asarray(labels); scores = np.asarray(scores)
    n_pos = int(labels.sum()); n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    ranks = rankdata(scores)  # ties 取平均秩
    sum_pos = float(ranks[labels == 1].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def summarize(records, taus, out_dir, ts):
    labels = np.array([r['label'] for r in records])
    scores = np.array([r['prob'] for r in records])
    n_pos = int(labels.sum()); n_neg = len(labels) - n_pos
    auc = compute_auc(labels, scores)

    # 多 tau 报告
    tau_table = []
    for tau in taus:
        pred = scores > tau
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        tn = int(((pred == 0) & (labels == 0)).sum())
        tpr = tp / n_pos if n_pos else float('nan')
        fpr = fp / n_neg if n_neg else float('nan')
        tau_table.append({'tau': tau, 'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
                          'TPR': round(tpr, 4), 'FPR': round(fpr, 4)})

    # 正/负样本 prob 分布
    def dist(scores_sub):
        if len(scores_sub) == 0:
            return None
        q = np.percentile(scores_sub, [0, 25, 50, 75, 100])
        counts, edges = np.histogram(scores_sub, bins=20, range=(0.0, 1.0))
        return {'n': len(scores_sub),
                'min': round(float(q[0]), 4), 'p25': round(float(q[1]), 4),
                'median': round(float(q[2]), 4), 'p75': round(float(q[3]), 4),
                'max': round(float(q[4]), 4),
                'mean': round(float(scores_sub.mean()), 4),
                'hist_counts': counts.tolist(), 'hist_edges': edges.tolist()}

    pos_dist = dist(scores[labels == 1])
    neg_dist = dist(scores[labels == 0])

    # K 分布（按正/负分组）
    def k_dist(records_sub):
        d = {}
        for r in records_sub:
            d.setdefault(r['K'], 0)
            d[r['K']] += 1
        return d

    summary = {
        'n_pos': n_pos, 'n_neg': n_neg, 'n_total': len(records),
        'auc': round(auc, 4) if auc == auc else None,
        'tau_table': tau_table,
        'pos_prob_dist': pos_dist, 'neg_prob_dist': neg_dist,
        'pos_k_dist': k_dist([r for r in records if r['label'] == 1]),
        'neg_k_dist': k_dist([r for r in records if r['label'] == 0]),
        'pos_saturated_frac': round(float((scores[labels == 1] >= 0.9999).mean()), 4),
    }
    return summary


def main():
    ap = argparse.ArgumentParser(description='APRICOT 含负样本评估 (negative-control protocol)')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--segmenter', choices=['yolo26-seg', 'sam'], default='yolo26-seg',
                    help='实例分割器/实验主标记: yolo26-seg(论文管线, 默认) | sam(与 run_sam_apricot 对比)')
    ap.add_argument('--pos', default='dev', help='APRICOT split: dev | test | both (test 很大, 慎用)')
    ap.add_argument('--neg', default='coco', help='负样本集: coco | casia | both')
    ap.add_argument('--max_pos', type=int, default=0, help='正样本上限, 0=全部(默认 138)')
    ap.add_argument('--max_neg', type=int, default=200, help='每类负样本上限, 0=全部')
    ap.add_argument('--neg_seed', type=int, default=42)
    ap.add_argument('--patch_threshold_frac', type=float, default=0.01,
                    help='APRICOT 补丁真值阈值: mask 占比超过该值判为正样本 (旧脚本口径=0.01, 得 132 正)')
    ap.add_argument('--taus', default=','.join(str(t) for t in TAUS_DEFAULT),
                    help='要报告的 τ 列表, 逗号分隔')
    ap.add_argument('--out_dir', default=OUT_DIR)
    ap.add_argument('--tag', default='v1',
                    help='实验版本标签(如 v1/v2/...), 用于区分不同配置的运行; 会进入文件名')
    ap.add_argument('--model', default='digital_baseline_v2/fusion_best.pth',
                    help='待评估模型路径(相对 results/models/), 如 clean_acc_expand/fusion_best.pth')
    args = ap.parse_args()

    taus = [float(x) for x in args.taus.split(',') if x.strip()]
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    # 正式结果命名: <脚本名>_<实验主标记=分割器>_<版本号>_<时间戳>
    json_path = os.path.join(args.out_dir, f'{SCRIPT}_{args.segmenter}_{args.tag}_{ts}.json')
    log_path = os.path.join(args.out_dir, f'{SCRIPT}_{args.segmenter}_{args.tag}_{ts}.log')

    fh = open(log_path, 'w')
    log(f'=== APRICOT 负样本评估 === device={args.device} segmenter={args.segmenter} '
        f'pos={args.pos} neg={args.neg} max_pos={args.max_pos} max_neg={args.max_neg} '
        f'patch_thr={args.patch_threshold_frac} taus={taus}', fh)

    # ==== 阶段 1/3: 收集任务并读取 APRICOT zip（本阶段最慢, 约 12 分钟, 必须输出进度）====
    pos_splits = ['dev', 'test'] if args.pos == 'both' else [args.pos]
    t_phase = time.time()

    def _ap_progress(split, i, total, last_secs):
        el = time.time() - t_phase
        eta = fmt_eta(el / i * (total - i))
        if i % 10 == 0 or i == total:
            log(f'  [阶段1/3 读取APRICOT-{split} {i}/{total}] '
                f'ETA={eta} elapsed={fmt_eta(el)} ({last_secs:.1f}s/张)', fh)

    log(f'阶段 1/3: 读取 APRICOT zip ({pos_splits}, 每张约5s, 预计10-15分钟)...', fh)
    tasks = collect_apricot(pos_splits, args.max_pos, args.neg_seed, progress=_ap_progress)
    log(f'  ✅ 阶段1完成: APRICOT 正样本 {len(tasks)} 张, 耗时 {fmt_eta(time.time()-t_phase)}', fh)

    neg_names = ['coco', 'casia'] if args.neg == 'both' else [args.neg]
    t_neg = time.time()
    tasks += collect_negatives(neg_names, args.max_neg, args.neg_seed)
    log(f'  ✅ 负样本收集完成: {sum(1 for t in tasks if "clean" in t["source"])} 张'
        f' ({",".join(neg_names)}), 耗时 {fmt_eta(time.time()-t_neg)}', fh)
    n = len(tasks)
    log(f'任务总数: {n} (APRICOT 正样本 {sum(1 for t in tasks if "apricot" in t["source"])}, '
        f'clean 负样本 {sum(1 for t in tasks if "clean" in t["source"])})', fh)

    # ==== 阶段 2/3: 加载模型与特征提取器 ====
    t_phase = time.time()
    log(f'阶段 2/3: 加载模型({args.model})与特征提取器...', fh)
    cic, model, spn, dct, clip_img, seg_model = load_pipeline(
        args.device, args.segmenter, args.model,
        log_fn=lambda m: log(f'  [阶段2/3] {m}', fh))
    log(f'  ✅ 阶段2完成: 模型加载完毕, 耗时 {fmt_eta(time.time()-t_phase)}', fh)

    # ==== 阶段 3/3: 推理 ====
    log(f'阶段 3/3: 推理 {n} 张 (每10张打印进度, 含 ETA/elapsed)...', fh)
    records, t0 = [], time.time()
    for idx, t in enumerate(tasks, 1):
        if 'apricot' in t['source']:
            img_orig = t['img']
            label = 1 if t['patch_frac'] > args.patch_threshold_frac else 0
        else:
            img_orig = load_image(t['path'])
            label = 0
        masks = get_masks(img_orig, seg_model, args.segmenter, args.device)
        K, prob = predict_one(img_orig, masks, cic, model, spn, dct, clip_img, args.device)
        pred = 1 if prob > 0.5 else 0
        rec = {'source': t['source'], 'key': t['key'], 'K': int(K),
               'label': int(label), 'pred': pred,
               'prob': round(float(prob), 4), 'correct': int(pred == label)}
        if 'apricot' in t['source']:
            rec['patch_frac'] = round(t['patch_frac'], 6)
        records.append(rec)

        if idx % 10 == 0 or idx == n:
            el = time.time() - t0
            eta = fmt_eta(el / idx * (n - idx))
            done = sum(1 for r in records if r['correct'])
            ks = [r['K'] for r in records]
            log(f'  [{idx}/{n}] acc={done}/{idx} K均值={np.mean(ks):.2f} '
                f'ETA={eta} elapsed={fmt_eta(el)}', fh)
            # 中间态保存（防中断丢失）
            partial = {'status': 'running', 'progress': f'{idx}/{n}', 'results': records}
            with open(json_path, 'w') as f:
                json.dump(partial, f, indent=2, ensure_ascii=False)

    el = time.time() - t0
    log(f'推理完成: {n} 张, 总耗时 {fmt_eta(el)} ({el:.1f}s, {el/n:.2f}s/张)', fh)

    # 汇总
    summary = summarize(records, taus, args.out_dir, ts)
    config = {'device': args.device, 'segmenter': args.segmenter, 'pos': args.pos,
              'neg': args.neg, 'max_pos': args.max_pos, 'max_neg': args.max_neg,
              'neg_seed': args.neg_seed, 'patch_threshold_frac': args.patch_threshold_frac,
              'taus': taus, 'model': args.model,
              'tag': args.tag, 'script_version': '1.4.0',
              'patch_label_basis': 'APRICOT ground-truth Mask, fraction > threshold = positive'}
    payload = {'status': 'done', 'script': 'run_apricot_neg_eval.py', 'tag': args.tag,
               'timestamp': ts, 'config': config, 'summary': summary,
               'elapsed_secs': round(el, 1), 'results': records}
    with open(json_path, 'w') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # latest 快捷指针（带脚本名前缀, 只作查找用, 非正式结果）
    for suffix in ('json', 'log'):
        src = os.path.join(args.out_dir, f'{SCRIPT}_{args.segmenter}_{args.tag}_{ts}.{suffix}')
        dst = os.path.join(args.out_dir, f'{SCRIPT}_latest.{suffix}')
        try:
            with open(src, 'rb') as fi, open(dst, 'wb') as fo:
                fo.write(fi.read())
        except OSError:
            pass

    # 终端汇总输出
    log(f'\n=== 结果 ===', fh)
    log(f'样本: 正 {summary["n_pos"]} / 负 {summary["n_neg"]} / 共 {summary["n_total"]}', fh)
    log(f'AUC = {summary["auc"]}', fh)
    log(f'正样本 prob: {summary["pos_prob_dist"]}', fh)
    log(f'负样本 prob: {summary["neg_prob_dist"]}', fh)
    log(f'正样本饱和率(prob>=0.9999): {summary["pos_saturated_frac"]}', fh)
    log(f'K分布 正样本: {summary["pos_k_dist"]}  负样本: {summary["neg_k_dist"]}', fh)
    for row in summary['tau_table']:
        log(f'  τ={row["tau"]:<4} TP={row["TP"]:<4} FP={row["FP"]:<4} FN={row["FN"]:<4} '
            f'TN={row["TN"]:<4} TPR={row["TPR"]*100:6.1f}% FPR={row["FPR"]*100:6.1f}%', fh)
    log(f'\n✅ 结果已保存: {json_path}', fh)
    log(f'   日志已保存: {log_path}', fh)
    log(f'   latest 指针: {os.path.join(args.out_dir, "latest.json")}', fh)
    fh.close()


if __name__ == '__main__':
    main()
