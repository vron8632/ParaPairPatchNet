#!/usr/bin/env python3
"""
SAM 重分割 APRICOT 零样本实验 — ForensicGuard
======================================================
目的: 用 SAM（更强的通用分割器）重新分割 APRICOT 真实物理补丁图像，
      验证 K>=2（跨实例一致性激活）场景下 ForensicGuard 的表现，
      针对"APRICOT 全 K=0 导致一致性未验证"的质疑。

用法:
    python3 code/run_sam_apricot.py
    （首次运行会自动下载 SAM-B 权重 ~90MB，需联网；可提前下载放到
      ~/weights/sam_b.pt 或当前目录）

输出:
    results/run_sam_apricot_sam_<版本号>_<时间戳>.json — 完整结果, 如 run_sam_apricot_sam_v1_20260801_003000.json
    results/logs/run_sam_apricot_sam_<版本号>_<时间戳>.log — 训练/评估日志（含进度、ETA、剩余时分秒）
    results/run_sam_apricot_latest.json / run_sam_apricot_latest.log — 快捷指针(非正式结果)

依赖: ultralytics>=8.3（自带 SAM），首次自动下载 sam_b.pt
"""
import sys, os, time, json, zipfile, torch, numpy as np
from PIL import Image
from torchvision import transforms

BASE = '/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard'
sys.path.insert(0, os.path.join(BASE, 'code'))
RESULT_DIR = os.path.join(BASE, 'results')
LOG_DIR = os.path.join(BASE, 'results/logs')
ZIP_PATH = os.path.join(BASE, 'data/apricot/dev.zip')
SCRIPT = os.path.splitext(os.path.basename(__file__))[0]  # run_sam_apricot

# 输出文件名: <脚本名>_<主标记>_<版本>_<时间戳>（由 main 解析 --tag 后设置）; *_latest.* 为快捷指针
JSON_PATH = None
LOG_PATH = None

from models.consistency import CrossInstanceConsistencyV2
from models.fusion_classifier import FusionClassifierV2
from scripts.train import compute_fused_features
from features.spn_extractor import get_spn_extractor
from features.dct_profile import DCTProfileExtractor
from features.clip_encoder import get_clip_extractor

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def log(msg):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')


def fmt_eta(secs):
    secs = int(max(0, secs)); h, r = divmod(secs, 3600); m, s = divmod(r, 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


def get_sam_model():
    """加载 SAM, 支持预下载权重"""
    cands = ['sam_b.pt', os.path.expanduser('~/weights/sam_b.pt'),
             f'{BASE}/weights/sam_b.pt']
    for c in cands:
        if os.path.exists(c):
            log(f'SAM 权重: {c}')
            return __import__('ultralytics').SAM(c)
    log('未找到本地 SAM 权重, 尝试自动下载 sam_b.pt (约 90MB)...')
    return __import__('ultralytics').SAM('sam_b.pt')


def main():
    import argparse
    ap = argparse.ArgumentParser(description='SAM 重分割 APRICOT 零样本实验 (版本化输出)')
    ap.add_argument('--tag', default='v1',
                    help='实验版本标签(如 v1/v2/...), 用于区分不同配置; 会进入文件名')
    ap.add_argument('--out_dir', default=RESULT_DIR)
    args = ap.parse_args()

    global JSON_PATH, LOG_PATH
    ts = time.strftime('%Y%m%d_%H%M%S')
    # 正式结果命名: <脚本名>_<实验主标记=sam>_<版本号>_<时间戳>
    JSON_PATH = os.path.join(args.out_dir, f'{SCRIPT}_sam_{args.tag}_{ts}.json')
    LOG_PATH = os.path.join(LOG_DIR, f'{SCRIPT}_sam_{args.tag}_{ts}.log')
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    open(LOG_PATH, 'w').close()
    log(f'输出: {JSON_PATH}')
    log('=== SAM 重分割 APRICOT 零样本实验 ===')
    log(f'device={device}')

    # 加载模型 + 特征提取器
    cic = CrossInstanceConsistencyV2()
    model = FusionClassifierV2(input_dim=698)
    model.load_state_dict(torch.load(
        os.path.join(BASE, 'results/models/digital_baseline_v2/fusion_best.pth'),
        map_location='cpu', weights_only=True))
    model = model.to(device).eval()
    spn = get_spn_extractor(mode='learned', device=device)
    dct = DCTProfileExtractor()
    clip_img = get_clip_extractor(device=device)
    sam = get_sam_model()

    # 处理 zip
    results = []
    t0 = time.time()
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        pt_files = sorted([f for f in z.namelist() if f.endswith('.pt')])
        log(f'APRICOT 图像数: {len(pt_files)}')
        for idx, pt_name in enumerate(pt_files):
            with z.open(pt_name) as f:
                data = torch.load(f, map_location='cpu', weights_only=False)
            img_np = data['Image'][0]           # (H, W, 3)
            patch_mask = data['Mask']            # (H, W) bool, 补丁真值
            has_patch = bool(patch_mask.any())
            label = 1 if has_patch else 0

            img_pil = Image.fromarray(img_np.astype(np.uint8))
            img_512 = transform(img_pil)
            img_orig = img_512 * std + mean     # [0,1] 反归一化

            # SAM 自动分割: 传 uint8 numpy (HWC), 让 ultralytics 自动加 batch/resize
            img_sam = (img_orig.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            res = sam(img_sam, verbose=False)
            if res is None or len(res) == 0 or res[0].masks is None:
                masks = []
            else:
                mdata = res[0].masks.data.cpu().numpy()  # (N, H_sam, W_sam)
                # 缩回 512x512 (与特征提取的 img 尺寸一致)
                masks = []
                for m in mdata:
                    m_pil = Image.fromarray((m * 255).astype(np.uint8)).resize((512, 512))
                    m512 = np.array(m_pil) > 0
                    if m512.sum() > 64 * 64:
                        masks.append(m512)

            K = len(masks)
            # 特征提取
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

            # 评估（统一设备: spn 在 CUDA, dct/clip 在 CPU → 全部转 device）
            spn_f, dct_f, clip_f = spn_f.to(device), dct_f.to(device), clip_f.to(device)
            with torch.no_grad():
                fused = compute_fused_features([spn_f], [dct_f], [clip_f], cic, device)
                prob = torch.sigmoid(model(fused)).item()
            pred = 1 if prob > 0.5 else 0
            correct = int(pred == label)

            results.append({'file': pt_name, 'K': int(K), 'label': label,
                            'pred': pred, 'prob': round(float(prob), 4),
                            'correct': correct})

            if (idx + 1) % 10 == 0 or idx + 1 == len(pt_files):
                el = time.time() - t0
                eta = fmt_eta(el / (idx + 1) * (len(pt_files) - idx - 1))
                done = sum(1 for r in results if r['correct'])
                log(f'  [{idx+1}/{len(pt_files)}] acc={done}/{idx+1} '
                    f'K均值={np.mean([r["K"] for r in results]):.2f} '
                    f'ETA={eta} elapsed={fmt_eta(el)}')
                # 累积保存（防中断丢失）
                save(results)

    # 汇总统计
    save(results)
    n = len(results)
    acc = sum(r['correct'] for r in results) / n * 100
    k_dist = {}
    for r in results:
        k_dist.setdefault(r['K'], [0, 0])
        k_dist[r['K']][0] += r['correct']; k_dist[r['K']][1] += 1

    log(f'\n=== 结果 ===')
    log(f'整体: {sum(r["correct"] for r in results)}/{n} = {acc:.1f}%')
    kge2_c = sum(v[0] for k, v in k_dist.items() if k >= 2)
    kge2_n = sum(v[1] for k, v in k_dist.items() if k >= 2)
    log(f'K分布: ' + ', '.join(f'K={k}: {v[1]}张(acc={v[0]/v[1]*100:.0f}%)'
                               for k, v in sorted(k_dist.items())))
    log(f'K>=2: {kge2_c}/{kge2_n} = {kge2_c/kge2_n*100:.1f}% (一致性激活时)' if kge2_n else 'K>=2: 0 张')
    log(f'\n✅ 结果已保存: {JSON_PATH}')

    # latest 快捷指针（带脚本名前缀, 只作查找用, 非正式结果）
    for suffix in ('json', 'log'):
        src = JSON_PATH if suffix == 'json' else LOG_PATH
        dst = os.path.join(os.path.dirname(src), f'{SCRIPT}_latest.{suffix}')
        try:
            with open(src, 'rb') as fi, open(dst, 'wb') as fo:
                fo.write(fi.read())
        except OSError:
            pass
    log(f'latest 指针: {os.path.join(os.path.dirname(JSON_PATH), f"{SCRIPT}_latest.json")}')


def save(results):
    """保存结果到 JSON（含中间态）"""
    payload = {
        'experiment': 'SAM re-segmentation of APRICOT (zero-shot)',
        'model': 'digital_baseline_v2/fusion_best.pth',
        'segmenter': 'SAM (sam_b.pt, auto mode)',
        'n_total': len(results),
        'accuracy_pct': round(sum(r['correct'] for r in results) / max(1, len(results)) * 100, 1),
        'results': results,
    }
    with open(JSON_PATH, 'w') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
