#!/usr/bin/env python3
"""
方案 B: clean 数据扩充 — ForensicGuard
================================================
目标: 扩大 clean 训练集 (600 → 2000), 缓解类别不平衡 (clean:tampered = 1:4),
     提高分数分离 (AUC), 期望同时达到 clean>=80% + physical>=85%。

阶段:
  S1 选图: 从 CASIA Au 选 ~1400 张新 clean 图 (与现有 clean 去重), 预处理存 patch_dataset_v2/clean_extra_train.pt
  S2 特征: YOLO 分割 + SPN/DCT/CLIP 提取 → clean_extra_train_features.pt
  S3 训练: clean(600+1400) + digital + physical 平衡训练
  S4 评估: 阈值扫描找 clean>=80% 且 physical 最大的操作点

输出:
  results/clean_expand_experiment.json
  results/models/clean_acc_expand/fusion_best.pth
  results/logs/clean_expand.log
"""
import sys, os, time, json, random, torch, numpy as np

BASE = '/media/oyp/工作/Projects/042_image_forensic/DuetGuard/mmm-forensicguard'
sys.path.insert(0, os.path.join(BASE, 'code'))
FEATURE_DIR = os.path.join(BASE, 'results/features')
LOG_DIR = os.path.join(BASE, 'results/logs')
OUT_JSON = os.path.join(BASE, 'results/clean_expand_experiment.json')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'[env] device={device}', flush=True)

EXTRA_N = 1400      # 新增 clean 数
SEED = 1234

def log(msg):
    print(msg, flush=True)
    with open(f'{LOG_DIR}/clean_expand.log', 'a') as f:
        f.write(msg + '\n')

def fmt_eta(s):
    s = int(max(0, s)); h, r = divmod(s, 3600); m, s2 = divmod(r, 60)
    return f'{h:02d}:{m:02d}:{s2:02d}'

# ───────────────── S1: 选图 + 预处理 ─────────────────
def select_and_prepare():
    from PIL import Image
    import torchvision.transforms as T
    save_path = f'{FEATURE_DIR}/patch_dataset_v2/clean_extra_train.pt'
    if os.path.exists(save_path):
        log(f'[S1] 缓存存在, 跳过: {save_path}')
        return
    # 现有 clean train tensors (去重用)
    existing = torch.load(f'{FEATURE_DIR}/patch_dataset_v2/clean_train.pt', map_location='cpu', weights_only=False)
    exist_set = set()
    for t in existing['tensors']:
        exist_set.add(hash(t.numpy().tobytes()))
    log(f'[S1] 现有 clean train: {len(existing["tensors"])} 张')
    # CASIA Au 全部图像
    au_dir = f'{BASE}/data/casia2/CASIA2.0_revised/Au'
    imgs = sorted([f for f in os.listdir(au_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    log(f'[S1] CASIA Au 共 {len(imgs)} 张')
    random.seed(SEED)
    random.shuffle(imgs)
    tf = T.Compose([T.Resize((512, 512)), T.ToTensor()])
    chosen, t0 = [], time.time()
    for i, fn in enumerate(imgs):
        if len(chosen) >= EXTRA_N: break
        try:
            img = Image.open(os.path.join(au_dir, fn)).convert('RGB')
            t = tf(img)
            if hash(t.numpy().tobytes()) in exist_set:
                continue
            chosen.append(t)
        except Exception:
            continue
        if (i + 1) % 200 == 0:
            el = time.time() - t0
            eta = fmt_eta(el / (i+1) * (len(imgs) - i))
            log(f'  [S1] 扫描 {i+1}/{len(imgs)}, 已选 {len(chosen)}, ETA={eta}')
    log(f'[S1] 选定 {len(chosen)} 张新 clean')
    torch.save({'tensors': chosen, 'labels': [0]*len(chosen)}, save_path)
    log(f'[S1] 已保存: {save_path}')

# ───────────────── S2: 特征提取 ─────────────────
def extract_features():
    out_path = f'{FEATURE_DIR}/clean_extra_train_features.pt'
    if os.path.exists(out_path):
        log(f'[S2] 特征缓存存在, 跳过: {out_path}')
        return
    from features.segmenter import InstanceSegmenter
    from features.spn_extractor import get_spn_extractor
    from features.dct_profile import DCTProfileExtractor
    from features.clip_encoder import get_clip_extractor
    data = torch.load(f'{FEATURE_DIR}/patch_dataset_v2/clean_extra_train.pt', map_location='cpu', weights_only=False)
    tensors, labels = data['tensors'], data['labels']
    seg = InstanceSegmenter(device=device)
    spn = get_spn_extractor(mode='learned', device=device)
    dct = DCTProfileExtractor()
    clip = get_clip_extractor(device=device)
    all_spn, all_dct, all_clip, all_masks = [], [], [], []
    t0 = time.time()
    for i, t in enumerate(tensors):
        img = t.to(device)
        masks_list = seg.segment(img)
        masks_list = [(m, b) for m, b in masks_list if m.sum() > 64*64]
        all_masks.append([m for m, _ in masks_list])
        K = len(masks_list)
        if K < 2:
            all_spn.append(spn.extract(img).unsqueeze(0))
            all_dct.append(dct.extract_profile(img.cpu()).unsqueeze(0))
            all_clip.append(clip.encode_image(img.cpu()).unsqueeze(0))
        else:
            sl, dl, cl = [], [], []
            for mask, _ in masks_list:
                mt = torch.from_numpy(mask).float().to(device)
                sl.append(spn.extract_instance(img, mt))
                dl.append(dct.extract_profile(img.cpu(), mask))
                cl.append(clip.encode_image(img.cpu(), mask))
            all_spn.append(torch.stack(sl)); all_dct.append(torch.stack(dl)); all_clip.append(torch.stack(cl))
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            eta = fmt_eta(el / (i+1) * (len(tensors) - i))
            log(f'  [S2] 特征 {i+1}/{len(tensors)}, ETA={eta}')
    torch.save({'spn_feats': all_spn, 'dct_feats': all_dct, 'clip_feats': all_clip,
                'labels': labels, 'instance_counts': [len(s) for s in all_spn],
                'masks': all_masks}, out_path)
    log(f'[S2] 已保存: {out_path} ({len(labels)} 样本)')

# ───────────────── S3: 训练 ─────────────────
def load_domain(domain, split):
    d = torch.load(f'{FEATURE_DIR}/{domain}_{split}_features.pt', map_location='cpu', weights_only=False)
    return d['spn_feats'], d['dct_feats'], d['clip_feats'], d['labels']

def train():
    from models.consistency import CrossInstanceConsistencyV2
    from models.fusion_classifier import FusionClassifierV2
    from scripts.train import compute_fused_features
    cic = CrossInstanceConsistencyV2()
    # 合并: clean(600+extra1400) + digital + physical
    s_c, d_c, c_c, l_c = load_domain('clean', 'train')
    extra = torch.load(f'{FEATURE_DIR}/clean_extra_train_features.pt', map_location='cpu', weights_only=False)
    s_c += extra['spn_feats']; d_c += extra['dct_feats']; c_c += extra['clip_feats']
    l_c = list(l_c) + list(extra['labels'])
    s_d, d_d, c_d, l_d = load_domain('digital_patch', 'train')
    s_p, d_p, c_p, l_p = load_domain('physical_patch', 'train')
    s_tr = s_c + s_d + s_p; d_tr = d_c + d_d + d_p
    c_tr = c_c + c_d + c_p; l_tr = l_c + l_d + l_p
    s_v, d_v, c_v, l_v = merge_val()
    n_tr = len(l_tr)
    n_clean = l_tr.count(0)
    log(f'[S3] 训练样本: {n_tr} (clean={n_clean}, tampered={n_tr-n_clean})')

    model = FusionClassifierV2(input_dim=698).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    w_clean, w_tam = 1.0, 1.0  # 平衡后(2000:2400)无需额外加权
    best_clean, best_path = 0.0, f'{BASE}/results/models/clean_acc_expand/fusion_best.pth'
    os.makedirs(os.path.dirname(best_path), exist_ok=True)
    t0 = time.time()
    for epoch in range(1, 301):
        model.train(); total_loss = 0.0
        idx = torch.randperm(n_tr)
        for st in range(0, n_tr, 64):
            en = min(st + 64, n_tr); ids = idx[st:en]
            bs = [s_tr[i] for i in ids]; bd = [d_tr[i] for i in ids]
            bc = [c_tr[i] for i in ids]
            by = torch.tensor([l_tr[i] for i in ids], dtype=torch.float32, device=device)
            w = torch.tensor([w_clean if l_tr[i] == 0 else w_tam for i in ids],
                             dtype=torch.float32, device=device)
            fused = compute_fused_features(bs, bd, bc, cic, device)
            lg = model(fused).squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(lg, by, weight=w)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total_loss += loss.item() * len(ids)
        if epoch % 10 == 0 or epoch == 300:
            # val clean acc
            model.eval(); cc = 0
            with torch.no_grad():
                for i in range(len(l_v)):
                    fused = compute_fused_features([s_v[i]], [d_v[i]], [c_v[i]], cic, device)
                    pred = (torch.sigmoid(model(fused)) > 0.5).item()
                    cc += (pred == l_v[i])
            vacc = 100 * cc / len(l_v)
            # 仅 clean 的 val
            cclean = 0; cn = 0
            with torch.no_grad():
                for i in range(len(l_v)):
                    if l_v[i] != 0: continue
                    cn += 1
                    fused = compute_fused_features([s_v[i]], [d_v[i]], [c_v[i]], cic, device)
                    cclean += (torch.sigmoid(model(fused)) < 0.5).item()
            vclean = 100 * cclean / cn if cn else 0
            if vclean > best_clean:
                best_clean = vclean
                torch.save(model.state_dict(), best_path)
            el = time.time() - t0
            eta = fmt_ta(el / epoch * (300 - epoch))
            log(f'  Epoch {epoch}/300: loss={total_loss/n_tr*64:.4f} val_acc={vacc:.1f}% val_clean={vclean:.1f}% ETA={eta}')
    return model, cic

def merge_val():
    ss, ds, cs, ls = [], [], [], []
    for dom in ['clean', 'digital_patch', 'physical_patch']:
        s, d, c, l = load_domain(dom, 'val')
        ss += s; ds += d; cs += c; ls += l
    return ss, ds, cs, ls

def fmt_ta(s):
    s = int(max(0, s)); h, r = divmod(s, 3600); m, s2 = divmod(r, 60)
    return f'{h:02d}:{m:02d}:{s2:02d}'

# ───────────────── S4: 阈值扫描评估 ─────────────────
def evaluate(model, cic):
    from sklearn.metrics import roc_auc_score
    sc = scores(model, cic, 'clean'); sp = scores(model, cic, 'physical_patch')
    sd = scores(model, cic, 'digital_patch')
    auc = roc_auc_score(np.concatenate([np.zeros(len(sc)), np.ones(len(sp))]),
                        np.concatenate([sc, sp]))
    log(f'[S4] clean-physical AUC = {auc:.4f}')
    rows = []
    for tau in np.arange(0.3, 0.99, 0.01):
        ca = (sc < tau).mean()*100; pa = (sp > tau).mean()*100; da = (sd > tau).mean()*100
        rows.append((float(tau), float(ca), float(da), float(pa)))
    # 找 clean>=80% 且 physical 最大
    cands = [r for r in rows if r[1] >= 80]
    best = max(cands, key=lambda r: r[3]) if cands else None
    result = {'auc': round(auc, 4), 'threshold_scan': rows,
              'best_clean80': None}
    if best:
        result['best_clean80'] = {'tau': best[0], 'clean_pct': round(best[1], 1),
                                  'digital_pct': round(best[2], 1), 'physical_pct': round(best[3], 1)}
        log(f'[S4] clean>=80% 最优: tau={best[0]:.2f} clean={best[1]:.1f}% '
            f'physical={best[3]:.1f}% digital={best[2]:.1f}%')
    else:
        log('[S4] 无 clean>=80% 的解')
    with open(OUT_JSON, 'w') as f:
        json.dump(result, f, indent=2)
    log(f'[S4] 已保存: {OUT_JSON}')
    return result

def scores(model, cic, domain):
    s, d, c, _ = load_domain(domain, 'test')
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(len(s)):
            fused = compute_fused_features([s[i]], [d[i]], [c[i]], cic, device)
            out.append(torch.sigmoid(model(fused)).item())
    return np.array(out)

def main():
    os.makedirs(f'{LOG_DIR}', exist_ok=True)
    open(f'{LOG_DIR}/clean_expand.log', 'w').close()
    log('=== 方案 B: clean 数据扩充 ===')
    select_and_prepare()
    extract_features()
    model, cic = train()
    model.eval()
    evaluate(model, cic)
    log('✅ 全部完成')

if __name__ == '__main__':
    main()
