#!/usr/bin/env python3
"""
实验产出完整性检查
=======================================
核对所有预期输出文件是否已生成，报告完成进度与缺失项。
用法: $PY code/experiments_tvc/verify_outputs.py
"""
import os, json

WS = "/media/oyp/工作/Projects/042_image_forensic/tvc-forensicguard"
TVC = f"{WS}/results/tvc"

# (显示名, 文件路径, 所属队列)
EXPECTED = [
    # REV1 分解消融补 seed
    ("分解消融 mean seed1", f"{TVC}/lora_mllm_eval_abl_mean_v2_s1.json", "REV1"),
    ("分解消融 mean seed2", f"{TVC}/lora_mllm_eval_abl_mean_v2_s2.json", "REV1"),
    ("分解消融 wp seed1", f"{TVC}/lora_mllm_eval_abl_wp_soft_s1.json", "REV1"),
    ("分解消融 wp seed2", f"{TVC}/lora_mllm_eval_abl_wp_soft_s2.json", "REV1"),
    ("数据规模 25%", f"{TVC}/lora_mllm_eval_scale_025.json", "REV1"),
    ("数据规模 50%", f"{TVC}/lora_mllm_eval_scale_05.json", "REV1"),
    # REV2 主表 v2
    ("主表 v2（双口径×3 seed）", f"{TVC}/main_table_v2.json", "REV2"),
    # REV3
    ("CLIP@512", f"{TVC}/clip_linear_512.json", "REV3"),
    ("Prompt 鲁棒性", f"{TVC}/prompt_robustness.json", "REV3"),
    ("超参 β=0.05", f"{TVC}/lora_mllm_eval_hp_beta0.05.json", "REV3"),
    ("超参 β=0.30", f"{TVC}/lora_mllm_eval_hp_beta0.30.json", "REV3"),
    ("超参 λ=0.5", f"{TVC}/lora_mllm_eval_hp_lam0.5.json", "REV3"),
    ("超参 λ=2.0", f"{TVC}/lora_mllm_eval_hp_lam2.0.json", "REV3"),
    ("超参 τ0=0.3", f"{TVC}/lora_mllm_eval_hp_tau0.3.json", "REV3"),
    ("超参 τ0=0.7", f"{TVC}/lora_mllm_eval_hp_tau0.7.json", "REV3"),
    ("数据扩充 3.3k 重训", f"{TVC}/lora_mllm_eval_scale_3k.json", "REV3"),
    ("低饱和负样本增强重训", f"{TVC}/lora_mllm_eval_negaug.json", "REV3"),
    # REV4
    ("鲁棒性强化 v2", f"{TVC}/adv_attack_v2.json", "REV4"),
    # REV5
    ("黑盒迁移攻击", f"{TVC}/blackbox_transfer.json", "REV5"),
    ("粗定位（遮挡法）", f"{TVC}/patch_localization.json", "REV5"),
    # 统一协议（用户决策：1024 + 1000 负样本）
    ("统一协议主表（3 seed + CLIP @1024）", f"{TVC}/unified_main_1024.json", "REV7"),
    ("统一协议基线（NutNet+轻量）", f"{TVC}/unified_baselines.json", "REV8"),
    ("统一协议 TruFor", f"{TVC}/trufor_unified.json", "REV9"),
    ("统一协议 PSCC-Net", f"{TVC}/pscc_unified.json", "REV9"),
    ("统一评测集（138 正 + 1000 负）", f"{TVC}/unified_eval_set", "已完成"),
    # 已完成的零成本分析
    ("CI/阈值策略/APDE 拆表", f"{TVC}/review_analyses.json", "已完成"),
    ("训练数据 3.3k 索引", f"{TVC}/lora_3k/index.json", "已完成"),
    ("低饱和增强索引", f"{TVC}/lora_lowsat/index.json", "已完成"),
    ("APRICOT 138 正样本", f"{TVC}/baseline_imgs/pos_all", "已完成"),
]

done = missing = 0
by_q = {}
for name, p, q in EXPECTED:
    ok = os.path.exists(p)
    by_q.setdefault(q, [0, 0])
    by_q[q][0 if ok else 1] += 1
    done += ok; missing += (not ok)

print(f"{'状态':<4} {'队列':<6} {'项':<28} {'详情'}")
print("-" * 78)
for name, p, q in EXPECTED:
    ok = os.path.exists(p)
    detail = ""
    if ok and p.endswith(".json") and os.path.isfile(p):
        try:
            d = json.load(open(p))
            if "auc" in d: detail = f"AUC={d['auc']:.4f}"
            elif "table" in d: detail = f"{len(d['table'])} 模型"
            elif "tpr" in d: detail = "多口径 TPR"
            elif "tpr_clean" in d: detail = f"ASR={d.get('asr_transfer') or d.get('asr')}"
            elif "hit_rate" in d: detail = f"命中率={d['hit_rate']}"
            elif "apde_by_attack" in d: detail = "APDE 拆表"
            elif "TPR@5pct" in d or "3src" in d: detail = "已生成"
        except Exception: detail = "（读取失败）"
    elif ok: detail = f"{len(os.listdir(p))} 文件" if os.path.isdir(p) else ""
    print(f"{'✅' if ok else '⏳':<4} {q:<6} {name:<28} {detail}")
print("-" * 78)
print(f"完成 {done}/{len(EXPECTED)}　缺失 {missing}")
print("\n按队列汇总:")
for q in ("已完成", "REV1", "REV2", "REV3", "REV4", "REV5", "REV6", "REV7", "REV8", "REV9"):
    if q in by_q:
        d, m = by_q[q]
        print(f"  {q}: {d}/{d+m} {'✅ 全部完成' if m == 0 else f'（缺 {m}）'}")
print("\n队列日志检查（/tmp）:")
for q in ("rev1", "rev2", "rev3", "rev4", "rev5", "rev6", "rev7", "rev8", "rev9"):
    log = f"/tmp/queue_{q}.log"
    if os.path.exists(log):
        st = "✅ 完成" if "ALL DONE" in open(log, errors="ignore").read() else "🔄 进行/排队"
        print(f"  {q}: {st}")
    else:
        print(f"  {q}: 无日志")
