#!/bin/bash
# 对抗训练 + 双评估（clean / 白盒 PGD）
# 目的：压低下 adv_attack_pgd.json 暴露的 ASR 42%，并验证精度代价
set -u
PY=/home/oyp/miniconda3/envs/apjf/bin/python
cd "$(dirname "$0")/../.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1

echo "########## START $(date '+%F %T') ##########"

echo "########## [1/3] 对抗训练 (FGSM, eps=0.03, constraint=workpoint, bs=2) ##########"
$PY code/experiments_tvc/lora_mllm.py --stage train --tag adv_robust \
    --bs 2 --adv-train 0.03 --adv-steps 1 --constraint-mode workpoint

echo "########## [2/3] clean 评估 (val 1000) ##########"
$PY code/experiments_tvc/lora_mllm.py --stage eval --tag adv_robust

echo "########## [3/3] 白盒 PGD 对抗评估 ##########"
$PY code/experiments_tvc/adv_attack_eval.py --ckpt results/tvc/adv_robust \
    --n 50 --eps 0.03 --steps 10 --out results/tvc/adv_attack_pgd_robust.json

echo "########## ALL DONE $(date '+%F %T') ##########"
