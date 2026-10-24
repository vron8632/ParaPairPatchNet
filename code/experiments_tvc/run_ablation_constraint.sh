#!/bin/bash
# 串行批处理：FPR 约束项消融（理论对齐）+ 负样本规模扩容到 n=1000
# 用法: bash code/experiments_tvc/run_ablation_constraint.sh > /tmp/ablation_main.log 2>&1
# 依赖: lora_mllm.py --constraint-mode {both,mean,workpoint,none}
#       eval_neg_scale.py
set -u
PY=/home/oyp/miniconda3/envs/apjf/bin/python
cd "$(dirname "$0")/../.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1

echo "########## START $(date '+%F %T') ##########"

echo "########## [1/5] train: mode=mean (仅 (i) 均值约束) ##########"
$PY code/experiments_tvc/lora_mllm.py --stage train --tag ablate_mean \
    --lam 1.0 --constraint-mode mean
echo "########## [2/5] eval: ablate_mean ##########"
$PY code/experiments_tvc/lora_mllm.py --stage eval --tag ablate_mean

echo "########## [3/5] train: mode=workpoint (仅 (ii) 工作点惩罚) ##########"
$PY code/experiments_tvc/lora_mllm.py --stage train --tag ablate_workpoint \
    --lam 1.0 --constraint-mode workpoint
echo "########## [4/5] eval: ablate_workpoint ##########"
$PY code/experiments_tvc/lora_mllm.py --stage eval --tag ablate_workpoint

echo "########## [5/5] 负样本扩容 n=1000 (主模型, 4 源 x 250) ##########"
$PY code/experiments_tvc/eval_neg_scale.py \
    --ckpt results/tvc/lora_patch_detector \
    --n-per-source 250 \
    --pos-json results/tvc/confound_main_withscores.json \
    --out results/tvc/confound_scale1k.json

echo "########## ALL DONE $(date '+%F %T') ##########"
