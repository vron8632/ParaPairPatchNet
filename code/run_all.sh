#!/bin/bash
# ============================================================
# ForensicGuard — 一键实验运行脚本 v4
#
# 终端友好版: 进度条 + 剩余时间 + 结构化结果存储
#
# 使用方法:
#   bash code/run_all.sh                  # 完整运行全部8个阶段
#   bash code/run_all.sh --stage 3        # 从阶段3开始(清除3~8标记)
#   bash code/run_all.sh --force-stage 3  # 只重跑阶段3(保留4~8结果)
#   bash code/run_all.sh --stage 3 --force-stage 5  # 从3开始,重跑5
#   bash code/run_all.sh --list           # 查看各阶段耗时估算
#   bash code/run_all.sh --status         # 查看各阶段完成状态和结果
# ============================================================

set -e

# 强制Python输出不缓冲(解决tee重定向时的输出卡顿)
export PYTHONUNBUFFERED=1

# ============================================================
# 配置
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_DIR/data"
FEATURE_DIR="$PROJECT_DIR/results/features"
MODEL_DIR="$PROJECT_DIR/results/models"
RESULT_DIR="$PROJECT_DIR/results"
LOG_DIR="$RESULT_DIR/logs"
SUMMARY_FILE="$RESULT_DIR/all_results.txt"
STAGE_RESULTS_FILE="$RESULT_DIR/stage_results.json"

mkdir -p "$FEATURE_DIR" "$MODEL_DIR" "$LOG_DIR"

# ============================================================
# 彩色输出 & 计时
# ============================================================
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

TOTAL_STAGES=10
GLOBAL_START_SEC=$SECONDS

# ---- 阶段耗时估算表 (基于GPU: RTX 4090) ----
# 格式: stage_index|stage_name|estimated_minutes
STAGE_TIMES=(
    "1|生成v2补丁数据集|15"
    "2|特征提取(patch数据集)|120"
    "3|数字域基线训练|8"
    "4|物理域衰减测试|3"
    "5|增强训练(含物理域)|8"
    "6|经典篡改检测(3个数据集)|45"
    "7|消融实验(4种配置)|30"
    "8|结果汇总|2"
    "9|SOTA Baseline对比(ResNet+全图特征)|20"
    "10|重提CASIA特征(v2)+FocalLoss训练|60"
)

# ---- 函数 ----
log()     { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $1"; }
warn()    { echo -e "${YELLOW}[警告]${NC} $1"; }
error()   { echo -e "${RED}[错误]${NC} $1"; }
header()  {
    echo -e "\n${BLUE}══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  █ $1${NC}"
    echo -e "${BLUE}══════════════════════════════════════════════════════════════${NC}"
}
bold()    { echo -e "${BOLD}$1${NC}"; }

# ---- 显示剩余时间 ----
format_time() {
    local total_seconds=$1
    if [ $total_seconds -lt 0 ]; then total_seconds=0; fi
    local hours=$((total_seconds / 3600))
    local minutes=$(( (total_seconds % 3600) / 60 ))
    local secs=$((total_seconds % 60))
    printf "%02d:%02d:%02d" $hours $minutes $secs
}

show_progress() {
    local current=$1
    local total=$2
    local elapsed=$3
    local pct=$(( current * 100 / total ))

    # 计算剩余时间
    if [ $current -gt 0 ] && [ $elapsed -gt 0 ]; then
        local estimated_total=$(( elapsed * total / current ))
        local remaining=$(( estimated_total - elapsed ))
        local remaining_str=$(format_time $remaining)
    else
        local remaining_str="--:--:--"
    fi

    local elapsed_str=$(format_time $elapsed)

    # 进度条
    local bar_width=40
    local filled=$(( pct * bar_width / 100 ))
    local empty=$(( bar_width - filled ))

    printf "\n${CYAN}━━━ 进度 ━━━${NC}\n"
    printf "  阶段: ${BOLD}%d/%d${NC}  (共 %d 个阶段)\n" $current $total $total
    printf "  进度: ["
    for ((i=0; i<filled; i++)); do printf "█"; done
    for ((i=0; i<empty; i++)); do printf "░"; done
    printf "] ${BOLD}%3d%%${NC}\n" $pct
    printf "  已用: ${BOLD}%s${NC}  剩余: ${BOLD}%s${NC}  总预计: ~%s\n" \
           "$elapsed_str" "$remaining_str" "$(format_time $((elapsed + remaining)))"
    echo ""
}

# ---- 显示下阶段预告 ----
show_next_stage() {
    local next=$1
    local total=$2

    for entry in "${STAGE_TIMES[@]}"; do
        local idx="${entry%%|*}"
        local rest="${entry#*|}"
        local name="${rest%%|*}"
        local est="${rest##*|}"

        if [ "$idx" = "$next" ]; then
            local remaining_min=0
            for e in "${STAGE_TIMES[@]}"; do
                local ei="${e%%|*}"
                local er="${e##*|}"
                if [ "$ei" -ge "$next" ] 2>/dev/null; then
                    remaining_min=$((remaining_min + er))
                fi
            done
            echo -e "\n${YELLOW}⏭  下一阶段: ${BOLD}${name}${NC}"
            echo -e "${YELLOW}  预计耗时: ~${est} 分钟${NC}"
            echo -e "${YELLOW}  剩余阶段: ~${remaining_min} 分钟${NC}"
            break
        fi
    done
}

get_stage_est_time() {
    local target=$1
    for entry in "${STAGE_TIMES[@]}"; do
        local idx="${entry%%|*}"
        if [ "$idx" = "$target" ]; then
            echo "${entry##*|}"
            return
        fi
    done
    echo "?"
}

get_stage_name() {
    local target=$1
    for entry in "${STAGE_TIMES[@]}"; do
        local idx="${entry%%|*}"
        if [ "$idx" = "$target" ]; then
            local rest="${entry#*|}"
            echo "${rest%%|*}"
            return
        fi
    done
    echo "未知"
}

# ---- 检查点 ----
run_stage() {
    local marker="$RESULT_DIR/.stage${1}_done"
    if [ -f "$marker" ]; then
        return 0  # 已存在
    fi
    return 1
}

mark_stage_done() {
    touch "$RESULT_DIR/.stage${1}_done"
}

clear_stage_marker() {
    local stage=$1
    for ((s=stage; s<=TOTAL_STAGES; s++)); do
        rm -f "$RESULT_DIR/.stage${s}_done"
    done
}

# ---- 结构化结果存储 ----
save_stage_result() {
    local stage=$1
    local status=$2      # completed / skipped / failed
    local elapsed=$3
    local summary="$4"   # JSON片段: {"acc": 94.5, "f1": 93.2}

    local name=$(get_stage_name $stage)

    # 读取已有结果
    local tmp_file="/tmp/stage_results_$$.json"
    if [ -f "$STAGE_RESULTS_FILE" ]; then
        cat "$STAGE_RESULTS_FILE" > "$tmp_file"
    else
        echo "{}" > "$tmp_file"
    fi

    # 用 Python 写入结构化 JSON
    python3 -c "
import json, sys
with open('$tmp_file', 'r') as f:
    data = json.load(f)

stage_key = 'stage_$stage'
data[stage_key] = {
    'stage': $stage,
    'name': '$name',
    'status': '$status',
    'elapsed_seconds': $elapsed,
    'elapsed_str': '$(format_time $elapsed)',
    'timestamp': '$(date +%Y-%m-%d_%H:%M:%S)',
    'summary': $summary
}

with open('$tmp_file', 'w') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
print(f'[结果] 阶段{$stage} 已保存 (elapsed={$elapsed}s)')
"

    mv "$tmp_file" "$STAGE_RESULTS_FILE"
}

show_stage_status() {
    echo "============================================"
    echo " ForensicGuard — 阶段状态"
    echo "============================================"
    for ((s=1; s<=TOTAL_STAGES; s++)); do
        local name=$(get_stage_name $s)
        local marker="$RESULT_DIR/.stage${s}_done"
        if [ -f "$marker" ]; then
            local result=""
            if [ -f "$STAGE_RESULTS_FILE" ]; then
                result=$(python3 -c "
import json
try:
    with open('$STAGE_RESULTS_FILE') as f:
        data = json.load(f)
    s = data.get('stage_$s', {})
    summary = s.get('summary', {})
    parts = []
    for k in ['acc','f1','accuracy']:
        if k in summary:
            parts.append(f'{k}={summary[k]}')
    print(' | '.join(parts) if parts else '')
except: pass
" 2>/dev/null)
            fi
            printf "  ✅ 阶段 %d/9: %-25s %s\n" "$s" "$name" "$result"
        else
            printf "  ⏳ 阶段 %d/9: %-25s (待运行)\n" "$s" "$name"
        fi
    done
    echo "============================================"
}

# ---- 检查数据 ----
check_dataset() {
    local name=$1
    local path=$2
    local min_files=$3
    local count=$(ls $path 2>/dev/null | wc -l)
    if [ "$count" -lt "$min_files" ]; then
        warn "数据集 $name ($path) 文件不足 (需要>$min_files, 当前$count)"
        warn "请先运行: bash code/download_datasets.sh"
        return 1
    fi
    return 0
}

# ---- 命令别名 ----
PY="/home/oyp/miniconda3/bin/python3 -c"

# ============================================================
# 参数解析
# ============================================================
START_FROM=1
FORCE_STAGES=()  # 需要强制重跑的阶段列表

# 先扫描所有参数
ARGS=("$@")
for ((i=0; i<${#ARGS[@]}; i++)); do
    case "${ARGS[$i]}" in
        --list)
            echo "============================================"
            echo " ForensicGuard — 各阶段耗时估算"
            echo " (基于RTX 4090, 数据量2000样本/域)"
            echo "============================================"
            total=0
            for entry in "${STAGE_TIMES[@]}"; do
                idx="${entry%%|*}"
                rest="${entry#*|}"
                name="${rest%%|*}"
                est="${rest##*|}"
                printf "  阶段 %s/9:  %-25s ~%3d 分钟\n" "$idx" "$name" "$est"
                total=$((total + est))
            done
            echo "--------------------------------------------"
            printf "  总计: ~%d 分钟 (~%.1f 小时)\n" $total $(echo "scale=1; $total/60" | bc)
            echo "============================================"
            exit 0
            ;;
        --status)
            show_stage_status
            exit 0
            ;;
        --stage)
            START_FROM="${ARGS[$((i+1))]}"
            i=$((i+1))
            ;;
        --force-stage)
            FORCE_STAGES+=("${ARGS[$((i+1))]}")
            i=$((i+1))
            ;;
    esac
done

# 应用 --stage: 清除 START_FROM 及后续标记
if [ "$START_FROM" -gt 1 ]; then
    log "从阶段 $START_FROM 开始运行 (清除 $START_FROM~$TOTAL_STAGES 的检查点)"
    clear_stage_marker $START_FROM
fi

# 应用 --force-stage: 只清除指定阶段的标记
for fs in "${FORCE_STAGES[@]}"; do
    log "强制重跑阶段 $fs (清除其检查点, 保留前后阶段)"
    rm -f "$RESULT_DIR/.stage${fs}_done"
    # 也从结果文件中移除
    if [ -f "$STAGE_RESULTS_FILE" ]; then
        python3 -c "
import json
with open('$STAGE_RESULTS_FILE', 'r') as f:
    data = json.load(f)
data.pop('stage_${fs}', None)
with open('$STAGE_RESULTS_FILE', 'w') as f:
    json.dump(data, f, indent=2)
" 2>/dev/null || true
    fi
done

# ============================================================
# 数据集检查
# ============================================================
header "数据集检查"
DATASET_OK=true
check_dataset "CASIA v2"    "$DATA_DIR/casia2/CASIA2.0_revised/Au/*.jpg" 100   || DATASET_OK=false
check_dataset "Columbia"    "$DATA_DIR/columbia/ImSpliceDataset/*/*.bmp"  100   || DATASET_OK=false
check_dataset "COCO"        "$DATA_DIR/coco/val2017/*.jpg"              100   || DATASET_OK=false
if [ "$DATASET_OK" = false ]; then
    warn "部分数据集缺失, 但实验仍可使用已有特征继续"
    warn "特征文件检查: $(ls $FEATURE_DIR/*_test_features.pt 2>/dev/null | wc -l) 个"
fi

# ============================================================
# 主循环
# ============================================================
header "🚀 ForensicGuard 完整实验"
log "开始时间: $(date)"
log "共 $TOTAL_STAGES 个阶段, 从阶段 $START_FROM 开始"
echo ""

for ((stage=START_FROM; stage<=TOTAL_STAGES; stage++)); do

    # ---- 阶段计时 ----
    STAGE_START=$SECONDS

    # ---- 解析阶段信息 ----
    for entry in "${STAGE_TIMES[@]}"; do
        idx="${entry%%|*}"
        if [ "$idx" = "$stage" ]; then
            rest="${entry#*|}"
            STAGE_NAME="${rest%%|*}"
            STAGE_EST="${rest##*|}"
            break
        fi
    done

    # ---- 进度显示 ----
    show_progress $((stage - 1)) $TOTAL_STAGES $SECONDS
    show_next_stage $stage $TOTAL_STAGES

    # ---- 跳过已完成 ----
    if run_stage $stage; then
        log "阶段 ${stage}/${TOTAL_STAGES} [${STAGE_NAME}] ✅ 已完成, 跳过"
        # 确保结果已保存 (如果之前没保存过)
        if [ -f "$STAGE_RESULTS_FILE" ]; then
            python3 -c "
import json
with open('$STAGE_RESULTS_FILE') as f:
    data = json.load(f)
if 'stage_${stage}' not in data:
    data['stage_${stage}'] = {
        'stage': ${stage},
        'name': '${STAGE_NAME}',
        'status': 'skipped',
        'timestamp': '$(date +%Y-%m-%d_%H:%M:%S)',
        'note': '已完成, 跳过重跑'
    }
    with open('$STAGE_RESULTS_FILE', 'w') as f:
        json.dump(data, f, indent=2)
    print(f'[结果] 阶段${stage} 已记录为 skipped')
" 2>/dev/null || true
        fi
        echo ""
        continue
    fi

    # ---- 开始阶段 ----
    echo ""
    header "阶段 ${stage}/${TOTAL_STAGES}: ${STAGE_NAME}"
    log "预计耗时: ~${STAGE_EST} 分钟"
    log "开始时间: $(date +%H:%M:%S)"

    # ============================================================
    # 阶段1: 生成v2补丁数据集
    # ============================================================
    if [ $stage -eq 1 ]; then
        log "检查数据源..."
        $PY "
import sys, os, torch
sys.path.insert(0, '$SCRIPT_DIR')
from scripts.patch_attack import AdversarialPatchGeneratorV2 as APG
from dataset.instance_dataset import InstanceForensicDataset

DATA_DIR='$DATA_DIR'
SAVE_DIR='$FEATURE_DIR/patch_dataset_v2'
os.makedirs(SAVE_DIR, exist_ok=True)
device='cuda' if torch.cuda.is_available() else 'cpu'
print(f'[阶段1] 设备: {device}')
mean=torch.tensor([0.485,0.456,0.406]).view(3,1,1)
std=torch.tensor([0.229,0.224,0.225]).view(3,1,1)

# 收集干净图像
all_clean=[]
for ds_name in ['casia2','augmented_coco']:
    try:
        ds=InstanceForensicDataset(DATA_DIR,ds_name,'train',img_size=512)
        for i in range(len(ds)):
            img_t,label,_=ds[i]
            if label==0:
                all_clean.append(img_t*std+mean)
            if len(all_clean)>=2000:
                break
        print(f'  {ds_name}: {sum(1 for _,l,_ in ds if l==0)} Au → 取 {len([c for c in all_clean])} 张')
    except:
        print(f'  {ds_name}: 加载失败, 跳过')

# 保存为三域数据
import random
random.shuffle(all_clean)
all_clean=all_clean[:2000]
print(f'总干净图像: {len(all_clean)} 张')

gen=APG(patch_size=128,device=device)
clean_pairs=[(img,0) for img in all_clean]

print('生成数字域补丁 (无PC仿真)...')
d_clean=clean_pairs[:min(1000,len(clean_pairs)//2)]
_,digital_patches=gen.generate_dataset(d_clean,augment_pc=False)

print('生成物理域补丁 (增强PC仿真)...')
p_clean=clean_pairs[min(1000,len(clean_pairs)//2):min(2000,len(clean_pairs))]
_,physical_patches=gen.generate_dataset(p_clean,augment_pc=True,severity='strong')

def to_records(samples,label):
    return [{'tensor':s[0] if isinstance(s,tuple) else s,'label':label} for s in samples]

clean_records=to_records(d_clean,0)
digital_records=[{'tensor':p[0],'label':1} for p in digital_patches if p[1]==1]
physical_records=[{'tensor':p[0],'label':1} for p in physical_patches if p[1]==1]

for domain,records in [('clean',clean_records),('digital_patch',digital_records),('physical_patch',physical_records)]:
    random.shuffle(records)
    n=len(records)
    n_train,n_val=int(n*0.6),int(n*0.2)
    for split_name,split_records in [('train',records[:n_train]),
                                       ('val',records[n_train:n_train+n_val]),
                                       ('test',records[n_train+n_val:])]:
        torch.save({'tensors':[r['tensor'] for r in split_records],
                     'labels':[r['label'] for r in split_records]},
                    f'{SAVE_DIR}/{domain}_{split_name}.pt')
    print(f'  {domain}: train={n_train} val={n_val} test={n-n_train-n_val}')
print('[阶段1] 完成!')
" 2>&1 | tee "$LOG_DIR/step1_patchgen.log"
        stage_elapsed=$((SECONDS - STAGE_START))
        mark_stage_done 1
        save_stage_result 1 "completed" $stage_elapsed '{"status": "patch_dataset_generated", "domain_count": 3}'
        log "阶段1 完成 ⏱ $(format_time $stage_elapsed)"

    # ============================================================
    # 阶段2: 特征提取 (v2补丁数据集)
    # ============================================================
    elif [ $stage -eq 2 ]; then
        log "检查patch_dataset_v2目录..."
        for domain in clean digital_patch physical_patch; do
            for split in train val test; do
                f="$FEATURE_DIR/patch_dataset_v2/${domain}_${split}.pt"
                if [ ! -f "$f" ]; then
                    error "缺少: $f, 请先运行阶段1"
                    exit 1
                fi
                log "提取特征: ${domain}/${split}..."
                $PY "
import sys,os,torch
sys.path.insert(0,'$SCRIPT_DIR')
from features.segmenter import InstanceSegmenter
from features.spn_extractor import get_spn_extractor
from features.dct_profile import DCTProfileExtractor
from features.clip_encoder import get_clip_extractor
from tqdm import tqdm
device='cuda' if torch.cuda.is_available() else 'cpu'
data=torch.load('$FEATURE_DIR/patch_dataset_v2/${domain}_${split}.pt',map_location='cpu', weights_only=False)
tensors=data['tensors']; labels=data['labels']
seg=InstanceSegmenter(device=device)
spn=get_spn_extractor(mode='learned',device=device)
dct=DCTProfileExtractor()
clip=get_clip_extractor(device=device)
all_spn,all_dct,all_clip,all_masks=[],[],[],[]
for i in tqdm(range(len(tensors)),desc='${domain}/${split}'):
    img=tensors[i].to(device)
    masks_list=seg.segment(img)
    masks_list=[(m,b) for m,b in masks_list if m.sum()>64*64]
    all_masks.append([m for m,_ in masks_list])
    K=len(masks_list)
    if K<2:
        spn_f=spn.extract(img); dct_f=dct.extract_profile(img.cpu()); clip_f=clip.encode_image(img.cpu())
        all_spn.append(spn_f.unsqueeze(0)); all_dct.append(dct_f.unsqueeze(0)); all_clip.append(clip_f.unsqueeze(0))
    else:
        spn_list,dct_list,clip_list=[],[],[]
        for mask,_ in masks_list:
            mask_t=torch.from_numpy(mask).float().to(device)
            spn_list.append(spn.extract_instance(img,mask_t))
            dct_list.append(dct.extract_profile(img.cpu(),mask))
            clip_list.append(clip.encode_image(img.cpu(),mask))
        all_spn.append(torch.stack(spn_list)); all_dct.append(torch.stack(dct_list)); all_clip.append(torch.stack(clip_list))
    if (i+1)%100==0:
        print(f'  progress: {i+1}/{len(tensors)}')
torch.save({'spn_feats':all_spn,'dct_feats':all_dct,'clip_feats':all_clip,
             'labels':labels,'instance_counts':[len(s) for s in all_spn],
             'masks':all_masks},'$FEATURE_DIR/${domain}_${split}_features.pt')
print(f'  ✓ 保存: {len(labels)} samples')
" 2>&1 | tee -a "$LOG_DIR/step2_features.log"
                done
        done
        stage_elapsed=$((SECONDS - STAGE_START))
        mark_stage_done 2
        # 从日志提取样本数
        _n_samples=$(grep -h "saved:" "$LOG_DIR/step2_features.log" 2>/dev/null | grep -oP '\d+(?= samples)' | tail -1)
        save_stage_result 2 "completed" $stage_elapsed "{\"total_samples\": ${_n_samples:-0}}"
        log "阶段2 完成 ⏱ $(format_time $stage_elapsed)"

    # ============================================================
    # 阶段3: 数字域基线 (clean + digital_patch)
    # ============================================================
    elif [ $stage -eq 3 ]; then
        $PY "
import sys,torch,torch.nn as nn
sys.path.insert(0,'$SCRIPT_DIR')
FEATURE_DIR='$FEATURE_DIR'; MODEL_DIR='$MODEL_DIR/digital_baseline_v2'
import os; os.makedirs(MODEL_DIR,exist_ok=True)
from models.consistency import CrossInstanceConsistencyV2
from models.fusion_classifier import FusionClassifierV2
from scripts.train import compute_fused_features, evaluate
device='cuda' if torch.cuda.is_available() else 'cpu'
cic=CrossInstanceConsistencyV2()

def load_and_merge(domains,split):
    all_s,all_d,all_c,all_l=[],[],[],[]
    for d in domains:
        data=torch.load(f'{FEATURE_DIR}/{d}_{split}_features.pt',map_location='cpu', weights_only=False)
        all_s.extend(data['spn_feats']); all_d.extend(data['dct_feats'])
        all_c.extend(data['clip_feats']); all_l.extend(data['labels'])
    return all_s,all_d,all_c,all_l

s_tr,d_tr,c_tr,l_tr=load_and_merge(['clean','digital_patch'],'train')
s_val,d_val,c_val,l_val=load_and_merge(['clean','digital_patch'],'val')
n_tr=len(l_tr); n_val=len(l_val)
print(f'训练: {n_tr} 样本, 验证: {n_val} 样本')

        # clean 样本权重 2.0, patch 权重 1.0
model=FusionClassifierV2(input_dim=698).to(device)
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
        cls_weights = torch.where(batch_y==0, 2.0, 1.0)
criterion=nn.BCEWithLogitsLoss()
best_acc=0.0
for epoch in range(300):
    model.train(); total_loss=0
    indices=torch.randperm(n_tr)
    for start in range(0,n_tr,64):
        end=min(start+64,n_tr); idx=indices[start:end]
        batch_s=[s_tr[i] for i in idx]; batch_d=[d_tr[i] for i in idx]
        batch_c=[c_tr[i] for i in idx]; batch_y=torch.tensor([l_tr[i] for i in idx],dtype=torch.float32,device=device)
        fused=compute_fused_features(batch_s,batch_d,batch_c,cic,device)
        logits=model(fused).squeeze(-1); loss=criterion(logits,batch_y)
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
        total_loss+=loss.item()*len(idx)
    if (epoch+1)%10==0:
        m=evaluate(model,s_val,d_val,c_val,l_val,cic,device)
        if m['accuracy']>best_acc:
            best_acc=m['accuracy']; torch.save(model.state_dict(),f'{MODEL_DIR}/fusion_best.pth')
        print(f'  Epoch {epoch+1}/100: loss={total_loss/n_tr*64:.4f}, val_acc={m[\"accuracy\"]*100:.1f}%')

print(f'最佳验证精度: {best_acc*100:.1f}%')
model.load_state_dict(torch.load(f'{MODEL_DIR}/fusion_best.pth',map_location='cpu', weights_only=False))
model=model.to(device); model.eval()
print(f'{\"测试域\":<25}{\"Acc\":>8}{\"Prec\":>8}{\"Recall\":>8}{\"F1\":>8}')
print('-'*57)
for domain in ['clean','digital_patch','physical_patch']:
    d=torch.load(f'{FEATURE_DIR}/{domain}_test_features.pt',map_location='cpu', weights_only=False)
    m=evaluate(model,d['spn_feats'],d['dct_feats'],d['clip_feats'],d['labels'],cic,device)
    print(f'  {domain:<23}{m[\"accuracy\"]*100:>7.1f}%{m[\"precision\"]*100:>7.1f}%{m[\"recall\"]*100:>7.1f}%{m[\"f1\"]*100:>7.1f}%')
" 2>&1 | tee "$LOG_DIR/step3_digital_v2.log"
        stage_elapsed=$((SECONDS - STAGE_START))
        mark_stage_done 3
        # 提取准确率
        _acc3=$(grep -oP 'physical_patch\s+\K[\d.]+(?=%)' "$LOG_DIR/step3_digital_v2.log" 2>/dev/null | tail -1)
        save_stage_result 3 "completed" $stage_elapsed "{\"physical_patch_acc\": ${_acc3:-0}}"
        log "阶段3 完成 ⏱ $(format_time $stage_elapsed)"

    # ============================================================
    # 阶段4: 物理域衰减
    # ============================================================
    elif [ $stage -eq 4 ]; then
        $PY "
import sys,torch
sys.path.insert(0,'$SCRIPT_DIR')
FEATURE_DIR='$FEATURE_DIR'; MODEL_PATH='$MODEL_DIR/digital_baseline_v2/fusion_best.pth'
from models.consistency import CrossInstanceConsistencyV2
from models.fusion_classifier import FusionClassifierV2
from scripts.train import compute_fused_features, evaluate
device='cuda' if torch.cuda.is_available() else 'cpu'
cic=CrossInstanceConsistencyV2()
model=FusionClassifierV2(input_dim=698)
model.load_state_dict(torch.load(MODEL_PATH,map_location='cpu', weights_only=False)); model=model.to(device); model.eval()
print('=== 数字模型→各域测试 (物理域衰减) ===')
for domain in ['clean','digital_patch','physical_patch']:
    d=torch.load(f'{FEATURE_DIR}/{domain}_test_features.pt',map_location='cpu', weights_only=False)
    m=evaluate(model,d['spn_feats'],d['dct_feats'],d['clip_feats'],d['labels'],cic,device)
    print(f'  {domain:20s}: acc={m[\"accuracy\"]*100:.1f}%  f1={m[\"f1\"]*100:.1f}%')
" 2>&1 | tee "$LOG_DIR/step4_physical_v2.log"
        stage_elapsed=$((SECONDS - STAGE_START))
        mark_stage_done 4
        # 提取各域准确率
        _acc4=$(grep -oP 'physical_patch\s*: acc=\K[\d.]+(?=%)' "$LOG_DIR/step4_physical_v2.log" 2>/dev/null | tail -1)
        save_stage_result 4 "completed" $stage_elapsed "{\"physical_patch_acc\": ${_acc4:-0}}"
        log "阶段4 完成 ⏱ $(format_time $stage_elapsed)"

    # ============================================================
    # 阶段5: 增强训练 (含物理域样本)
    # ============================================================
    elif [ $stage -eq 5 ]; then
        $PY "
import sys,torch,torch.nn as nn
sys.path.insert(0,'$SCRIPT_DIR')
FEATURE_DIR='$FEATURE_DIR'; MODEL_DIR='$MODEL_DIR/enhanced_v2'
import os; os.makedirs(MODEL_DIR,exist_ok=True)
from models.consistency import CrossInstanceConsistencyV2
from models.fusion_classifier import FusionClassifierV2
from scripts.train import compute_fused_features, evaluate
device='cuda' if torch.cuda.is_available() else 'cpu'
cic=CrossInstanceConsistencyV2()
def load_and_merge(domains,split):
    all_s,all_d,all_c,all_l=[],[],[],[]
    for d in domains:
        data=torch.load(f'{FEATURE_DIR}/{d}_{split}_features.pt',map_location='cpu', weights_only=False)
        all_s.extend(data['spn_feats']); all_d.extend(data['dct_feats'])
        all_c.extend(data['clip_feats']); all_l.extend(data['labels'])
    return all_s,all_d,all_c,all_l
s_tr,d_tr,c_tr,l_tr=load_and_merge(['clean','physical_patch'],'train')
s_val,d_val,c_val,l_val=load_and_merge(['clean','physical_patch'],'val')
n_tr=len(l_tr); print(f'训练: {n_tr} 样本')
model=FusionClassifierV2(input_dim=698).to(device)
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
criterion=nn.BCEWithLogitsLoss()
best_acc=0.0
for epoch in range(300):
    model.train(); total_loss=0
    indices=torch.randperm(n_tr)
    for start in range(0,n_tr,64):
        end=min(start+64,n_tr); idx=indices[start:end]
        batch_s=[s_tr[i] for i in idx]; batch_d=[d_tr[i] for i in idx]
        batch_c=[c_tr[i] for i in idx]; batch_y=torch.tensor([l_tr[i] for i in idx],dtype=torch.float32,device=device)
        fused=compute_fused_features(batch_s,batch_d,batch_c,cic,device)
        logits=model(fused).squeeze(-1); loss=criterion(logits,batch_y)
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
        total_loss+=loss.item()*len(idx)
    if (epoch+1)%10==0:
        m=evaluate(model,s_val,d_val,c_val,l_val,cic,device)
        if m['accuracy']>best_acc: best_acc=m['accuracy']; torch.save(model.state_dict(),f'{MODEL_DIR}/fusion_best.pth')
        print(f'  Epoch {epoch+1}/100: loss={total_loss/n_tr*64:.4f}, val_acc={m[\"accuracy\"]*100:.1f}%')
print(f'最佳验证: {best_acc*100:.1f}%')
model.load_state_dict(torch.load(f'{MODEL_DIR}/fusion_best.pth',map_location='cpu', weights_only=False)); model=model.to(device); model.eval()
for domain in ['clean','digital_patch','physical_patch']:
    d=torch.load(f'{FEATURE_DIR}/{domain}_test_features.pt',map_location='cpu', weights_only=False)
    m=evaluate(model,d['spn_feats'],d['dct_feats'],d['clip_feats'],d['labels'],cic,device)
    print(f'  {domain:<23}{m[\"accuracy\"]*100:>7.1f}%')
" 2>&1 | tee "$LOG_DIR/step5_enhanced_v2.log"
        stage_elapsed=$((SECONDS - STAGE_START))
        mark_stage_done 5
        _acc5=$(grep -oP 'physical_patch\s+\K[\d.]+(?=%)' "$LOG_DIR/step5_enhanced_v2.log" 2>/dev/null | tail -1)
        save_stage_result 5 "completed" $stage_elapsed "{\"physical_patch_acc\": ${_acc5:-0}}"
        log "阶段5 完成 ⏱ $(format_time $stage_elapsed)"

    # ============================================================
    # 阶段6: 经典篡改检测
    # ============================================================
    elif [ $stage -eq 6 ]; then
        for ds in casia2 columbia augmented_coco; do
            log "训练: ${ds}..."
            F="$FEATURE_DIR/${ds}_train_features.pt"
            if [ ! -f "$F" ]; then
                warn "特征不存在: $F, 跳过 $ds"
                continue
            fi
            $PY "
import sys,torch,torch.nn as nn
sys.path.insert(0,'$SCRIPT_DIR')
FEATURE_DIR='$FEATURE_DIR'; MODEL_DIR='$MODEL_DIR/${ds}'; DS='$ds'
import os; os.makedirs(MODEL_DIR,exist_ok=True)
from models.consistency import CrossInstanceConsistencyV2
from models.fusion_classifier import FusionClassifierV2, FocalLoss
from scripts.train import compute_fused_features, evaluate
device='cuda' if torch.cuda.is_available() else 'cpu'
cic=CrossInstanceConsistencyV2()
data=torch.load(f'{FEATURE_DIR}/{DS}_train_features.pt',map_location='cpu', weights_only=False)
s_tr,d_tr,c_tr=data['spn_feats'],data['dct_feats'],data['clip_feats']; l_tr=data['labels']
data=torch.load(f'{FEATURE_DIR}/{DS}_val_features.pt',map_location='cpu', weights_only=False)
s_val,d_val,c_val=data['spn_feats'],data['dct_feats'],data['clip_feats']; l_val=data['labels']
n_tr=len(l_tr)
model=FusionClassifierV2(input_dim=698).to(device)
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
criterion=FocalLoss(alpha=0.75,gamma=2.0)
best_acc=0.0
for epoch in range(300):
    model.train(); total_loss=0
    indices=torch.randperm(n_tr)
    for start in range(0,n_tr,64):
        end=min(start+64,n_tr); idx=indices[start:end]
        batch_s=[s_tr[i] for i in idx]; batch_d=[d_tr[i] for i in idx]; batch_c=[c_tr[i] for i in idx]
        batch_y=torch.tensor([l_tr[i] for i in idx],dtype=torch.float32,device=device)
        fused=compute_fused_features(batch_s,batch_d,batch_c,cic,device)
        logits=model(fused).squeeze(-1); loss=criterion(logits,batch_y)
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
        total_loss+=loss.item()*len(idx)
    if (epoch+1)%10==0:
        m=evaluate(model,s_val,d_val,c_val,l_val,cic,device)
        if m['accuracy']>best_acc: best_acc=m['accuracy']; torch.save(model.state_dict(),f'{MODEL_DIR}/fusion_best.pth')
        print(f'  Epoch {epoch+1}/100: loss={total_loss/n_tr*64:.4f}, val_acc={m[\"accuracy\"]*100:.1f}%')
model.load_state_dict(torch.load(f'{MODEL_DIR}/fusion_best.pth',map_location='cpu', weights_only=False)); model=model.to(device); model.eval()
data=torch.load(f'{FEATURE_DIR}/{DS}_test_features.pt',map_location='cpu', weights_only=False)
m=evaluate(model,data['spn_feats'],data['dct_feats'],data['clip_feats'],data['labels'],cic,device)
print(f'\\n=== {DS} ===')
print(f'Test Acc: {m[\"accuracy\"]*100:.2f}%, F1: {m[\"f1\"]*100:.2f}%')
print(f'Au(0): tp={m[\"tp\"]} fp={m[\"fp\"]} tn={m[\"tn\"]} fn={m[\"fn\"]}')
" 2>&1 | tee "$LOG_DIR/step6_${ds}.log"
        done
        stage_elapsed=$((SECONDS - STAGE_START))
        mark_stage_done 6
        # 提取各数据集准确率
        _acc_casia=$(grep "Test Acc" "$LOG_DIR/step6_casia2.log" 2>/dev/null | awk '{print $3}' | sed 's/%,//')
        _acc_columbia=$(grep "Test Acc" "$LOG_DIR/step6_columbia.log" 2>/dev/null | awk '{print $3}' | sed 's/%,//')
        _acc_coco=$(grep "Test Acc" "$LOG_DIR/step6_augmented_coco.log" 2>/dev/null | awk '{print $3}' | sed 's/%,//')
        save_stage_result 6 "completed" $stage_elapsed "{\"casia2\": ${_acc_casia:-0}, \"columbia\": ${_acc_columbia:-0}, \"coco\": ${_acc_coco:-0}}"
        log "阶段6 完成 ⏱ $(format_time $stage_elapsed)"

    # ============================================================
    # 阶段7: 消融实验
    # ============================================================
    elif [ $stage -eq 7 ]; then
        for ablation in full wospn wodct woclip; do
            log "消融: ${ablation}..."
            $PY "
import sys,torch,torch.nn as nn
sys.path.insert(0,'$SCRIPT_DIR')
FEATURE_DIR='$FEATURE_DIR'; DS='casia2'
from models.consistency import CrossInstanceConsistencyV2
from models.fusion_classifier import FusionClassifierV2
from scripts.train import compute_fused_features, evaluate
device='cuda' if torch.cuda.is_available() else 'cpu'
cic=CrossInstanceConsistencyV2()
data=torch.load(f'{FEATURE_DIR}/{DS}_train_features.pt',map_location='cpu', weights_only=False)
s_tr,d_tr,c_tr=data['spn_feats'],data['dct_feats'],data['clip_feats']; l_tr=data['labels']
data=torch.load(f'{FEATURE_DIR}/{DS}_test_features.pt',map_location='cpu', weights_only=False)
s_te,d_te,c_te=data['spn_feats'],data['dct_feats'],data['clip_feats']; l_te=data['labels']
ablation_name='$ablation'
if ablation_name=='wospn':   s_tr=[torch.zeros_like(f) for f in s_tr]; s_te=[torch.zeros_like(f) for f in s_te]
if ablation_name=='wodct':   d_tr=[torch.zeros_like(f) for f in d_tr]; d_te=[torch.zeros_like(f) for f in d_te]
if ablation_name=='woclip':  c_tr=[torch.zeros_like(f) for f in c_tr]; c_te=[torch.zeros_like(f) for f in c_te]
model=FusionClassifierV2(input_dim=698).to(device)
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
criterion=nn.BCEWithLogitsLoss()
for epoch in range(50):
    model.train(); total_loss=0
    indices=torch.randperm(len(l_tr))
    for start in range(0,len(l_tr),64):
        end=min(start+64,len(l_tr)); idx=indices[start:end]
        batch_s=[s_tr[i] for i in idx]; batch_d=[d_tr[i] for i in idx]; batch_c=[c_tr[i] for i in idx]
        batch_y=torch.tensor([l_tr[i] for i in idx],dtype=torch.float32,device=device)
        fused=compute_fused_features(batch_s,batch_d,batch_c,cic,device)
        logits=model(fused).squeeze(-1); loss=criterion(logits,batch_y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss+=loss.item()*len(idx)
m=evaluate(model,s_te,d_te,c_te,l_te,cic,device)
print(f'Ablation {ablation_name}: Acc={m[\"accuracy\"]*100:.2f}%, F1={m[\"f1\"]*100:.2f}%')
" 2>&1 | tee "$LOG_DIR/step7_ablation_${ablation}.log"
        done
        stage_elapsed=$((SECONDS - STAGE_START))
        mark_stage_done 7
        # 提取消融结果
        _full=$(grep "Ablation full" "$LOG_DIR/step7_ablation_full.log" 2>/dev/null | grep -oP 'Acc=\K[\d.]+')
        _wospn=$(grep "Ablation wospn" "$LOG_DIR/step7_ablation_wospn.log" 2>/dev/null | grep -oP 'Acc=\K[\d.]+')
        _wodct=$(grep "Ablation wodct" "$LOG_DIR/step7_ablation_wodct.log" 2>/dev/null | grep -oP 'Acc=\K[\d.]+')
        _woclip=$(grep "Ablation woclip" "$LOG_DIR/step7_ablation_woclip.log" 2>/dev/null | grep -oP 'Acc=\K[\d.]+')
        save_stage_result 7 "completed" $stage_elapsed "{\"full\": ${_full:-0}, \"w\/o SPN\": ${_wospn:-0}, \"w\/o DCT\": ${_wodct:-0}, \"w\/o CLIP\": ${_woclip:-0}}"
        log "阶段7 完成 ⏱ $(format_time $stage_elapsed)"

    # ============================================================
    # 阶段8: 结果汇总
    # ============================================================
    elif [ $stage -eq 8 ]; then
        echo "============================================" | tee "$SUMMARY_FILE"
        echo " ForensicGuard — 完整实验结果" | tee -a "$SUMMARY_FILE"
        echo " 生成时间: $(date)" | tee -a "$SUMMARY_FILE"
        echo " 总耗时: $(format_time $SECONDS)" | tee -a "$SUMMARY_FILE"
        echo "============================================" | tee -a "$SUMMARY_FILE"

        echo "" | tee -a "$SUMMARY_FILE"
        echo "--- 1. 经典篡改检测 ---" | tee -a "$SUMMARY_FILE"
        for ds in casia2 columbia augmented_coco; do
            logfile="$LOG_DIR/step6_${ds}.log"
            if [ -f "$logfile" ]; then
                acc=$(grep "Test Acc" "$logfile" | awk '{print $3}')
                f1=$(grep "F1:" "$logfile" | awk '{print $2}')
                echo "  $ds: Acc=$acc, F1=$f1" | tee -a "$SUMMARY_FILE"
            fi
        done

        echo "" | tee -a "$SUMMARY_FILE"
        echo "--- 2. 物理补丁检测 ---" | tee -a "$SUMMARY_FILE"
        for logfile in step3_digital_v2.log step4_physical_v2.log step5_enhanced_v2.log; do
            if [ -f "$LOG_DIR/$logfile" ]; then
                echo "  [$logfile]" | tee -a "$SUMMARY_FILE"
                grep -E "^\s+clean|^\s+digital|^\s+physical" "$LOG_DIR/$logfile" 2>/dev/null | while read line; do
                    echo "    $line" | tee -a "$SUMMARY_FILE"
                done
            fi
        done

        echo "" | tee -a "$SUMMARY_FILE"
        echo "--- 3. 消融实验 (CASIA v2) ---" | tee -a "$SUMMARY_FILE"
        for ablation in full wospn wodct woclip; do
            logfile="$LOG_DIR/step7_ablation_${ablation}.log"
            if [ -f "$logfile" ]; then
                result=$(grep "Ablation" "$logfile")
                echo "  $result" | tee -a "$SUMMARY_FILE"
            fi
        done

        echo "" | tee -a "$SUMMARY_FILE"
        echo "============================================" | tee -a "$SUMMARY_FILE"
        log "全部完成! 结果汇总: $SUMMARY_FILE"
        mark_stage_done 8

    # ============================================================
    # 阶段9: SOTA Baseline对比
    # ============================================================
    elif [ $stage -eq 9 ]; then
        log "阶段9: ResNet-18 图像级分类 baseline..."
        $PY "
import sys, os, torch, torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import time

sys.path.insert(0, '$SCRIPT_DIR')
FEATURE_DIR='$FEATURE_DIR'
LOG_DIR='$LOG_DIR'
SAVE_DIR='$MODEL_DIR/baseline_resnet18'
os.makedirs(SAVE_DIR, exist_ok=True)

# 清理GPU内存
torch.cuda.empty_cache()
device='cuda' if torch.cuda.is_available() else 'cpu'
print(f'设备: {device}, GPU可用内存: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')

# 加载patch数据集
class PatchDataset(Dataset):
    def __init__(self, domain, split):
        data = torch.load(f'$FEATURE_DIR/patch_dataset_v2/{domain}_{split}.pt', map_location='cpu', weights_only=False)
        self.tensors = data['tensors']
        self.labels = data['labels']
        # 归一化到ImageNet标准
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    def __len__(self): return len(self.tensors)
    def __getitem__(self, i):
        img = self.tensors[i]
        if img.max() > 1.0: img = img / 255.0
        img = (img - self.mean) / self.std  # 归一化
        return img, self.labels[i]

# ResNet-18
model = models.resnet18(weights='DEFAULT')
model.fc = nn.Linear(512, 1)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
criterion = nn.BCEWithLogitsLoss()

def evaluate_resnet(model, loader, device):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.float().to(device)
            out = model(imgs).squeeze(-1)
            preds = (torch.sigmoid(out) > 0.5).float()
            correct += (preds == lbls).sum().item()
            total += len(lbls)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(lbls.cpu().tolist())
    acc = correct / total if total > 0 else 0
    return acc

# 训练
BATCH_SIZE = 32
EPOCHS = 30

from torch.utils.data import ConcatDataset
train_ds = ConcatDataset([PatchDataset(d,'train') for d in ['clean','digital_patch','physical_patch']])
val_ds   = ConcatDataset([PatchDataset(d,'val') for d in ['clean','digital_patch','physical_patch']])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f'  train: {len(train_ds)} samples, val: {len(val_ds)} samples')

start_t = time.time()
best_acc = 0.0
for epoch in range(EPOCHS):
    model.train(); total_loss = 0
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(device), lbls.float().to(device)
        loss = criterion(model(imgs).squeeze(-1), lbls)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item() * len(imgs)
    val_acc = evaluate_resnet(model, val_loader, device)
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), f'{SAVE_DIR}/resnet18_best.pth')
    # 每epoch显示：loss + 准确率 + 已用/剩余时间
    elapsed = time.time() - start_t
    remaining = elapsed / (epoch+1) * (EPOCHS - epoch - 1)
    print(f'  Epoch {epoch+1}/{EPOCHS}: loss={total_loss/len(train_ds):.4f}, val_acc={val_acc*100:.1f}%'
          f' | ⏱ {int(elapsed//60):02d}:{int(elapsed%60):02d} 剩余{int(remaining//60):02d}:{int(remaining%60):02d}')
    sys.stdout.flush()

model.load_state_dict(torch.load(f'{SAVE_DIR}/resnet18_best.pth', map_location='cpu', weights_only=True))
model = model.to(device)

# 逐域测试
print(f'\\n{\"ResNet-18 Baseline\":<25}{\"Acc\":>8}')
print('-'*35)
for domain in ['clean','digital_patch','physical_patch']:
    ds = PatchDataset(domain, 'test')
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    acc = evaluate_resnet(model, loader, device)
    print(f'  {domain:<23}{acc*100:>7.1f}%')
" 2>&1 | tee "$LOG_DIR/step9_resnet18.log"

        log "阶段9: 全图特征+MLP baseline..."
        $PY "
import sys, torch, torch.nn as nn
sys.path.insert(0, '$SCRIPT_DIR')
FEATURE_DIR='$FEATURE_DIR'
from models.fusion_classifier import FusionClassifierV2
from scripts.train import compute_fused_features, evaluate

device='cuda' if torch.cuda.is_available() else 'cpu'

# 不使用实例分割, 直接用整图提取一份特征
# 对每个样本只用第1个实例的特征 (如果K>=1)
print('加载全图特征 (不使用实例分割)...')
train_data = torch.load('$FEATURE_DIR/clean_train_features.pt', map_location='cpu', weights_only=False)
for split_name in ['train','val','test']:
    all_s, all_d, all_c, all_l = [], [], [], []
    for domain in ['clean','digital_patch','physical_patch']:
        data = torch.load(f'$FEATURE_DIR/{domain}_{split_name}_features.pt', map_location='cpu', weights_only=False)
        for i in range(len(data['labels'])):
            # 取第0个实例的特征 (如果有)
            spn = data['spn_feats'][i][:1]  # (1, 128)
            dct = data['dct_feats'][i][:1]  # (1, 21)
            clip = data['clip_feats'][i][:1]  # (1, 512)
            all_s.append(spn); all_d.append(dct); all_c.append(clip)
            all_l.append(data['labels'][i])
    if split_name == 'train':
        s_tr, d_tr, c_tr, l_tr = all_s, all_d, all_c, all_l
    elif split_name == 'val':
        s_val, d_val, c_val, l_val = all_s, all_d, all_c, all_l
    else:
        s_te, d_te, c_te, l_te = all_s, all_d, all_c, all_l
    print(f'  {split_name}: {len(all_l)} samples')

# 不计算跨实例一致性 (因为只有1个实例)
# 直接拼接: SPN+DCT+CLIP 全局特征 = 661维, 用简单MLP
from models.consistency import CrossInstanceConsistencyV2
cic = CrossInstanceConsistencyV2()
model = FusionClassifierV2(input_dim=698).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
criterion = nn.BCEWithLogitsLoss()

n_tr = len(l_tr)
best_acc = 0.0
for epoch in range(100):
    model.train(); total_loss = 0
    indices = torch.randperm(n_tr)
    for start in range(0, n_tr, 64):
        end = min(start+64, n_tr); idx = indices[start:end]
        fused = compute_fused_features([s_tr[i] for i in idx],[d_tr[i] for i in idx],
                                        [c_tr[i] for i in idx], cic, device)
        batch_y = torch.tensor([l_tr[i] for i in idx], dtype=torch.float32, device=device)
        logits = model(fused).squeeze(-1); loss = criterion(logits, batch_y)
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        total_loss += loss.item() * len(idx)
    if (epoch+1)%20 == 0:
        fused_val = compute_fused_features(s_val,d_val,c_val,cic,device)
        val_y = torch.tensor(l_val, dtype=torch.float32, device=device)
        val_acc = (torch.sigmoid(model(fused_val).squeeze(-1)) > 0.5).float().eq(val_y).float().mean().item()
        if val_acc > best_acc: best_acc = val_acc
        print(f'  Epoch {epoch+1}/100: loss={total_loss/n_tr:.4f}, val_acc={val_acc*100:.1f}%')

# 测试
fused_te = compute_fused_features(s_te,d_te,c_te,cic,device)
te_y = torch.tensor(l_te, dtype=torch.float32, device=device)
preds = (torch.sigmoid(model(fused_te).squeeze(-1)) > 0.5).float()
total, correct = len(te_y), (preds == te_y).sum().item()
print(f'\\n{\"全图特征+MLP Baseline\":<25}{\"Acc\":>8}')
print('-'*35)
for domain in ['clean','digital_patch','physical_patch']:
    dom_data = torch.load(f'$FEATURE_DIR/{domain}_test_features.pt', map_location='cpu', weights_only=False)
    dom_s, dom_d, dom_c = dom_data['spn_feats'],dom_data['dct_feats'],dom_data['clip_feats']
    dom_s = [f[:1] for f in dom_s]; dom_d = [f[:1] for f in dom_d]; dom_c = [f[:1] for f in dom_c]
    dom_y = dom_data['labels']
    fused_dom = compute_fused_features(dom_s,dom_d,dom_c,cic,device)
    dom_preds = (torch.sigmoid(model(fused_dom).squeeze(-1)) > 0.5).float()
    dom_acc = (dom_preds == torch.tensor(dom_y, dtype=torch.float32, device=device)).float().mean().item()
    print(f'  {domain:<23}{dom_acc*100:>7.1f}%')
" 2>&1 | tee "$LOG_DIR/step9_global_feat.log"

        stage_elapsed=$((SECONDS - STAGE_START))
        mark_stage_done 9
        # 提取两个baseline的物理域准确率
        _r50=$(grep "physical_patch" "$LOG_DIR/step9_resnet18.log" 2>/dev/null | grep -oP '\d+\.\d+(?=%)' | tail -1)
        _gf=$(grep "physical_patch" "$LOG_DIR/step9_global_feat.log" 2>/dev/null | grep -oP '\d+\.\d+(?=%)' | tail -1)
        save_stage_result 9 "completed" $stage_elapsed "{\"resnet18\": ${_r50:-0}, \"global_feat\": ${_gf:-0}}"
        log "阶段9 完成 ⏱ $(format_time $stage_elapsed)"

    # ============================================================
    # 阶段10: 重提CASIA v2特征(v2管线) + FocalLoss训练
    # ============================================================
    elif [ $stage -eq 10 ]; then
        log "阶段10: 重提CASIA v2特征 (v2管线)..."
        # 用v2管线重新提取CASIA v2特征 (含masks)
        for split in train val test; do
            log "提取CASIA ${split}..."
            $PY "
import sys,os
os.chdir('$PROJECT_DIR')
import torch
sys.path.insert(0,'$SCRIPT_DIR')
from features.segmenter import InstanceSegmenter
from features.spn_extractor import get_spn_extractor
from features.dct_profile import DCTProfileExtractor
from features.clip_encoder import get_clip_extractor
from dataset.instance_dataset import InstanceForensicDataset
from tqdm import tqdm

device='cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR='$DATA_DIR'
SAVE_DIR='$FEATURE_DIR'
os.makedirs(SAVE_DIR, exist_ok=True)
mean=torch.tensor([0.485,0.456,0.406]).view(3,1,1)
std=torch.tensor([0.229,0.224,0.225]).view(3,1,1)

ds=InstanceForensicDataset(DATA_DIR,'casia2','$split',img_size=512)
seg=InstanceSegmenter(device=device)
spn=get_spn_extractor(mode='learned',device=device)
dct=DCTProfileExtractor()
clip=get_clip_extractor(device=device)

all_spn,all_dct,all_clip,all_masks,all_labels=[],[],[],[],[]
for i in tqdm(range(len(ds)),desc='casia2/$split'):
    img_t,label,_=ds[i]
    img_orig=img_t*std+mean
    masks_list=seg.segment(img_orig.to(device))
    masks_list=[(m,b) for m,b in masks_list if m.sum()>64*64]
    all_masks.append([m for m,_ in masks_list])
    K=len(masks_list)
    if K<2:
        spn_f=spn.extract(img_orig.to(device))
        dct_f=dct.extract_profile(img_orig.cpu())
        clip_f=clip.encode_image(img_orig.cpu())
        all_spn.append(spn_f.unsqueeze(0)); all_dct.append(dct_f.unsqueeze(0)); all_clip.append(clip_f.unsqueeze(0))
    else:
        sl,dl,cl=[],[],[]
        for mask,_ in masks_list:
            mt=torch.from_numpy(mask).float().to(device)
            sl.append(spn.extract_instance(img_orig.to(device),mt))
            dl.append(dct.extract_profile(img_orig.cpu(),mask))
            cl.append(clip.encode_image(img_orig.cpu(),mask))
        all_spn.append(torch.stack(sl)); all_dct.append(torch.stack(dl)); all_clip.append(torch.stack(cl))
    all_labels.append(label)
    if (i+1)%500==0:
        print(f'  casia2/$split: {i+1}/{len(ds)}')

torch.save({'spn_feats':all_spn,'dct_feats':all_dct,'clip_feats':all_clip,
             'labels':all_labels,'instance_counts':[len(s) for s in all_spn],
             'masks':all_masks},f'{SAVE_DIR}/casia2_v2_${split}_features.pt')
print(f'  casia2 ${split} 完成: {len(all_labels)} samples')
" 2>&1 | tee "$LOG_DIR/step10_casia_extract_${split}.log"
        done

        # 用新特征 + FocalLoss训练
        log "阶段10: 用v2特征+FocalLoss训练CASIA..."
        $PY "
import sys,os,torch,torch.nn as nn
sys.path.insert(0,'$SCRIPT_DIR')
FEATURE_DIR='$FEATURE_DIR'; SAVE_DIR='$FEATURE_DIR'; DS='casia2_v2'
from models.consistency import CrossInstanceConsistencyV2
from models.fusion_classifier import FusionClassifierV2, FocalLoss
from scripts.train import compute_fused_features, evaluate

device='cuda' if torch.cuda.is_available() else 'cpu'
cic=CrossInstanceConsistencyV2()

print('加载v2特征...')
data=torch.load(f'{SAVE_DIR}/casia2_v2_train_features.pt',map_location='cpu', weights_only=False)
s_tr,d_tr,c_tr=data['spn_feats'],data['dct_feats'],data['clip_feats']; l_tr=data['labels']
data=torch.load(f'{SAVE_DIR}/casia2_v2_val_features.pt',map_location='cpu', weights_only=False)
s_val,d_val,c_val=data['spn_feats'],data['dct_feats'],data['clip_feats']; l_val=data['labels']
data=torch.load(f'{SAVE_DIR}/casia2_v2_test_features.pt',map_location='cpu', weights_only=False)
s_te,d_te,c_te=data['spn_feats'],data['dct_feats'],data['clip_feats']; l_te=data['labels']
print(f'  train: {len(l_tr)}, val: {len(l_val)}, test: {len(l_te)}')

model=FusionClassifierV2(input_dim=698).to(device)
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
criterion=FocalLoss(alpha=0.75,gamma=2.0)
best_acc=0.0
for epoch in range(300):
    model.train(); total_loss=0
    indices=torch.randperm(len(l_tr))
    for start in range(0,len(l_tr),64):
        end=min(start+64,len(l_tr)); idx=indices[start:end]
        batch_s=[s_tr[i] for i in idx]; batch_d=[d_tr[i] for i in idx]; batch_c=[c_tr[i] for i in idx]
        batch_y=torch.tensor([l_tr[i] for i in idx],dtype=torch.float32,device=device)
        fused=compute_fused_features(batch_s,batch_d,batch_c,cic,device)
        logits=model(fused).squeeze(-1); loss=criterion(logits,batch_y)
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
        total_loss+=loss.item()*len(idx)
    if (epoch+1)%30==0:
        m=evaluate(model,s_val,d_val,c_val,l_val,cic,device)
        if m['accuracy']>best_acc: best_acc=m['accuracy']
        print(f'  Epoch {epoch+1}/300: loss={total_loss/len(l_tr)*64:.4f}, val_acc={m[\"accuracy\"]*100:.1f}%')
print(f'最佳验证: {best_acc*100:.1f}%')

# 测试
model.load_state_dict(torch.load(f'{MODEL_DIR}/{DS}/fusion_best.pth',map_location='cpu', weights_only=True)) if os.path.exists(f'{MODEL_DIR}/{DS}/fusion_best.pth') else None
m=evaluate(model,s_te,d_te,c_te,l_te,cic,device)
print(f'\\n=== CASIA v2 (v2特征+FocalLoss) ===')
print(f'Test Acc: {m[\"accuracy\"]*100:.2f}%, F1: {m[\"f1\"]*100:.2f}%')
print(f'Au(0): tp={m[\"tp\"]} fp={m[\"fp\"]} tn={m[\"tn\"]} fn={m[\"fn\"]}')
" 2>&1 | tee "$LOG_DIR/step10_casia_focalloss.log"

        stage_elapsed=$((SECONDS - STAGE_START))
        mark_stage_done 10
        _acc10=$(grep "Test Acc" "$LOG_DIR/step10_casia_focalloss.log" 2>/dev/null | awk '{print $3}')
        _f1_10=$(grep "F1:" "$LOG_DIR/step10_casia_focalloss.log" 2>/dev/null | awk '{print $2}')
        save_stage_result 10 "completed" $stage_elapsed "{\"casia_v2_acc\": ${_acc10:-0}, \"f1\": ${_f1_10:-0}}"
        log "阶段10 完成 ⏱ $(format_time $stage_elapsed)"

    fi

    # ---- 阶段完成显示 ----
    stage_elapsed=$((SECONDS - STAGE_START))
    log "阶段 ${stage}/${TOTAL_STAGES} [${STAGE_NAME}] 完成 ⏱ $(format_time $stage_elapsed)"
    echo ""
done

# ============================================================
# 完成
# ============================================================
header "🎉 全部实验完成!"
total_elapsed=$((SECONDS - GLOBAL_START_SEC))
log "总耗时: $(format_time $total_elapsed)"
log "结果汇总: $SUMMARY_FILE"
log "详细日志: $LOG_DIR/"
echo ""
log "关键结果预览:"
if [ -f "$SUMMARY_FILE" ]; then
    head -30 "$SUMMARY_FILE"
fi
