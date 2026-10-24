#!/bin/bash
# ============================================================
# 数据集下载脚本 — ForensicGuard
# 用法: bash download_datasets.sh
# ============================================================
set -e

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log()     { echo -e "${GREEN}[下载]${NC} $1"; }
warn()    { echo -e "${YELLOW}[注意]${NC} $1"; }
error()   { echo -e "${RED}[错误]${NC} $1"; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
mkdir -p "$DATA_DIR"

# ============================================================
# 1. CASIA v2 — 已有
# ============================================================
log "CASIA v2: $(ls $DATA_DIR/casia2/CASIA2.0_revised/Au/*.jpg 2>/dev/null | wc -l) 张"
if [ ! -d "$DATA_DIR/casia2/CASIA2.0_revised" ]; then
    warn "CASIA v2 数据集需要手动放置到 $DATA_DIR/casia2/"
    warn "下载地址: http://forensics.idealtest.org/"
fi

# ============================================================
# 2. Columbia — 已有
# ============================================================
if [ -d "$DATA_DIR/columbia/ImSpliceDataset" ]; then
    log "Columbia: $(ls $DATA_DIR/columbia/ImSpliceDataset/*/*.jpg 2>/dev/null | wc -l) 张"
else
    warn "Columbia 数据集需要手动放置"
    warn "下载地址: https://www.ee.columbia.edu/ln/dvmm/downloads/"
fi

# ============================================================
# 3. APRICOT / APRICOT-Mask — 空目录
# ============================================================
log "\n═══════════════════════════════════════════"
log " APRICOT / APRICOT-Mask 数据集下载指南"
log "═══════════════════════════════════════════"
echo ""
echo "APRICOT-Mask 是 APRICOT 物理对抗补丁数据集的像素级标注增强版。"
echo "它包含 138 张 dev 图像 + 873 张 test 图像，每张都有像素级补丁 mask。"
echo ""
echo "【方法1: FTP 下载 (推荐)】"
echo "  在能访问 FTP 的环境中执行:"
echo "    wget -r ftp://ftp.cis.jhu.edu/pub/apricot-mask"
echo "  下载后把 .pt 文件放到 $DATA_DIR/apricot/"
echo ""
echo "【方法2: OneDrive 下载】"
echo "  在浏览器中打开:"
echo "  https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/jliu214_jh_edu/EnZTQY21vGRMsrRY03cD9HYBYlYSgzT-7wzAAkMo6LiozA?e=2x4ErW"
echo "  下载所有 .pt 文件到 $DATA_DIR/apricot/"
echo ""
echo "【方法3: Google Drive (SAC预训练模型+数据)】"
echo "  https://drive.google.com/drive/folders/1o9Ftkh6ecR2DcoRL3ae3ZjFeYOsXBCdp"
echo ""

# ============================================================
# 4. NIST16 (MFC2016) — 空目录
# ============================================================
log "\n═══════════════════════════════════════════"
log " NIST16 (MFC2016) 数据集下载指南"
log "═══════════════════════════════════════════"
echo ""
echo "NIST16 是 NIST Nimble Challenge 2016 的图像篡改检测数据集。"
echo ""
echo "【下载方法】"
echo "  方案A: NIST 官网 (需要申请)"
echo "    https://www.nist.gov/itl/iad/mig/nimble-challenge-2016-evaluation"
echo "    可能需要联系 NIST 工作人员获取下载权限。"
echo ""
echo "  方案B: 学术镜像"
echo "    https://cvlab.epfl.ch/research/research-projects/trustai/"
echo "    (部分学术站点提供镜像)"
echo ""
echo "  将下载的图片放置到: $DATA_DIR/nist16/"
echo "  目录结构: $DATA_DIR/nist16/ (子目录按分类组织)"
echo ""

# ============================================================
# 5. COCO — 自动下载
# ============================================================
log "\n═══════════════════════════════════════════"
log " COCO val2017 (自动下载)"
log "═══════════════════════════════════════════"
COCO_DIR="$DATA_DIR/coco"
if [ ! -d "$COCO_DIR/val2017" ] || [ $(ls "$COCO_DIR/val2017"/*.jpg 2>/dev/null | wc -l) -lt 100 ]; then
    log "正在下载 COCO val2017 (~1GB)..."
    mkdir -p "$COCO_DIR"
    wget -q --show-progress http://images.cocodataset.org/zips/val2017.zip -O /tmp/val2017.zip
    unzip -q /tmp/val2017.zip -d "$COCO_DIR"
    rm /tmp/val2017.zip
    log "COCO val2017: $(ls $COCO_DIR/val2017/*.jpg 2>/dev/null | wc -l) 张"
else
    log "COCO val2017: $(ls $COCO_DIR/val2017/*.jpg 2>/dev/null | wc -l) 张 (已存在)"
fi

# ============================================================
# 汇总
# ============================================================
log "\n═══════════════════════════════════════════"
log " 数据集状态汇总"
log "═══════════════════════════════════════════"
echo ""
for ds in "casia2:$(ls $DATA_DIR/casia2/CASIA2.0_revised/Au/*.jpg 2>/dev/null | wc -l)" \
          "columbia:$(ls $DATA_DIR/columbia/ImSpliceDataset/*/*.jpg 2>/dev/null | wc -l)" \
          "coco:$(ls $DATA_DIR/coco/val2017/*.jpg 2>/dev/null | wc -l)" \
          "apricot:$(ls $DATA_DIR/apricot/*.pt 2>/dev/null | wc -l)" \
          "nist16:$(ls $DATA_DIR/nist16/*.jpg 2>/dev/null | wc -l)"; do
    name="${ds%%:*}"
    count="${ds##*:}"
    if [ "$count" -eq 0 ] 2>/dev/null || [ "$count" = "0" ]; then
        echo "  ⚠️  $name: 空 (需手动下载)"
    else
        echo "  ✅ $name: $count 文件"
    fi
done
echo ""
log "完成! 如有缺失数据集, 请按上方指南手动下载。"
