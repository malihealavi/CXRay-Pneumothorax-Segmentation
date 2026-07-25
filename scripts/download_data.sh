#!/usr/bin/env bash
# ============================================================
# download_data.sh
#
# Downloads the SIIM-ACR Pneumothorax data into the exact paths
# expected by configs/config.py (DataPaths), so no manual
# download/upload is needed.
#
#   Stage 1 (DICOM tree + train-rle.csv)
#     Kaggle dataset: jesperdramsch/siim-acr-pneumothorax-segmentation-data
#     -> $PNX_DATASET_ROOT   (default: data/raw/siim-acr-pneumothorax-segmentation-data/pneumothorax)
#
#   Stage 2 (competition held-out test set)
#     Kaggle competition: siim-acr-pneumothorax-segmentation
#     -> $PNX_STAGE2_ROOT    (default: data/raw/siim-acr-pneumothorax-segmentation)
#
# Requirements:
#   1. pip install kaggle
#   2. Kaggle API credentials, either:
#        a) ~/.kaggle/kaggle.json   (chmod 600)
#        b) exported KAGGLE_USERNAME and KAGGLE_KEY env vars
#   3. You must "Join / Accept Rules" on the competition page first,
#      or the competition download will fail with a 403:
#      https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/rules
#
# Usage:
#   chmod +x download_data.sh
#   ./download_data.sh
#
# Troubleshooting:
#   If you get 401/403 errors even with a valid-looking token, this is
#   a known issue where newly-generated Kaggle tokens are sometimes not
#   parsed correctly by the legacy 'kaggle' package. Try regenerating an
#   older-format token at https://www.kaggle.com/settings, or install
#   'kagglehub' (pip install kagglehub) as an alternative downloader.
# ============================================================

set -euo pipefail

# ---- Target paths (override via env vars to match config.py) ----
DATASET_ROOT="${PNX_DATASET_ROOT:-data/raw/siim-acr-pneumothorax-segmentation-data/pneumothorax}"
STAGE2_ROOT="${PNX_STAGE2_ROOT:-data/raw/siim-acr-pneumothorax-segmentation}"

DATASET_SLUG="jesperdramsch/siim-acr-pneumothorax-segmentation-data"
COMPETITION_SLUG="siim-acr-pneumothorax-segmentation"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

echo "==> Checking for the Kaggle CLI ..."
if ! command -v kaggle &> /dev/null; then
    echo "ERROR: 'kaggle' CLI not found. Install it with: pip install kaggle"
    exit 1
fi

if [[ ! -f "${HOME}/.kaggle/kaggle.json" && -z "${KAGGLE_USERNAME:-}" ]]; then
    echo "ERROR: No Kaggle credentials found."
    echo "  Place your API token at ~/.kaggle/kaggle.json (chmod 600),"
    echo "  or export KAGGLE_USERNAME and KAGGLE_KEY before running this script."
    exit 1
fi

# ------------------------------------------------------------
# 1) Stage-1 dataset: DICOM tree + train-rle.csv
# ------------------------------------------------------------
echo "==> Downloading Stage-1 dataset: ${DATASET_SLUG}"
mkdir -p "$TMP_DIR/stage1"
kaggle datasets download -d "$DATASET_SLUG" -p "$TMP_DIR/stage1" --unzip

# The mirror wraps everything in a "pneumothorax/" folder.
# Handle wrapped vs. flat layout so this doesn't silently break if
# the mirror's structure ever changes.
if [[ -d "$TMP_DIR/stage1/pneumothorax" ]]; then
    SRC="$TMP_DIR/stage1/pneumothorax"
else
    SRC="$TMP_DIR/stage1"
fi

mkdir -p "$DATASET_ROOT"
echo "==> Moving Stage-1 data into: $DATASET_ROOT"
shopt -s dotglob
mv "$SRC"/* "$DATASET_ROOT"/
shopt -u dotglob

echo "    dicom-images-train: $(find "$DATASET_ROOT/dicom-images-train" -name '*.dcm' 2>/dev/null | wc -l) files"
echo "    dicom-images-test:  $(find "$DATASET_ROOT/dicom-images-test" -name '*.dcm' 2>/dev/null | wc -l) files"
echo "    train-rle.csv present: $(test -f "$DATASET_ROOT/train-rle.csv" && echo yes || echo NO)"

# ------------------------------------------------------------
# 2) Stage-2 competition data (held-out leaderboard test set)
# ------------------------------------------------------------
echo "==> Downloading Stage-2 competition data: ${COMPETITION_SLUG}"
mkdir -p "$TMP_DIR/stage2"
kaggle competitions download -c "$COMPETITION_SLUG" -p "$TMP_DIR/stage2"

ZIP_FILE=$(find "$TMP_DIR/stage2" -maxdepth 1 -name '*.zip' | head -n 1)
if [[ -n "$ZIP_FILE" ]]; then
    unzip -q "$ZIP_FILE" -d "$TMP_DIR/stage2"
    rm -f "$ZIP_FILE"
fi

mkdir -p "$STAGE2_ROOT"
echo "==> Moving Stage-2 data into: $STAGE2_ROOT"
shopt -s dotglob
mv "$TMP_DIR/stage2"/* "$STAGE2_ROOT"/
shopt -u dotglob

echo "    stage_2_images/:                       $(find "$STAGE2_ROOT/stage_2_images" -name '*.dcm' 2>/dev/null | wc -l) files"
echo "    stage_2_train.csv present:              $(test -f "$STAGE2_ROOT/stage_2_train.csv" && echo yes || echo NO)"
echo "    stage_2_sample_submission.csv present:  $(test -f "$STAGE2_ROOT/stage_2_sample_submission.csv" && echo yes || echo NO)"

echo ""
echo "==> Done. Data is ready under:"
echo "    $DATASET_ROOT"
echo "    $STAGE2_ROOT"