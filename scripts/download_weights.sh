#!/usr/bin/env bash
# ============================================================
# download_weights.sh
#
# Downloads the classifier + segmentor checkpoints from Kaggle
# into the exact paths expected by configs/config.py (ModelPath):
#
#   Kaggle dataset: malihealavi/pneumothorax-classifier
#     https://www.kaggle.com/datasets/malihealavi/pneumothorax-classifier
#   -> best_classifier.pth
#   -> best_segmentor.pth
#
# Requirements:
#   1. pip install kaggle
#   2. Kaggle API credentials, either:
#        a) ~/.kaggle/kaggle.json   (chmod 600)
#        b) exported KAGGLE_USERNAME and KAGGLE_KEY env vars
#
# Usage:
#   chmod +x download_weights.sh
#   ./download_weights.sh
#
# Note on versioning:
#   `kaggle datasets download` always fetches the LATEST version of
#   a dataset; the CLI has no reliably-documented flag to pin an
#   older version (e.g. version 13) if a newer one gets pushed later.
#   If exact-version reproducibility ever matters, use the Python
#   fallback commented at the bottom of this file instead.
#
# Troubleshooting:
#   If you get 401/403 errors even with a valid-looking token, this
#   is a known issue where newly-generated Kaggle tokens are
#   sometimes not parsed correctly by the legacy 'kaggle' package.
#   Try regenerating an older-format token at
#   https://www.kaggle.com/settings, or install 'kagglehub'
#   (pip install kagglehub) as an alternative downloader.
# ============================================================

set -euo pipefail

DATASET_SLUG="malihealavi/pneumothorax-classifier"

CLASSIFIER_FILE="best_classifier.pth"
SEGMENTOR_FILE="best_segmentor.pth"

DEST_DIR="."
mkdir -p "$DEST_DIR"

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

echo "==> Downloading ${CLASSIFIER_FILE} from ${DATASET_SLUG} ..."
kaggle datasets download -d "$DATASET_SLUG" -f "$CLASSIFIER_FILE" -p "$DEST_DIR" -o

echo "==> Downloading ${SEGMENTOR_FILE} from ${DATASET_SLUG} ..."
kaggle datasets download -d "$DATASET_SLUG" -f "$SEGMENTOR_FILE" -p "$DEST_DIR" -o

# When downloading a single file, the CLI sometimes wraps it in a
# zip named after the file itself. Unzip it if that happened.
for f in "$CLASSIFIER_FILE" "$SEGMENTOR_FILE"; do
    ZIP_PATH="$DEST_DIR/${f}.zip"
    if [[ -f "$ZIP_PATH" ]]; then
        unzip -o -q "$ZIP_PATH" -d "$DEST_DIR"
        rm -f "$ZIP_PATH"
    fi
done

echo ""
echo "==> Verifying downloads ..."
for f in "$CLASSIFIER_FILE" "$SEGMENTOR_FILE"; do
    if [[ -f "$DEST_DIR/$f" ]]; then
        SIZE=$(du -h "$DEST_DIR/$f" | cut -f1)
        echo "    $DEST_DIR/$f  ($SIZE)"
    else
        echo "    WARNING: $DEST_DIR/$f not found after download."
    fi
done

echo "==> Done."

# ------------------------------------------------------------
# Fallback: pin to an exact dataset version (e.g. version 13) via
# the Python Kaggle API, in case a newer version has since been
# pushed and you need reproducibility. Check your installed
# 'kaggle' package's API signature first (dataset_download_file /
# dataset_download_files), since version-pinning support has varied
# across releases:
#
# python3 - <<'PYEOF'
# from kaggle.api.kaggle_api_extended import KaggleApi
#
# api = KaggleApi()
# api.authenticate()
# api.dataset_download_file(
#     "malihealavi/pneumothorax-classifier",
#     file_name="best_classifier.pth",
#     path="checkpoints",
#     force=True,
# )
# api.dataset_download_file(
#     "malihealavi/pneumothorax-classifier",
#     file_name="best_segmentor.pth",
#     path="checkpoints",
#     force=True,
# )
# PYEOF
# ------------------------------------------------------------