#!/usr/bin/env bash
# download_weights.sh
# Downloads classifier + segmentor checkpoints from Kaggle into ./
# Dataset: https://www.kaggle.com/datasets/malihealavi/pneumothorax-classifier
#
# Requires: pip install kaggle, and credentials via
#   ~/.kaggle/kaggle.json (chmod 600) OR KAGGLE_USERNAME + KAGGLE_KEY env vars.
#
# Usage: chmod +x download_weights.sh && ./download_weights.sh

set -euo pipefail

DATASET_SLUG="malihealavi/pneumothorax-classifier"
CLASSIFIER_FILE="best_classifier.pth"
SEGMENTOR_FILE="best_segmentor.pth"
DEST_DIR="."

mkdir -p "$DEST_DIR"

if ! command -v kaggle &> /dev/null; then
    echo "ERROR: 'kaggle' CLI not found. Install it with: pip install kaggle" >&2
    exit 1
fi

if ! command -v unzip &> /dev/null; then
    echo "ERROR: 'unzip' not found. Install it (e.g. apt-get install unzip)." >&2
    exit 1
fi

if [[ ! -f "${HOME}/.kaggle/kaggle.json" ]] && { [[ -z "${KAGGLE_USERNAME:-}" ]] || [[ -z "${KAGGLE_KEY:-}" ]]; }; then
    echo "ERROR: No Kaggle credentials found." >&2
    echo "Place your API token at ~/.kaggle/kaggle.json (chmod 600)," >&2
    echo "or export BOTH KAGGLE_USERNAME and KAGGLE_KEY." >&2
    exit 1
fi

download_file() {
    local filename="$1"
    echo "==> Downloading ${filename} ..."

    if ! kaggle datasets download -d "$DATASET_SLUG" -f "$filename" -p "$DEST_DIR" -o; then
        echo "ERROR: download failed for ${filename}." >&2
        echo "If this is a 401/403 with a valid-looking token, it may be a new-format" >&2
        echo "Kaggle token the legacy CLI can't parse. Try 'pip install kagglehub' instead," >&2
        echo "or regenerate an older-format token at https://www.kaggle.com/settings." >&2
        exit 1
    fi

    # A single-file download is sometimes wrapped in a zip named after the file.
    local zip_path="${DEST_DIR}/${filename}.zip"
    if [[ -f "$zip_path" ]]; then
        unzip -o -q "$zip_path" -d "$DEST_DIR"
        rm -f "$zip_path"
    fi
}

download_file "$CLASSIFIER_FILE"
download_file "$SEGMENTOR_FILE"

echo ""
echo "==> Verifying downloads ..."
for f in "$CLASSIFIER_FILE" "$SEGMENTOR_FILE"; do
    if [[ -f "$DEST_DIR/$f" ]]; then
        size=$(du -h "$DEST_DIR/$f" | cut -f1)
        echo "    $DEST_DIR/$f  ($size)"
    else
        echo "    WARNING: $DEST_DIR/$f not found after download." >&2
    fi
done

echo "==> Done."