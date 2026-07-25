"""
main.py
=======
End-to-end training entry point that reproduces the data-prep + training
flow that was actually *executed* in `notebook/pneumothorax-seg-model (7).ipynb`:

  1. Read the Stage-1 DICOMs + train-rle.csv and build train_df / test_df
     (encoding Sex/ViewPosition, scaling Age/Sex/ViewPosition), then
     re-label test_df with the Stage-2 ground truth
     -> src/data/prepare.py:load_stage1_dataframes()

  2. Split 80/20 off the labeled test_df and fold the 80% into train_df to
     form `combined_df` (the notebook's "Train on complete Stage1 data,
     no validation" approach). The 20% (`val_df`) is held out, matching
     the notebook -- it isn't used for training here either.

  3. Train the classifier on combined_df -> src/training/train_cls_seg.py:classifier()
  4. Train the segmentor on combined_df  -> src/training/train_cls_seg.py:segmentor()

Both training functions save their own checkpoints
(best_classifier.pth / best_segmentor.pth) into the current working
directory, exactly like the notebook did.

Before running, set your dataset locations (defaults shown, see configs/configs.py):
    export PNX_DATASET_ROOT=data/raw/siim-acr-pneumothorax-segmentation-data/pneumothorax
    export PNX_STAGE2_ROOT=data/raw/siim-acr-pneumothorax-segmentation
(scripts/download_data.sh will download the data into these exact paths.)

Usage:
    python main.py
    python main.py --classifier-epochs 6 --segmentor-epochs 10
"""
import argparse
import gc

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from configs.configs import get_config
from src.data.prepare import load_stage1_dataframes
from src.training.train_cls_seg import classifier, segmentor
from src.utils.utils import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Train the pneumothorax classifier + segmentor (notebook parity).")
    parser.add_argument("--classifier-epochs", type=int, default=6,
                         help="Matches the notebook's classifier(combined_df, device=device, epoch=6) call.")
    parser.add_argument("--segmentor-epochs", type=int, default=10,
                         help="Matches the notebook's segmentor(combined_df, device=device, epoch=10) call.")
    parser.add_argument("--seed", type=int, default=1,
                         help="The notebook calls seed_everything(1) (note: configs.py's default train seed is 42).")
    parser.add_argument("--val-size", type=float, default=0.2,
                         help="Fraction of the labeled Stage-1 test set held out as val_df (unused for training, "
                              "kept only for parity with the notebook, which computes it but doesn't train on it).")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = get_config()
    device = torch.device(cfg.device)
    print(f"Using device: {device}")

    seed_everything(args.seed)

    # ------------------------------------------------------------------
    # 1) Read + prepare the data (identical to the notebook's data-prep cells)
    # ------------------------------------------------------------------
    train_df, test_df, masks = load_stage1_dataframes(cfg)

    # ------------------------------------------------------------------
    # 2) Build combined_df exactly like the notebook: fold 80% of the
    #    (now Stage-2-labeled) Stage-1 test split into the training data;
    #    hold out the remaining 20% as val_df.
    # ------------------------------------------------------------------
    combine_with_train, val_df = train_test_split(
        test_df, test_size=args.val_size, random_state=42, stratify=test_df["class"]
    )
    combined_df = pd.concat([train_df, combine_with_train], ignore_index=True)
    print(
        f"len combined_df (train+test): {len(combined_df)} | "
        f"len val_df: {len(val_df)} | first 10 index of val_df {val_df.head(10).index.tolist()}"
    )

    # ------------------------------------------------------------------
    # 3) Stage 1: train the classifier on the full combined dataframe.
    #    Saves best_classifier.pth as a side effect.
    # ------------------------------------------------------------------
    classifier_model = classifier(combined_df, device=device, epoch=args.classifier_epochs)

    # VRAM cleanup between stages, matching the notebook
    classifier_model.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 4) Stage 2: train the segmentor on the same combined dataframe
    #    (internally filtered down to the positive/pneumothorax cases).
    #    Saves best_segmentor.pth as a side effect.
    # ------------------------------------------------------------------
    segmentor_model = segmentor(combined_df, device=device, epoch=args.segmentor_epochs)

    print("Done. Checkpoints saved to best_classifier.pth / best_segmentor.pth in the current directory.")
    return classifier_model, segmentor_model


if __name__ == "__main__":
    main()