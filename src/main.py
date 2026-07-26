
import argparse
import gc

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from configs.configs import get_config
from src.data.prepare import load_stage1_dataframes
from src.training.train_cls_seg import classifier, segmentor
from src.training.train_classifier import classifier_train
from src.training.train_segmentor import segmentor_train
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
    classifier_model = classifier_train(combined_df, val_df, epoch=args.classifier_epochs, device=device )
    # VRAM cleanup between stages, matching the notebook
    classifier_model.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 4) Stage 2: train the segmentor on the same combined dataframe
    #    (internally filtered down to the positive/pneumothorax cases).
    #    Saves best_segmentor.pth as a side effect.
    # ------------------------------------------------------------------
    segmentor_model = segmentor_train(combined_df, val_df, epoch=args.classifier_epochs, device=device )
    print("Done. Checkpoints saved to best_classifier.pth / best_segmentor.pth in the current directory.")
    return classifier_model, segmentor_model


if __name__ == "__main__":
    main()