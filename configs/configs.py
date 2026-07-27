
from dataclasses import dataclass, field
import os
import torch

def _get_default_device() -> str:
    """Returns 'cuda' if a GPU is available, otherwise 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class DataPaths:
    # Root of the SIIM-ACR pneumothorax dataset (stage-1 style DICOM tree).
    dataset_root: str = _env(
        "PNX_DATASET_ROOT",
        "data/raw/siim-acr-pneumothorax-segmentation-data/pneumothorax",
    )
    train_dicom_glob: str = "dicom-images-train/*/*/*.dcm"
    test_dicom_glob: str = "dicom-images-test/*/*/*.dcm"
    train_rle_csv: str = "train-rle.csv"

    # Stage-2 competition data (used for the held-out leaderboard test set).
    stage2_root: str = _env(
        "PNX_STAGE2_ROOT", "data/raw/siim-acr-pneumothorax-segmentation"
    )
    stage2_train_csv: str = "stage_2_train.csv"
    stage2_test_dicom_glob: str = "stage_2_images/*.dcm"
    stage2_sample_submission: str = "stage_2_sample_submission.csv"

    # Where prepared dataframes / checkpoints / plots / submissions go.
    processed_dir: str = _env("PNX_PROCESSED_DIR", "data/processed")
    checkpoint_dir: str = _env("PNX_CHECKPOINT_DIR", "outputs/checkpoints")
    output_dir: str = _env("PNX_OUTPUT_DIR", "outputs")


@dataclass
class ModelPath:
    classifier_checkpoint: str =  "best_classifier.pth"
    segmentor_checkpoint: str =  "best_segmentor.pth"

@dataclass
class ModelConfig:
    classifier_backbone: str = "efficientnet_b3"
    num_meta_features: int = 3  # Age, Sex, ViewPosition
    # Any timm-compatible SMP encoder works here, e.g. "se_resnext50_32x4d".
    segmentor_encoder: str = "timm-efficientnet-b4"
    segmentor_encoder_weights: str = "imagenet"

@dataclass
class TrainConfig:
    seed: int = 42
    classifier_epochs: int = 12
    segmentor_epochs: int = 15
    classifier_batch_size: int = 32
    segmentor_batch_size: int = 32
    classifier_lr: float = 5e-4
    segmentor_lr: float = 1e-4
    early_stopping_patience: int = 4
    device: str = field(default_factory=_get_default_device)


@dataclass
class InferenceConfig:
    image_size: int = 1024
    classification_threshold: float = 0.45
    segmentation_threshold: float = 0.5



@dataclass
class Config:
    paths: DataPaths = field(default_factory=DataPaths)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    model_path: ModelPath = field(default_factory=ModelPath)
    device: str = field(default_factory=_get_default_device)

def get_config() -> Config:
    return Config()