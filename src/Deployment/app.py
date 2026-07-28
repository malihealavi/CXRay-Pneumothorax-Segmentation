"""
SIIM-ACR Pneumothorax Diagnostic Assistant
===========================================
Two-stage inference pipeline for chest X-ray DICOM images:
  Stage 1 - Binary classifier (EfficientNet-B3 + metadata fusion) gates healthy vs. positive.
  Stage 2 - Segmentation model (U-Net / EfficientNet-B4) runs only on images flagged positive,
            producing a pixel-level pneumothorax mask.

Weights are pulled from the Hugging Face Hub repo `MaliheAlavi/Pneomothorax_model_weights`.
"""

import cv2
import numpy as np
import pydicom
import torch
import torch.nn as nn
import timm
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from huggingface_hub import hf_hub_download
import gradio as gr
import spaces
import joblib
from sklearn.preprocessing import StandardScaler
# =============================================================================
# Configuration
# =============================================================================

REPO_ID = "MaliheAlavi/Pneomothorax_model_weights"
CLASSIFIER_FILENAME = "best_classifier.pth"
SEGMENTOR_FILENAME = "best_segmentor.pth"
SCALER_NAME = "metadata_scaler.pkl"

INFERENCE_SIZE = 1024
CLASSIFIER_THRESHOLD = 0.45  # Tuned on validation F2 curve
MASK_PROB_THRESHOLD = 0.5

IMAGENET_MEAN = (0.485,)  # single-channel normalization
IMAGENET_STD = (0.229,)

SEX_ENCODING = {"M": 1, "F": 0}
VIEW_ENCODING = {"AP": 1, "PA": 0}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Model Definitions
# =============================================================================

class MedicalFusionClassifier(nn.Module):
    """Stage 1: image encoder (timm backbone) fused with a small metadata MLP."""

    def __init__(self, backbone_name: str = "efficientnet_b3", num_meta_features: int = 3):
        super().__init__()

        self.image_encoder = timm.create_model(
            backbone_name,
            pretrained=True,
            in_chans=1,
            num_classes=0,
        )
        num_image_features = self.image_encoder.num_features

        self.meta_encoder = nn.Sequential(
            nn.Linear(num_meta_features, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(num_image_features + 16, 1)

    def forward(self, image: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        img_features = self.image_encoder(image)
        meta_features = self.meta_encoder(metadata)
        fused = torch.cat((img_features, meta_features), dim=1)
        return self.classifier(fused)


def build_segmentor() -> nn.Module:
    """Stage 2: U-Net with an EfficientNet-B4 encoder, single-channel in/out."""
    return smp.Unet(
        encoder_name="timm-efficientnet-b4",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
        activation=None,  # raw logits for numerical stability
    )


# =============================================================================
# Model Loading
# =============================================================================

def load_models():
    """Download weights from the Hub (if not cached) and load them into fresh models."""
    classifier = MedicalFusionClassifier(backbone_name="efficientnet_b3", num_meta_features=3).to(DEVICE)
    segmentor = build_segmentor().to(DEVICE)

    classifier_path = hf_hub_download(repo_id=REPO_ID, filename=CLASSIFIER_FILENAME)
    segmentor_path = hf_hub_download(repo_id=REPO_ID, filename=SEGMENTOR_FILENAME)

    classifier.load_state_dict(torch.load(classifier_path, map_location=DEVICE))
    segmentor.load_state_dict(torch.load(segmentor_path, map_location=DEVICE))

    classifier.eval()
    segmentor.eval()

    return classifier, segmentor

def load_scaler():
    scaler_path = hf_hub_download(repo_id=REPO_ID, filename=SCALER_NAME)
    return joblib.load(scaler_path)

classifier_model, segmentor_model = load_models()
scaler = load_scaler()

val_transform = A.Compose([
    A.Resize(INFERENCE_SIZE, INFERENCE_SIZE),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])


# =============================================================================
# Image / DICOM Preprocessing
# =============================================================================

def load_dicom_as_uint8(file_path: str) -> np.ndarray:
    """Read a DICOM file, correct MONOCHROME1 inversion, and min-max scale to uint8."""
    dicom = pydicom.dcmread(file_path)
    image = dicom.pixel_array.astype(np.float32)

    if getattr(dicom, "PhotometricInterpretation", "") == "MONOCHROME1":
        image = np.amax(image) - image

    image_min, image_max = np.min(image), np.max(image)
    if image_max - image_min > 0:
        image = (image - image_min) / (image_max - image_min) * 255.0
    else:
        image = np.zeros_like(image)

    return image.astype(np.uint8)  # (H, W)


def build_metadata_tensor(age: float, sex: str, view_position: str) -> torch.Tensor:
    """Encode patient metadata into the tensor shape expected by the classifier."""
    sex_val = SEX_ENCODING.get(sex, 0.0)
    view_val = VIEW_ENCODING.get(view_position, 0.0)
    scaled_meta = scaler.transform([[float(age), sex_val, view_val]])
    return torch.tensor(scaled_meta, dtype=torch.float32).to(DEVICE)


def postprocess_mask(seg_logits: torch.Tensor) -> np.ndarray:
    """Threshold segmentation logits and undo the orientation transform applied at train time."""
    seg_probs = torch.sigmoid(seg_logits)
    pred_mask = (seg_probs > MASK_PROB_THRESHOLD).cpu().numpy().squeeze()
    mask = np.ascontiguousarray(pred_mask)
    return (mask.astype(np.float32) * 255).astype(np.uint8)


def draw_overlay(display_img: np.ndarray, mask_visual: np.ndarray) -> np.ndarray:
    """Blend a red mask overlay onto the RGB display image."""
    overlay = display_img.copy()
    overlay[mask_visual == 255] = [255, 0, 0]
    return cv2.addWeighted(display_img, 0.7, overlay, 0.3, 0)


# =============================================================================
# Inference Pipeline
# =============================================================================

@spaces.GPU
def predict_pneumothorax(dicom_file, age: float, sex: str, view_position: str):
    """Run the full two-stage pipeline on a single uploaded DICOM file."""

    # 1. Load & preprocess image
    raw_image = load_dicom_as_uint8(dicom_file.name)
    img_tensor = val_transform(image=raw_image)["image"].unsqueeze(0).to(DEVICE)
    display_img = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2RGB)

    # 2. Stage 1: classification
    meta_tensor = build_metadata_tensor(age, sex, view_position)
    with torch.no_grad():
        cls_prob = torch.sigmoid(classifier_model(img_tensor, meta_tensor)).item()

    if cls_prob <= CLASSIFIER_THRESHOLD:
        return display_img, f"Healthy (Confidence: {(1 - cls_prob) * 100:.1f}%)"

    # 3. Stage 2: segmentation (only reached if flagged positive)
    with torch.no_grad():
        seg_logits = segmentor_model(img_tensor)
    mask_visual = postprocess_mask(seg_logits)
    final_output = draw_overlay(display_img, mask_visual)

    return final_output, f"Pneumothorax Detected! (Confidence: {cls_prob * 100:.1f}%)"


# =============================================================================
# Gradio UI
# =============================================================================

def build_ui() -> gr.Interface:
    return gr.Interface(
        fn=predict_pneumothorax,
        inputs=[
            gr.File(label="Upload DICOM X-Ray (.dcm)"),
            gr.Number(label="Patient Age", value=45),
            gr.Dropdown(choices=["M", "F"], label="Patient Sex", value="M"),
            gr.Dropdown(choices=["AP", "PA"], label="View Position", value="PA"),
        ],
        outputs=[
            gr.Image(label="Segmentation Result"),
            gr.Textbox(label="Diagnosis & Confidence"),
        ],
        title="SIIM-ACR Pneumothorax Diagnostic Assistant",
        description=(
            "Upload a 1-channel DICOM chest X-ray and provide clinical metadata. "
            "The two-stage pipeline classifies the image and highlights collapsed "
            "lungs in red if detected."
        ),
    )


if __name__ == "__main__":
    ui = build_ui()
    ui.launch()  # add share=True for a public URL