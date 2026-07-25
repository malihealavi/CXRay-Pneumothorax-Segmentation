
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

def build_stage2_segmentor():
    """
    Stage 2: Segmentation Model
    Predicts: The pixel-level mask of the pneumothorax
    Encoder: SE-ResNeXt-50 (Pre-trained on ImageNet)
    Decoder: U-Net
    """
    model = smp.Unet(
        encoder_name="timm-efficientnet-b4",
        # encoder_name="se_resnext50_32x4d", # first 
        encoder_weights="imagenet",        # Use ImageNet initialization
        in_channels=1,                     # Input must be 3 channels
        classes=1,                         # 1 output channel for the binary mask
        activation=None                    # Return raw logits for numeric stability
    )
    return model

# Initialize the models
