import gradio as gr
import torch
import numpy as np
import cv2
import pydicom
import albumentations as A
from albumentations.pytorch import ToTensorV2
from configs.configs import get_config  
from src.models.classifier import MedicalFusionClassifier
from src.models.segmentor import build_stage2_segmentor
import joblib
# ---------------------------------------------------------
# 1. Global Setup & Model Loading
# ---------------------------------------------------------
cfg = get_config()

device = torch.device(cfg.device)
classifier_model = MedicalFusionClassifier(
    backbone_name=cfg.model.classifier_backbone, 
    num_meta_features=cfg.model.num_meta_features
).to(device)
classifier_model.load_state_dict(
    torch.load(cfg.model_path.classifier_checkpoint, map_location=device)
)

segmentor_model = build_stage2_segmentor().to(device) # If build_stage2_segmentor takes args, use cfg.model here too
segmentor_model.load_state_dict(
    torch.load(cfg.model_path.segmentor_checkpoint, map_location=device)
)


classifier_model.eval()
segmentor_model.eval()

INFERENCE_SIZE = 1024 
best_threshold = 0.45 # Default to 0.5 if not found

# Your validation transformer
val_transform = A.Compose([
    A.Resize(INFERENCE_SIZE, INFERENCE_SIZE),
    A.Normalize(mean=(0.485,), std=(0.229,)), # Single channel normalization
    ToTensorV2()
])

# ---------------------------------------------------------
# 2. Single Image Processing Logic
# ---------------------------------------------------------
def process_single_dicom(file_path):
    """Reads DICOM, fixes photometric inversion, applies Min-Max, returns 1-channel array."""
    dicom = pydicom.dcmread(file_path)
    image = dicom.pixel_array.astype(np.float32)
    
    if getattr(dicom, "PhotometricInterpretation", "") == "MONOCHROME1":
        image = np.amax(image) - image
        
    image_min = np.min(image)
    image_max = np.max(image)
    if image_max - image_min > 0:
        image = (image - image_min) / (image_max - image_min) * 255.0
    else:
        image = np.zeros_like(image)
        
    return image.astype(np.uint8) # Returns 2D (H, W) array

# ---------------------------------------------------------
# 3. The Inference Wrapper
# ---------------------------------------------------------
def predict_pneumothorax(dicom_file, age, sex, view_position):
    """The main function executed when the user clicks 'Submit'."""
    
    # 1. Process Image
    raw_image = process_single_dicom(dicom_file.name)
    
    # 2. Apply Albumentations & prepare tensor
    augmented = val_transform(image=raw_image)
    img_tensor = augmented['image'].unsqueeze(0).to(device) # Shape: (1, 1, 1024, 1024)
    
    # 3. Prepare Metadata Tensor
    # Assuming Sex: M=0, F=1 and ViewPosition: AP=0, PA=1 based on standard encoding
    scaler = joblib.load(cfg.paths.processed_dir + "/meta_scaler.joblib")
    sex_val = 1.0 if sex == "M" else 0.0
    view_val = 1.0 if view_position == "AP" else 0.0
    scaled = scaler.transform([[float(age), sex_val, view_val]])
    meta_tensor = torch.tensor(scaled, dtype=torch.float32).to(device)
  
    # 4. Run Stage 1: Classifier
    with torch.no_grad():
        cls_logits = classifier_model(img_tensor, meta_tensor)
        cls_prob = torch.sigmoid(cls_logits).item()
        
    # Create a 3-channel RGB version of the raw image purely for the UI visualization
    display_img = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2RGB)
    
    # 5. The Gating Logic
    if cls_prob <= best_threshold:
        return display_img, f"Healthy (Confidence: {(1 - cls_prob)*100:.1f}%)"
        
    # 6. Run Stage 2: Segmentor (Only if positive)
    with torch.no_grad():
        seg_logits = segmentor_model(img_tensor)
        seg_probs = torch.sigmoid(seg_logits)
        pred_mask = (seg_probs > 0.5).cpu().numpy().squeeze()
        
    # Apply your exact inverse transformations
    mask = np.flip(pred_mask, axis=1)
    mask = np.rot90(mask, 1)
    mask = np.ascontiguousarray(mask)
    mask_visual = (mask.astype(float) * 255).astype(np.uint8)
    
    # 7. Create Red Overlay for the UI
    overlay = display_img.copy()
    overlay[mask_visual == 255] = [255, 0, 0] # Highlight pneumothorax in pure red
    final_output = cv2.addWeighted(display_img, 0.7, overlay, 0.3, 0)
    
    return final_output, f"Pneumothorax Detected! (Confidence: {cls_prob*100:.1f}%)"

# ---------------------------------------------------------
# 4. Gradio UI Definition
# ---------------------------------------------------------
ui = gr.Interface(
    fn=predict_pneumothorax,
    inputs=[
        gr.File(label="Upload DICOM X-Ray (.dcm)"),
        gr.Number(label="Patient Age", value=45),
        gr.Dropdown(choices=["M", "F"], label="Patient Sex", value="M"),
        gr.Dropdown(choices=["AP", "PA"], label="View Position", value="PA")
    ],
    outputs=[
        gr.Image(label="Segmentation Result"),
        gr.Textbox(label="Diagnosis & Confidence")
    ],
    title="SIIM-ACR Pneumothorax Diagnostic Assistant",
    description="Upload a 1-channel DICOM chest X-ray and provide clinical metadata. The Two-Stage pipeline will classify the image and highlight collapsed lungs in red if detected."
)

if __name__ == "__main__":
    ui.launch() # Add share=True inside launch() if you want a public URL