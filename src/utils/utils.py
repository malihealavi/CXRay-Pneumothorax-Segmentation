import random
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import cv2
import pydicom


def calculate_dice_coef(preds, targets, smooth=1e-6):
    """
    Calculates the Dice Coefficient for a batch of binarized predictions and targets.
    """
    # Flatten tensors
    preds = preds.contiguous().view(-1)
    targets = targets.contiguous().view(-1)
    
    intersection = (preds * targets).sum()
    dice = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    return dice.item()

def plot_loss(val_losses, train_losses, path):
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, label='Training Loss', color='blue', marker='o', linewidth=2)
    plt.plot(epochs, val_losses, label='Validation Loss', color='red', marker='s', linewidth=2)
    
    plt.title('Training and Validation Loss Over Epochs', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    
    plt.xticks(epochs)
    
    plt.legend(fontsize=11)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plot_path = path
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    
    print(f"Plot successfully saved to {os.path.abspath(plot_path)}")


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# seed_everything()


def custom_threshold(all_targets, all_probs, target_recall=0.90):
    """Finds the highest threshold that maintains at least the target recall."""
    precisions, recalls, thresholds = precision_recall_curve(all_targets, all_probs)
    
    valid = np.where(recalls[:-1] >= target_recall)[0]
    
    best_threshold = thresholds[valid[-1]] if len(valid) > 0 else thresholds[0]
    
    return best_threshold


def f2_best_threshold(all_targets, all_probs):
    """Finds the threshold that maximizes the F2 score (favors recall)."""
    precisions, recalls, thresholds = precision_recall_curve(all_targets, all_probs)
    p = precisions[:-1]
    r = recalls[:-1]
    
    # Safely calculate F2 scores for all thresholds
    with np.errstate(divide='ignore', invalid='ignore'):
        f2_scores = (5 * p * r) / ((4 * p) + r)
        f2_scores = np.nan_to_num(f2_scores)

    best_f2_idx = np.argmax(f2_scores)
    best_threshold_f2 = thresholds[best_f2_idx] # Fixed NameError
    
    return best_threshold_f2, f2_scores[best_f2_idx]


def gmean_best_threshold(all_targets, all_probs):
    """Finds the threshold that maximizes the G-Mean using ROC curve metrics."""
    fpr, tpr, roc_thresholds = roc_curve(all_targets, all_probs)
    
    # Formula: G-Mean = sqrt(Sensitivity * Specificity) = sqrt(tpr * (1 - fpr))
    g_means = np.sqrt(tpr * (1 - fpr))
    
    # Locate the threshold that maximizes G-Mean
    best_gmean_idx = np.argmax(g_means)
    best_threshold_gmean = roc_thresholds[best_gmean_idx]
    
    return best_threshold_gmean, g_means[best_gmean_idx]


def pr_auc_score(all_targets, all_probs):
    """Calculates the Area Under the Precision-Recall Curve (holistic metric, not a threshold)."""
    precisions, recalls, _ = precision_recall_curve(all_targets, all_probs)
    pr_auc = auc(recalls, precisions)
    
    return pr_auc


def process_dicom_to_3channel(dicom_path):
    """
    Reads a DICOM file, applies critical medical image normalization,
    and converts it to a 3-channel RGB image.
    """
    # 1. Read the DICOM file
    dicom = pydicom.dcmread(dicom_path)
    image = dicom.pixel_array.astype(np.float32)
    
    # 2. Photometric Interpretation Check (Fix Inverted X-Rays)
    # Some DICOMs are MONOCHROME1 (Air is White, Bones are Black).
    # We must invert them to MONOCHROME2 so the model gets consistent data.
    if getattr(dicom, "PhotometricInterpretation", "") == "MONOCHROME1":
        image = np.amax(image) - image
        
    # 3. Min-Max Scaling (Crucial for 16-bit to 8-bit conversion)
    # Native DICOMs can be up to 16-bit. 
    # Scaling protects the subtle visual details (like faint pleural lines) from being clipped or ruined.
    image_min = np.min(image)
    image_max = np.max(image)
    if image_max - image_min > 0:
        image = (image - image_min) / (image_max - image_min) * 255.0
    else:
        image = np.zeros_like(image)
        
    image = image.astype(np.uint8)
    
    # 4. Replicate Over 3 Channels (Grayscale -> RGB)
    # This efficiently duplicates the grayscale data identically into 3 layers (R=G=B)
    # to perfectly satisfy pre-trained ImageNet backbones (in_channels=3).
    image_3ch = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    return image_3ch


def mask2rle(img, width, height):
    rle = []
    lastColor = 0;
    currentPixel = 0;
    runStart = -1;
    runLength = 0;

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 255:
                    runStart = currentPixel;
                    runLength = 1;
                else:
                    rle.append(str(runStart));
                    rle.append(str(runLength));
                    runStart = -1;
                    runLength = 0;
                    currentPixel = 0;
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor;
            currentPixel+=1;

    return " ".join(rle)

def rle2mask(rle, width, height):
    mask= np.zeros(width* height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(width, height)