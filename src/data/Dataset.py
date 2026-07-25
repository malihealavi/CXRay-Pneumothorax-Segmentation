import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from src.data.preprocessing import normalizer
from src.utils.utils import rle2mask


class PneumothoraxDataset(Dataset):
    def __init__(self, df, augmentations=None):
        
        self.df = df
        self.augmentations = augmentations 
                    
    def __len__(self):
        return len(self.df)
 
    def __getitem__(self, idx):
        # print(f'{idx}\n')
        row = self.df.iloc[idx]
        metadata = torch.tensor(row[['Age', 'Sex', 'ViewPosition']], dtype=torch.float32)
        dicom_path = row['path']
        image = normalizer(dicom_path)
        # Load mask (0 for background, 255 for pneumothorax)
        encoded_pixels= row['EncodedPixels']
        if pd.isna(encoded_pixels) or str(encoded_pixels).strip() == '-1' or str(encoded_pixels).strip() == '':
            mask = np.zeros((1024, 1024), dtype=np.float32)
        else: 
            mask = rle2mask(row['EncodedPixels'], 1024,1024)
            mask = np.rot90(mask, 3) #rotating three times 90 to the right place
            mask = np.flip(mask, axis=1)
            mask = (mask > 0).astype(np.float32)
            
        if self.augmentations:
            augmented = self.augmentations(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.float32)
            
        if mask.ndim == 2:
            mask = torch.unsqueeze(mask, 0)

        if isinstance(image, torch.Tensor):
            assert image.shape[1:] == mask.shape[1:], f"Dimension mismatch! Image: {image.shape}, Mask: {mask.shape}"
        else:
            assert image.shape[:2] == mask.shape[:2], f"Dimension mismatch! Image: {image.shape}, Mask: {mask.shape}"

        target = 0 if (pd.isna(encoded_pixels) or str(encoded_pixels).strip() == '-1' or str(encoded_pixels).strip() == '') else 1
        # target_tensor = torch.tensor([target], dtype=torch.float32)
        
        # row['UID']
        return (image, mask , metadata , target , idx, row['UID'])