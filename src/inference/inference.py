
import torch
import gc
from torch.utils.data import DataLoader
import numpy as np
from src.data.Dataset import PneumothoraxDataset
from src.data.augment import data_transformer
from src.utils.utils import mask2rle    


def inference(test, trained_classifier_model, trained_segmentor_model, device):
    test_df = test.copy()
    test_df['pred'] = None           
    test_df['pred_mask'] = "-1"  
    
    # 1024 is best practice since your segmentor finished training at 1024
    INFERENCE_SIZE = 1024 
    best_threshold = 0.45
    print(f'best_threshold : {best_threshold}')
    # ------------------STAGE 1: CLASSIFICATION (All samples)----------------------
    trained_classifier_model.to(device)
    # Match the inference size
    test_dataset = PneumothoraxDataset(df=test_df, augmentations=data_transformer(phase='val', size=INFERENCE_SIZE))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    trained_classifier_model.eval() 
    all_class_preds = {}
    
    with torch.no_grad():
        for (imgs, masks, metadata, targets, indices, uid) in test_loader:
            imgs = imgs.to(device)
            metadata = metadata.to(device)
            logits = trained_classifier_model(imgs, metadata)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs > best_threshold).astype(int)
            
            for i, idx in enumerate(indices.numpy()):
                all_class_preds.update({idx: preds[i]})

    # Map classification results back to master test_df
    for idx, pred in all_class_preds.items():
        test_df.loc[idx, 'pred'] = pred

    # ------------------ VRAM CLEANUP STEP ------------------
    trained_classifier_model.to('cpu') 
    gc.collect()
    torch.cuda.empty_cache()
    # --------------------------------------------------------

    # ------------------STAGE 2: SEGMENTATION (Positive samples only)-------------------
    positive_samples = test_df[test_df['pred'] == 1]
    sample_mask = None
    if len(positive_samples) > 0:
        trained_segmentor_model.to(device)
        
        pos_dataset = PneumothoraxDataset(df=positive_samples, augmentations=data_transformer(phase='val', size=INFERENCE_SIZE))
        
        # NOTE: Processing 1024x1024 images takes significantly more VRAM.
        # Keep batch_size small (4 or 8) to prevent Cuda Out-Of-Memory errors.
        pos_loader = DataLoader(pos_dataset, batch_size=4, shuffle=False, num_workers=4)   
        
        trained_segmentor_model.eval()  
        all_seg_preds = {}
        
        with torch.no_grad():
            for (imgs, masks, metadata, targets, indices, uid) in pos_loader:
                imgs = imgs.to(device)
                logits = trained_segmentor_model(imgs)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).cpu().numpy() 
                
                for i, uid_val in enumerate(uid):
                    mask = preds[i].squeeze() 
                    # --- REVERSE TRANSFORMATIONS HERE ---
                    mask = np.flip(mask, axis=1)
                    mask = np.rot90(mask, 1)
                    mask = np.ascontiguousarray(mask)
                    # ------------------------------------
                    mask = mask.astype(float)* 255 # convert binary mask with (0,1) values to grayscale with (0,255) 
                    all_seg_preds.update({uid_val: mask})

        # Directly update the master test_df using global indices
        for idx, pred_mask in all_seg_preds.items():
            # test_df.loc[idx, 'pred_mask'] = mask2rle(pred_mask, 1024, 1024)
            test_df.loc[test_df['UID'] == idx, 'pred_mask'] = mask2rle(pred_mask, 1024, 1024)
            
    # Final cleanup before returning dataframe
    trained_segmentor_model.to('cpu')
    gc.collect()
    torch.cuda.empty_cache()

    return test_df

