
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import segmentation_models_pytorch as smp
import torch.optim as optim 
from src.models.segmentor import build_stage2_segmentor
from src.data.Dataset import PneumothoraxDataset
from src.data.augment import data_transformer
from src.utils.utils import calculate_dice_coef
from src.utils.utils import plot_loss
from configs.configs import get_config

cfg = get_config()

def segmentor_train(train_df, val_df, epoch=15, device='cuda'):
    df_positive_train = train_df[train_df['class'] == 1].reset_index(drop=True)
    df_positive_val = val_df[val_df['class'] == 1].reset_index(drop=True)
    
    # ------------------- model config ------------------------
    # Using SMP to build the U-Net with SE-ResNeXt-50 backbone
    segmentor_model = build_stage2_segmentor().to(device)
    # Define the Combo Loss Components
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_dice = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    
    optimizer = optim.AdamW(segmentor_model.parameters(), lr=1e-4, weight_decay=1e-1)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    scaler = GradScaler()
    # --------------------- data_loading ----------------------

    train_dataset = PneumothoraxDataset(df=df_positive_train, augmentations=data_transformer(phase='train' , size=256))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, drop_last=False)
    val_dataset = PneumothoraxDataset(df=df_positive_val, augmentations=data_transformer(phase='val', size=256))
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, drop_last=False)
    # -------------------------training----------------------
    best_val_dice = 0.0 # Track best Dice score (higher is better)
    val_loss_list=[]
    train_loss_list=[]
    for e in range(epoch):
        print(f"----------- Epoch {e+1} ------------")

        size = 256 if e<=6 else (512 if e<=14 else 1024)
        batch_size = 32 if size == 256 else (8 if size == 512 else 4)
        base_lr = 5e-4 if size == 256 else (1e-4 if size == 512 else 1e-4)        

    
        if e in [7,15]:
            optimizer.param_groups[0]['lr'] = base_lr
            print(f"\n--- Transitioning to next Stage : img_size : {size}, batch_size:{batch_size}, lr:{base_lr}")
            train_dataset = PneumothoraxDataset(df=df_positive_train, augmentations=data_transformer(phase='train' , size = size))
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)
            val_dataset = PneumothoraxDataset(df=df_positive_val, augmentations=data_transformer(phase='val', size = size))
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
            
        
        train_loss = 0.0
        train_dice_sum = 0.0
        segmentor_model.train()
        
        for (images, masks , metadata , targets , idx, uid) in train_loader: 
            images = images.to(device)
            # Ensure masks are (B, 1, H, W) and float32
            masks = masks.to(device, dtype=torch.float32)
            
            optimizer.zero_grad()
            
            # Forward pass with Mixed Precision
            with autocast():
                # Output is raw logits
                logits = segmentor_model(images)
                
                # Calculate Combo Loss (1:1 Ratio of BCE and Dice)
                loss_bce = criterion_bce(logits, masks)
                loss_dice = criterion_dice(logits, masks)
                loss = loss_bce + loss_dice
            
            # Backward pass and optimization using the scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Metrics Tracking
            current_batch_size = images.size(0)
            train_loss += loss.item() * current_batch_size
            
            # Calculate training Dice on the fly
            # Detach is safer here to avoid attaching graph memory during metrics
            probs = torch.sigmoid(logits).detach()
            preds = (probs > 0.5).float()
            train_dice_sum += calculate_dice_coef(preds, masks) * current_batch_size
       
            # pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        epoch_avg_loss = train_loss / len(train_loader.dataset)
        epoch_avg_dice = train_dice_sum / len(train_loader.dataset)
        train_loss_list.append(epoch_avg_loss)
        
        print(f"Epoch {e+1} | Train Loss: {epoch_avg_loss:.4f} | Train Dice: {epoch_avg_dice:.4f}")
    
        # ----------------validation-----------
        segmentor_model.eval()
        val_loss = 0.0
        val_dice_sum = 0.0
        
        with torch.no_grad():
            # pbar_val = tqdm(val_loader, desc=f"Epoch {e+1}/{epoch} [Val]")
            for (images, masks , metadata , target , idx, uid)in val_loader:
                images = images.to(device)
                masks = masks.to(device, dtype=torch.float32)
                
                # Forward pass (autocast is optional but slightly faster during inference too)
                with autocast():
                    logits = segmentor_model(images)
                    
                    # Combo Loss
                    loss_bce = criterion_bce(logits, masks)
                    loss_dice = criterion_dice(logits, masks)
                    loss = loss_bce + loss_dice
                
                # Metrics Tracking
                current_batch_size = images.size(0)
                val_loss += loss.item() * current_batch_size
                
                # Binarize and calculate Dice
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                val_dice_sum += calculate_dice_coef(preds, masks) * current_batch_size
                
                # pbar_val.set_postfix(loss=f"{loss.item():.4f}")
    
        epoch_avg_val_loss = val_loss / len(val_loader.dataset)
        epoch_avg_val_dice = val_dice_sum / len(val_loader.dataset)
        val_loss_list.append(epoch_avg_val_loss)
        scheduler.step(epoch_avg_val_loss)
        
        print(f"Epoch {e+1} | Val Loss: {epoch_avg_val_loss:.4f} | Val Dice: {epoch_avg_val_dice:.4f} , LR:{optimizer.param_groups[0]['lr']:.6f}\n")
        
        # Save model based on the highest Dice score
        if epoch_avg_val_dice > best_val_dice:
            best_val_dice = epoch_avg_val_dice
            best_sgm = segmentor_model
            save_path = f"best_segmentor.pth"
            torch.save(segmentor_model.state_dict(), save_path)
            print(f"[*] Val Dice improved. Model saved to {save_path}")
            patience=0    
        else:
            patience +=1
        if patience >= 4:
            plot_loss(val_loss_list, train_loss_list, 'segmentor_loss_trend.png')
            return best_sgm
    plot_loss(val_loss_list, train_loss_list, 'segmentor_loss_trend.png')  
    return best_sgm 
# Execution
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# segmentor_model = segmentor_train(combined_df,val_df, epoch=15, device=device)
