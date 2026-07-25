import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score, f1_score
from src.models.classifier import MedicalFusionClassifier
from src.models.segmentor import build_stage2_segmentor
from src.data.Dataset import PneumothoraxDataset
from src.data.augment import data_transformer
from src.utils.utils import calculate_dice_coef
import torch.optim as optim
from configs.configs import get_config

cfg = get_config()

def classifier(df, device, epoch = 15):
    
    train_df = df
    classifier_model = MedicalFusionClassifier(backbone_name='efficientnet_b3', num_meta_features=3)
    classifier_model.to(device)
    
    pos_weights = (train_df['class'] == 0).sum() / (train_df['class'] == 1).sum()  
    pos_weight_tensor = torch.tensor([pos_weights], dtype=torch.float32).to(device)
    criterion_wbce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.AdamW(classifier_model.parameters(), lr=5e-4 , weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=2)
    
    train_dataset = PneumothoraxDataset(df=train_df, augmentations=data_transformer(phase='train' , size = 256))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, drop_last=False)

    train_loss_list= []
    scaler = GradScaler()
    patience = 0
    best_cls = None
    
    for e in range(epoch):
        print(f"----------- Epoch {e+1} ------------")

        size = 256 if e<=3 else (512)
        batch_size = 32 if size == 256 else (16)
        base_lr = 5e-4 if size == 256 else (1e-4)
                # --------------------- data_loading ----------------------
        
        if e in [4]:
            print(f"\n--- Transitioning to next Stage : img_size : {size}, batch_size:{batch_size}, lr:{base_lr}")
            train_dataset = PneumothoraxDataset(df=train_df, augmentations=data_transformer(phase='train' , size = size))
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)
            optimizer.param_groups[0]['lr'] = base_lr
                    
        train_loss = 0.0
        classifier_model.train()
        
        all_ttargets = []
        all_tprobs = []
        all_tpreds = []
        for (images, mask , metadata , targets , idx, uid) in train_loader:
            targets = targets.float().unsqueeze(1).to(device) 
            images = images.to(device)
            metadata = metadata.to(device)
            
            optimizer.zero_grad()
            with autocast():
                logits = classifier_model(images, metadata)
                loss = criterion_wbce(logits, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            current_batch_size = images.size(0)
            train_loss += loss.item() * current_batch_size
            
            probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)
           
            all_ttargets.extend(targets.cpu().numpy().flatten())
            all_tprobs.extend(probs)
            all_tpreds.extend(preds)
        
        epoch_avg_loss = train_loss / len(train_loader.dataset)
        train_loss_list.append(epoch_avg_loss)
        auc = roc_auc_score(all_ttargets, all_tprobs)
        f1 = f1_score(all_ttargets, all_tpreds)
        print(f"Epoch {e+1} Completed Train : Average Loss: {epoch_avg_loss:.4f} , AUC : {auc:.4f}, F1: {f1:.4f} ")
        
    save_path = f"best_classifier.pth"
    torch.save(classifier_model.state_dict(), save_path)
    return classifier_model


def segmentor(df , device, epoch = 10):
    train_df = df
    df_positive_train = train_df[train_df['class'] == 1].reset_index(drop=True)
 
    # ------------------- model config ------------------------
    segmentor_model = build_stage2_segmentor().to(device)
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_dice = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    optimizer = optim.AdamW(segmentor_model.parameters(), lr=5e-4, weight_decay=1e-1)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    scaler = GradScaler()
    # --------------------- data_loading ----------------------
    train_dataset = PneumothoraxDataset(df=df_positive_train, augmentations=data_transformer(phase='train' , size=256))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, drop_last=False)
    # -------------------------training----------------------
    train_loss_list=[]
    for e in range(epoch):
        print(f"----------- Epoch {e+1} ------------")

        size = 256 if e<=3 else (512 if e<=6 else 1024)
        batch_size = 32 if size == 256 else (8 if size == 512 else 2)
        base_lr = 5e-4 if size == 256 else (1e-4 if size == 512 else 5e-5)        
        if e in [4,7]:
            
            print(f"\n--- Transitioning to next Stage : img_size : {size}, batch_size:{batch_size}, lr:{base_lr}")
            train_dataset = PneumothoraxDataset(df=df_positive_train, augmentations=data_transformer(phase='train' , size = size))
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)
            optimizer.param_groups[0]['lr'] = base_lr

        
        train_loss = 0.0
        train_dice_sum = 0.0
        segmentor_model.train()
        
        for (images, masks , metadata , targets , idx, uid) in train_loader: 
            images = images.to(device)
            # Ensure masks are (B, 1, H, W) and float32
            masks = masks.to(device, dtype=torch.float32)
            
            optimizer.zero_grad()
            
            with autocast():
                logits = segmentor_model(images)
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
            
            probs = torch.sigmoid(logits).detach()
            preds = (probs > 0.5).float()
            train_dice_sum += calculate_dice_coef(preds, masks) * current_batch_size
       
        
        epoch_avg_loss = train_loss / len(train_loader.dataset)
        epoch_avg_dice = train_dice_sum / len(train_loader.dataset)
        train_loss_list.append(epoch_avg_loss)
        
        print(f"Epoch {e+1} | Train Loss: {epoch_avg_loss:.4f} | Train Dice: {epoch_avg_dice:.4f}")
    save_path = f"best_segmentor.pth"
    torch.save(segmentor_model.state_dict(), save_path)
    return segmentor_model



# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# combine_with_train, val_df = train_test_split(test_df, test_size=0.2, random_state=42 , stratify=test_df['class'])
# combined_df = pd.concat([train_df, combine_with_train], ignore_index=True)
# print(f'len combined_df (train+test) : {len(combined_df)} | len val_df : {len(val_df)} | first 10 index of val_df {val_df.head(10).index}')
# classifier_model = classifier(combined_df, device=device , epoch=6)
# # ------------------ VRAM CLEANUP STEP ------------------
# classifier_model.to('cpu') 
# gc.collect()
# torch.cuda.empty_cache()
# # --------------------------------------------------------
# segmentor_model = segmentor(combined_df, device=device , epoch=10)
