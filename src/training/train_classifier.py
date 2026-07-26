import torch
import numpy as np
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import  DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import (roc_auc_score, f1_score , confusion_matrix , fbeta_score)
from src.models.classifier import MedicalFusionClassifier
from src.data.Dataset import PneumothoraxDataset
from src.data.augment import data_transformer
from src.utils.utils import f2_best_threshold, gmean_best_threshold, pr_auc_score
from src.utils.utils import plot_loss   
from configs.configs import get_config

cfg = get_config()



def classifier_train(train_df,val_df, epoch=10, device='cuda'):
    # classifier_model = build_stage1_classifier()
    classifier_model = MedicalFusionClassifier(backbone_name=cfg.model.classifier_backbone, num_meta_features=cfg.model.num_meta_features)
    classifier_model.to(device)
  
    # --------------------- model config ----------------------
    pos_weights = (train_df['class'] == 0).sum() / (train_df['class'] == 1).sum()  
    pos_weight_tensor = torch.tensor([pos_weights], dtype=torch.float32).to(device)
    criterion_wbce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.AdamW(classifier_model.parameters(), lr=5e-4 , weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=2)
    
    train_dataset = PneumothoraxDataset(df=train_df, augmentations=data_transformer(phase='train' , size = 256))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, drop_last=False)
    val_dataset = PneumothoraxDataset(df=val_df, augmentations=data_transformer(phase='val', size = 256))
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, drop_last=False)

    # -------------------------training------------------------
    train_loss_list= []
    val_loss_list= []
    best_val_loss = float('inf') 
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
            print(f"\n--- Transitioning to next Stage... img_size : {size}, batch_size:{batch_size}, lr:{base_lr}")
            optimizer.param_groups[0]['lr'] = base_lr
            train_dataset = PneumothoraxDataset(df=train_df, augmentations=data_transformer(phase='train' , size = size))
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)
            val_dataset = PneumothoraxDataset(df=val_df, augmentations=data_transformer(phase='val', size = size))
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)

        
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
        _ , best_f2 =  f2_best_threshold(all_ttargets, all_tprobs)
        _ , best_gmean =  gmean_best_threshold(all_ttargets, all_tprobs)
        pr_auc = pr_auc_score(all_ttargets, all_tprobs)
        
        print(f"Epoch {e+1} Completed Train : Average Loss: {epoch_avg_loss:.4f}| pr_auc : {pr_auc:.4f}| AUC : {auc:.4f} | gmean : {best_gmean:.4f}| F2: {best_f2:.4f} ")
    
        # ----------------------  validation  -------------------------
        classifier_model.eval()
        val_loss = 0.0
        all_vtargets = []
        all_vprobs = []
        all_vpreds = []
        with torch.no_grad():
            for (images, mask , metadata , targets , idx, uid) in val_loader:
                targets = targets.float().unsqueeze(1).to(device) 
                images = images.to(device)
                metadata = metadata.to(device)
                
                with autocast():
                    logits = classifier_model(images, metadata)
                    loss = criterion_wbce(logits, targets)
                
                current_batch_size = images.size(0)
                val_loss += loss.item() * current_batch_size
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                # preds = (probs > 0.5).astype(int)
                all_vtargets.extend(targets.cpu().numpy().flatten())
                all_vprobs.extend(probs)
                # all_vpreds.extend(preds)

       
        epoch_avg_val_loss = val_loss / len(val_loader.dataset)
        
        best_f2_threshold , best_f2 =  f2_best_threshold(all_vtargets, all_vprobs)
        best_gmean_threshold , best_gmean =  gmean_best_threshold(all_vtargets, all_vprobs)
        pr_auc = pr_auc_score(all_vtargets, all_vprobs)
        auc = roc_auc_score(all_vtargets, all_vprobs)

        all_vprobs = np.asarray(all_vprobs)
        all_vtargets = np.asarray(all_vtargets)
        
        all_vpreds= (all_vprobs > best_f2_threshold).astype(int)
        # best_threshold = 0.5
        val_f2 = fbeta_score(all_vtargets, all_vpreds ,  beta=2.0)
        tn, fp, fn, tp = confusion_matrix(all_vtargets, all_vpreds).ravel()
       
        scheduler.step(epoch_avg_val_loss)
        
        print(f'best_gmean_threshold:{best_gmean_threshold}')
        print(f'best_f2_threshold:{best_f2_threshold}')
        print(f"Epoch {e+1} Completed. Test  : AUC : {auc:.4f} |  pr_auc:{pr_auc:.4f} | F2:{val_f2:.4f} |  gmean : {best_gmean:.4f} | Precision: {tp/(tp + fp):.4f} | TP Rate (Recall): {tp/(tp+fn):.4f} | TN Rate (Specifity): {tn/(tn + fp):.4f} | Average Loss: {epoch_avg_val_loss:.4f} | LR:{optimizer.param_groups[0]['lr']:.5f} \n")
        
        val_loss_list.append(epoch_avg_val_loss)
        
        if epoch_avg_val_loss < best_val_loss:
            best_val_loss = epoch_avg_val_loss
            save_path = f"best_classifier.pth"
            torch.save(classifier_model.state_dict(), save_path)
            
            
            best_cls = classifier_model
            print(f"[*] Val loss improved. Model saved to {save_path}")
            patience=0
        else:
            patience +=1
        if patience >=4:
            tn, fp, fn, tp = confusion_matrix(all_vtargets, all_vpreds).ravel()

            # --- Print Results ---
            print(f"True Negatives  (TN): {tn}")
            print(f"False Positives (FP): {fp}  <- (False Alarms)")
            print(f"False Negatives (FN): {fn}  <- (Missed Diagnoses)")
            print(f"True Positives  (TP): {tp}")
            print(f"TP Rate (Recall): {tp/(tp+fn)}")
            print(f"TN Rate (Specifity): {tn/(tn + fp)}")
            plot_loss(val_loss_list, train_loss_list, 'classifier_loss_trend.png')
            return best_cls
    plot_loss(val_loss_list, train_loss_list, 'classifier_loss_trend.png')
    return  best_cls


# Execution
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# combine_with_train, val_df = train_test_split(test_df, test_size=0.5, random_state=42 , stratify=test_df['class'])
# combined_df = pd.concat([train_df, combine_with_train], ignore_index=True)
# combined_df = combined_df.reset_index(drop=True)
# val_df = val_df.reset_index(drop=True)
# classifier_model = classifier_train(combined_df, val_df,  epoch=12, device=device)
# classifier_model.to('cpu') 
# gc.collect()
# torch.cuda.empty_cache()