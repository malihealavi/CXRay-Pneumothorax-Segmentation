import torch
import torch.nn as nn
import timm

class MedicalFusionClassifier(nn.Module):
    def __init__(self, backbone_name='efficientnet_b3', num_meta_features=3):
        """
        Args:
            backbone_name (str): The timm backbone to use.
            num_meta_features (int): The number of tabular metadata columns you are passing in.
        """
        super().__init__()
        
        # ---------------------------------------------------
        # 1. The Image Encoder (Backbone)
        # ---------------------------------------------------
        self.image_encoder = timm.create_model(
            backbone_name, 
            pretrained=True, 
            in_chans=1, 
            num_classes=0 
        )
        
        num_image_features = self.image_encoder.num_features

        # ---------------------------------------------------
        # 2. The Metadata Encoder (MLP)
        # ---------------------------------------------------
        self.meta_encoder = nn.Sequential(
            nn.Linear(num_meta_features, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        # ---------------------------------------------------
        # 3. The Final Fusion Classification Head
        # ---------------------------------------------------
        total_features = num_image_features + 16
        self.classifier = nn.Linear(total_features, 1)

    def forward(self, image, metadata):
        img_features = self.image_encoder(image)      
        meta_features = self.meta_encoder(metadata)   
        fused_features = torch.cat((img_features, meta_features), dim=1) 
        logit = self.classifier(fused_features)
        
        return logit