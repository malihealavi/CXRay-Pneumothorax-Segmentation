import os

import torch
from glob import glob
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from src.models.classifier import MedicalFusionClassifier
from src.models.segmentor import build_stage2_segmentor
from src.data.prepare import load_stage2_test_dataframe
from src.inference.inference import inference
from configs.configs import get_config

cfg = get_config()

# No ground-truth masks exist for genuinely new inference data, so we pass
# an empty masks_df (create_df_patient just leaves EncodedPixels unset).
test_df_stage2 = load_stage2_test_dataframe(cfg)

torch.serialization.add_safe_globals([np._core.multiarray.scalar])

test_df= test_df_stage2

device = torch.device(cfg.device)
classifier_model = MedicalFusionClassifier(
    backbone_name=cfg.model.classifier_backbone, 
    num_meta_features=cfg.model.num_meta_features
).to(device)
classifier_model.load_state_dict(
    torch.load(cfg.model_path.classifier_checkpoint, map_location=device)
)

segmentor_model = build_stage2_segmentor().to(device) 
segmentor_model.load_state_dict(
    torch.load(cfg.model_path.segmentor_checkpoint, map_location=device)
)

df = inference(test_df_stage2, classifier_model, segmentor_model, device)

test_stage2 = pd.read_csv(os.path.join(cfg.paths.stage2_root, cfg.paths.stage2_sample_submission))
res=df[['UID', 'pred_mask']]
res = res.rename(columns={'UID': 'ImageId', 'pred_mask':'EncodedPixels'})
res = res.set_index('ImageId')
test_stage2 = test_stage2.set_index('ImageId')
res = res.loc[test_stage2.index]
res = res.reset_index()
res['EncodedPixels'] = res['EncodedPixels'].replace('-1', -1)

res.to_csv('submission.csv', index=False)