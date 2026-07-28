import os
from glob import glob
import joblib

import pandas as pd
import pydicom
from sklearn.preprocessing import StandardScaler

from configs.configs import get_config

scaler = StandardScaler()

def create_df_patient(dataframe, path, masks_df, ImageId=None, inference=False):
    """
    Builds a flat per-image dataframe from a list of DICOM file paths.
    (Same logic as the notebook's create_df_patient cell, except `masks_df`
    is now passed in explicitly instead of being read from a notebook-global
    `masks` variable.)

    Args:
        dataframe: iterable of .dcm file paths to read.
        path: directory prefix used to reconstruct the `path` column.
        masks_df: dataframe (e.g. train-rle.csv) used to look up EncodedPixels
            for each image by UID. Pass an empty dataframe with an
            "ImageId"/"EncodedPixels" columns if no labels are available.
        inference: if True, builds the flat "<uid>.dcm" path used for the
            stage-2 competition test images instead of the nested
            study/series/instance path used for the stage-1 DICOM tree.
    """
    patients = []
    missing = 0

    pd.reset_option('max_colwidth')

    for t in dataframe:
        data = pydicom.dcmread(t)
        patient = {}
        patient["UID"] = data.SOPInstanceUID
        try:
            encoded_pixels = masks_df[masks_df["ImageId"] == patient["UID"]].values[0][1]
            patient["EncodedPixels"] = encoded_pixels
        except Exception:
            missing += 1
        patient["Age"] = data.PatientAge
        patient["Sex"] = data.PatientSex
        patient["Modality"] = data.Modality
        patient["BodyPart"] = data.BodyPartExamined
        patient["ViewPosition"] = data.ViewPosition
        if not inference:
            patient["path"] = path + data.StudyInstanceUID + "/" + data.SeriesInstanceUID + "/" + data.SOPInstanceUID + ".dcm"
        else:
            patient["path"] = path + data.SOPInstanceUID + '.dcm'
        patients.append(patient)

    df = pd.DataFrame(
        patients,
        columns=["UID", "EncodedPixels", "Age", "Sex", "Modality", "BodyPart", "ViewPosition", "path"],
    )
    return df


def load_stage1_dataframes(cfg=None):
    """
    Reproduces the notebook's "Prepare Data" section end-to-end:
      1. list stage-1 DICOMs + train-rle.csv
      2. build + encode + scale train_df
      3. build + encode + scale test_df
      4. re-label test_df using the stage-2 train csv (the stage-1 test
         split becomes labeled data once stage 2 begins), then re-scale.

    Returns:
        (train_df, test_df, masks)
    """
    cfg = cfg or get_config()
    root = cfg.paths.dataset_root

    print(os.listdir(root))
    train_paths = sorted(glob(os.path.join(root, cfg.paths.train_dicom_glob)))
    test_paths = sorted(glob(os.path.join(root, cfg.paths.test_dicom_glob)))
    print("train files: ", len(train_paths))
    print("test files: ", len(test_paths))

    pd.reset_option('max_colwidth')
    print("the csv with the labels: -1 means no Pneumothorax, otherwise there is an encoding for the place of Pneumothorax")
    masks = pd.read_csv(os.path.join(root, cfg.paths.train_rle_csv), delimiter=",")

    # ---- train_df (Stage 1) ----

    train_dicom_dir = os.path.join(root, "dicom-images-train") + "/"
    train_df = create_df_patient(train_paths, train_dicom_dir, masks)
    train_df[['Age','Sex','ViewPosition']] = scaler.fit_transform(train_df[['Age','Sex','ViewPosition']])
    joblib.dump(scaler, cfg.paths.processed_dir + "/meta_scaler.pkl")

    # ---- test_df (Stage 1) ----
    test_dicom_dir = os.path.join(root, "dicom-images-test") + "/"
    test_df = create_df_patient(test_paths, test_dicom_dir, masks)
    
    
    # ---- re-label test_df with Stage 2 ground truth ----
    stage2_train = pd.read_csv(os.path.join(cfg.paths.stage2_root, cfg.paths.stage2_train_csv))
    stage2_train.set_index('ImageId', inplace=True)
    for i, uid in enumerate(test_df['UID']):
        try:
            test_df['EncodedPixels'].iloc[i] = stage2_train['EncodedPixels'].loc[uid]
        except KeyError:
            print(uid)
    test_df['path'] = test_df['path'].str.replace('dicom-images-train', 'dicom-images-test', regex=False)
    test_df[['Age','Sex','ViewPosition']] = scaler.transform(test_df[['Age','Sex','ViewPosition']])

    return train_df, test_df, masks


def load_stage2_test_dataframe(cfg=None, masks_df=None):
    """Reproduces the notebook's stage-2 test dataframe (used for final inference/submission)."""
    cfg = cfg or get_config()
    if masks_df is None:
        masks_df = pd.DataFrame(columns=["ImageId", "EncodedPixels"])

    stage2_image_path = sorted(glob(os.path.join(cfg.paths.stage2_root, cfg.paths.stage2_test_dicom_glob)))
    stage2_image_dir = os.path.join(cfg.paths.stage2_root, "stage_2_images") + "/"
    test_df_stage2 = create_df_patient(stage2_image_path, stage2_image_dir, masks_df, inference=True)

    test_df_stage2['class'] = test_df_stage2['EncodedPixels'].apply(
        lambda x: 0 if (pd.isna(x) or str(x).strip() == '-1' or str(x).strip() == '') else 1
    )
    test_df_stage2['Sex'] = test_df_stage2['Sex'].map({'M': 1, 'F': 0})
    test_df_stage2['ViewPosition'] = test_df_stage2['ViewPosition'].map({'AP': 1, 'PA': 0})
    test_df_stage2['Age'] = test_df_stage2['Age'].astype(int)

    test_df_stage2[['Age', 'Sex', 'ViewPosition']] = scaler.transform(
        test_df_stage2[['Age', 'Sex', 'ViewPosition']]
    )
    return test_df_stage2


if __name__ == "__main__":
    # Quick manual smoke-test: `python -m src.data.prepare`
    _cfg = get_config()
    _train_df, _test_df, _masks = load_stage1_dataframes(_cfg)
    print(_train_df.head())
    print(_test_df.head())