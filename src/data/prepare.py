
import os
from glob import glob
import pandas as pd
import pydicom
from sklearn.preprocessing import StandardScaler
from configs.configs import get_config
cfg = get_config()


def create_df_patient(dataframe, path, ImageId=None, inference=False):
    # dataframe to ease the access
    patients = []
    missing = 0

    pd.reset_option('max_colwidth')

    for t in dataframe:
        data = pydicom.dcmread(t)
        # print(data)
        # break
        patient = {}
        patient["UID"] = data.SOPInstanceUID
        try:
            encoded_pixels = masks[masks["ImageId"] == patient["UID"]].values[0][1]
            patient["EncodedPixels"] = encoded_pixels
        except:
            missing = missing + 1
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

    # pd.set_option('display.max_colwidth', -1)
    df = pd.DataFrame(patients, columns=["UID", "EncodedPixels", "Age", "Sex", "Modality", "BodyPart", "ViewPosition", "path"])
    df.head()
    return df


print(os.listdir(cfg.paths.dataset_root))

train = sorted(glob(os.path.join(cfg.paths.dataset_root, cfg.paths.train_dicom_glob)))
test = sorted(glob(os.path.join(cfg.paths.dataset_root, cfg.paths.test_dicom_glob)))
print("train files: ", len(train))
print("test files: ", len(test))

pd.reset_option('max_colwidth')

print("the csv with the labels: -1 means no Pneumothorax, othervise there is an encoding for the place of Pneumothorax")
masks = pd.read_csv(os.path.join(cfg.paths.dataset_root, cfg.paths.train_rle_csv), delimiter=",")
masks.head()


# --------------------------------
#   create train_df of Stage 1
# -------------------------------
train_dicom_dir = os.path.join(cfg.paths.dataset_root, "dicom-images-train") + "/"
train_df = create_df_patient(train, train_dicom_dir)
train_df['class'] = train_df['EncodedPixels'].apply(
    lambda x: 0 if (pd.isna(x) or str(x).strip() == '-1' or str(x).strip() == '') else 1
)
train_df['Sex'] = train_df['Sex'].map({'M': 1, 'F': 0})
train_df['ViewPosition'] = train_df['ViewPosition'].map({'AP': 1, 'PA': 0})
train_df.head()
scaler = StandardScaler()
train_df[['Age', 'Sex', 'ViewPosition']] = scaler.fit_transform(train_df[['Age', 'Sex', 'ViewPosition']])


# ------------------------------
#   create test_df of Stage 1
# -----------------------------
test_dicom_dir = os.path.join(cfg.paths.dataset_root, "dicom-images-test") + "/"
test_df = create_df_patient(test, test_dicom_dir)
test_df['class'] = test_df['EncodedPixels'].apply(
    lambda x: 0 if (pd.isna(x) or str(x).strip() == '-1' or str(x).strip() == '') else 1
)
test_df['Sex'] = test_df['Sex'].map({'M': 1, 'F': 0})
test_df['ViewPosition'] = test_df['ViewPosition'].map({'AP': 1, 'PA': 0})
scaler = StandardScaler()
test_df[['Age', 'Sex', 'ViewPosition']] = scaler.fit_transform(test_df[['Age', 'Sex', 'ViewPosition']])


# -------------------------------------------------------------------
#   labeling the test_df of 1st stage by train data of second stage
# ------------------------------------------------------------------
df = pd.read_csv(os.path.join(cfg.paths.stage2_root, cfg.paths.stage2_train_csv))
df.set_index('ImageId', inplace=True)
missing_uid = []
for i, val in enumerate(test_df['UID']):
    try:
        test_df['EncodedPixels'].iloc[i] = df['EncodedPixels'].loc[val]
    except KeyError:
        print(val)
        missing_uid.append(val)
test_df['path'] = test_df['path'].str.replace('dicom-images-train', 'dicom-images-test', regex=False)
test_df['class'] = test_df['EncodedPixels'].apply(
    lambda x: 0 if (pd.isna(x) or str(x).strip() == '-1' or str(x).strip() == '') else 1
)
test_df.head()
scaler = StandardScaler()
test_df[['Age', 'Sex', 'ViewPosition']] = scaler.fit_transform(test_df[['Age', 'Sex', 'ViewPosition']])
# test_df.to_csv('test_df.csv', index=False)

# ------------------------------
#   create test_df of Stage 2
# -----------------------------
stage2_image_path = sorted(glob(os.path.join(cfg.paths.stage2_root, cfg.paths.stage2_test_dicom_glob)))
stage2_image_dir = os.path.join(cfg.paths.stage2_root, "stage_2_images") + "/"
test_df_stage2 = create_df_patient(stage2_image_path, stage2_image_dir, inference=True)
test_df_stage2['class'] = test_df_stage2['EncodedPixels'].apply(
    lambda x: 0 if (pd.isna(x) or str(x).strip() == '-1' or str(x).strip() == '') else 1
)
test_df_stage2['Sex'] = test_df_stage2['Sex'].map({'M': 1, 'F': 0})
test_df_stage2['ViewPosition'] = test_df_stage2['ViewPosition'].map({'AP': 1, 'PA': 0})
test_df_stage2['Age'] = test_df_stage2['Age'].astype(int)


scaler = StandardScaler()
test_df_stage2[['Age', 'Sex', 'ViewPosition']] = scaler.fit_transform(test_df_stage2[['Age', 'Sex', 'ViewPosition']])
test_df_stage2.head(5)