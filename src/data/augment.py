import albumentations as A
from albumentations.pytorch import ToTensorV2


def data_transformer(phase, size=1024):
    if phase=='train':
        train_transform = A.Compose([
            A.Resize(size, size), 
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
                A.GridDistortion(p=0.5),
            ], p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # ImageNet stats
            ToTensorV2()
        ])
        return train_transform
        
    if phase =='val':
        val_transform = A.Compose([
            A.Resize(size, size), # Start with 256 or 512 for progressive resizing
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # ImageNet stats
            ToTensorV2()
        ])
        return val_transform
    