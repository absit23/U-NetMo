import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(img_size=512, is_train=True):
    """
    Gold Standard Data Augmentation Pipeline:
    Combines spatial transforms (flips, rotations, scaling) with color jittering
    to expand rare class diversity without altering pixel labels.
    """
    if is_train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.0625, 0.0625), rotate=(-45, 45), p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])


class ConcreteDamageDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        
        self.image_filenames = sorted([
            f for f in os.listdir(images_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

    def _find_mask_path(self, img_name):
        """
        Locates mask path even if the image and mask have different file extensions (.jpg vs .png).
        """
        # 1. Direct exact match
        direct_path = os.path.join(self.masks_dir, img_name)
        if os.path.exists(direct_path):
            return direct_path

        # 2. Match stem with alternative extensions
        stem = os.path.splitext(img_name)[0]
        for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
            alt_path = os.path.join(self.masks_dir, stem + ext)
            if os.path.exists(alt_path):
                return alt_path

        # 3. Check for suffix pattern (e.g. filename_mask.png)
        for ext in ['.png', '.jpg', '.jpeg']:
            alt_path = os.path.join(self.masks_dir, f"{stem}_mask{ext}")
            if os.path.exists(alt_path):
                return alt_path

        return None

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = self._find_mask_path(img_name)

        if mask_path is None:
            raise FileNotFoundError(
                f"Mask matching image '{img_name}' not found in '{self.masks_dir}'."
            )

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Failed to read image at: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Failed to read mask at: {mask_path}")

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return {'image': image, 'mask': mask.long()}
