import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ConcreteDamageDataset(Dataset):
    """
    PyTorch Dataset for Roboflow Multi-Class Concrete Damage Segmentation.
    Assumes images and masks share identical filenames in their respective folders.
    """
    def __init__(self, images_dir: str, masks_dir: str, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        # Get sorted list of image filenames
        self.image_names = sorted([
            f for f in os.listdir(images_dir) 
            if f.endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('.')
        ])

        if len(self.image_names) == 0:
            raise RuntimeError(f"No images found in directory: {images_dir}")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        # Match mask filename (Roboflow outputs matching filenames or PNG masks)
        mask_name = os.path.splitext(img_name)[0] + '.png'
        mask_path = os.path.join(self.masks_dir, mask_name)

        if not os.path.exists(mask_path):
            # Fallback if mask retains original image extension
            mask_path = os.path.join(self.masks_dir, img_name)

        # Load RGB Image and Grayscale Class Index Mask
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.int64)

        # Apply Albumentations Transforms (resizing, flips, normalization)
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            # Default fallback tensor conversion
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return {
            'image': image, # Shape: [3, H, W]
            'mask': mask    # Shape: [H, W] with integer class IDs (0, 1, 2, ...)
        }


def get_transforms(img_size=512, is_train=True):
    """Standard Albumentations pipeline for training and validation."""
    if is_train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
