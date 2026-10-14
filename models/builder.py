import torch
import torch.nn as nn
from unet import UNet, UNetMobileNetV3
from models.deeplabv3plus import DeepLabV3Plus
from models.segformer import SegFormerB2

def build_model(model_name="unet", n_channels=3, n_classes=6, pretrained=True):
    """
    Factory function for model benchmarking.
    """
    model_name = model_name.lower()
    
    if model_name == "unet":
        # Standard U-Net (scratch)
        return UNet(n_channels=n_channels, n_classes=n_classes)
        
    elif model_name in ["unet_mobilenet", "mobilenet_unet"]:
        # U-Net with Pre-trained MobileNetV3 Encoder
        return UNetMobileNetV3(n_channels=n_channels, n_classes=n_classes, pretrained=pretrained)
        
    elif model_name in ["deeplabv3plus", "deeplabv3+"]:
        return DeepLabV3Plus(num_classes=n_classes, pretrained=pretrained)
        
    elif model_name == "segformer":
        return SegFormerB2(num_classes=n_classes, pretrained=pretrained)
        
    else:
        raise ValueError(f"Unknown model_name: '{model_name}'.")
