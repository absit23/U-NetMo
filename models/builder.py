import torch
import torch.nn as nn
from unet.unet_model import UNet
from models.deeplabv3plus import DeepLabV3Plus
from models.segformer import SegFormerB2

def build_model(model_name="unet", num_classes=6, pretrained=True):
    """
    Factory function to instantiate benchmarking models.
    
    Args:
        model_name (str): 'unet', 'deeplabv3plus', or 'segformer'
        num_classes (int): Number of concrete damage classes
        pretrained (bool): Load pre-trained weights
    """
    model_name = model_name.lower()
    
    if model_name == "unet":
        return UNet(n_channels=3, n_classes=num_classes)
    elif model_name in ["deeplabv3plus", "deeplabv3+"]:
        return DeepLabV3Plus(num_classes=num_classes, pretrained=pretrained)
    elif model_name == "segformer":
        return SegFormerB2(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model_name: {model_name}. Select from ['unet', 'deeplabv3plus', 'segformer'].")
