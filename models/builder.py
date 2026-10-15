import torch
import torch.nn as nn

def build_model(model_name="unet", n_channels=3, n_classes=8, pretrained=True):
    """
    Factory function for model benchmarking.
    Imports models lazily inside the function so unfinished or empty model files
    do not crash execution when running a different model.
    """
    model_name = model_name.lower()

    if model_name == "unet":
        try:
            from unet.unet_model import UNet
        except ImportError:
            from unet import UNet
        return UNet(n_channels=n_channels, n_classes=n_classes)

    elif model_name in ["unet_mobilenet", "mobilenet_unet"]:
        try:
            from unet.unet_mobilenet import UNetMobileNetV3
        except ImportError:
            from unet import UNetMobileNetV3
        return UNetMobileNetV3(n_channels=n_channels, n_classes=n_classes, pretrained=pretrained)

    elif model_name in ["deeplabv3plus", "deeplabv3+"]:
        try:
            from models.deeplabv3plus import DeepLabV3Plus
            return DeepLabV3Plus(num_classes=n_classes, pretrained=pretrained)
        except (ImportError, AttributeError):
            raise NotImplementedError("models/deeplabv3plus.py is currently empty or incomplete.")

    elif model_name == "segformer":
        try:
            from models.segformer import SegFormerB2
            return SegFormerB2(num_classes=n_classes, pretrained=pretrained)
        except (ImportError, AttributeError):
            raise NotImplementedError("models/segformer.py is currently empty or incomplete.")

    else:
        raise ValueError(f"Unknown model_name: '{model_name}'.")
