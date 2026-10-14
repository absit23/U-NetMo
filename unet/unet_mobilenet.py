import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .unet_parts import DoubleConv, OutConv

class UpBlock(nn.Module):
    """Upscaling then DoubleConv matching unet_parts style."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad x1 if shapes differ slightly
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetMobileNetV3(nn.Module):
    """
    U-Net with ImageNet Pre-trained MobileNetV3-Large Encoder.
    Uses exact building blocks from unet_parts.py.
    """
    def __init__(self, n_channels=3, n_classes=6, pretrained=True):
        super(UNetMobileNetV3, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # 1. Load Pre-trained MobileNetV3 Encoder
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        mobilenet = models.mobilenet_v3_large(weights=weights)
        features = mobilenet.features

        # Handle grayscale inputs if n_channels != 3
        if n_channels != 3:
            features[0][0][0] = nn.Conv2d(
                n_channels, 16, kernel_size=3, stride=2, padding=1, bias=False
            )

        # Encoder Slices (Extracting multi-scale feature maps)
        self.inc = features[0:2]   # Out: [16, H/2, W/2]
        self.down1 = features[2:4] # Out: [24, H/4, W/4]
        self.down2 = features[4:7] # Out: [40, H/8, W/8]
        self.down3 = features[7:13]# Out: [112, H/16, W/16]
        self.down4 = features[13:16]# Out: [160, H/32, W/32] (Bottleneck)

        # 2. Decoder Path (Calculated from MobileNetV3 + Skip channels)
        # up1: 160 (down4) + 112 (down3 skip) = 272 in -> 128 out
        self.up1 = UpBlock(in_channels=272, out_channels=128)

        # up2: 128 (up1) + 40 (down2 skip) = 168 in -> 64 out
        self.up2 = UpBlock(in_channels=168, out_channels=64)

        # up3: 64 (up2) + 24 (down1 skip) = 88 in -> 32 out
        self.up3 = UpBlock(in_channels=88, out_channels=32)

        # up4: 32 (up3) + 16 (inc skip) = 48 in -> 16 out
        self.up4 = UpBlock(in_channels=48, out_channels=16)

        # Final Classification Convolution
        self.outc = OutConv(16, n_classes)

    def forward(self, x):
        # Encoder Pass
        x1 = self.inc(x)      # [16, H/2, W/2]
        x2 = self.down1(x1)   # [24, H/4, W/4]
        x3 = self.down2(x2)   # [40, H/8, W/8]
        x4 = self.down3(x3)   # [112, H/16, W/16]
        x5 = self.down4(x4)   # [160, H/32, W/32]

        # Decoder Pass
        x = self.up1(x5, x4)  # [128, H/16, W/16]
        x = self.up2(x, x3)   # [64, H/8, W/8]
        x = self.up3(x, x2)   # [32, H/4, W/4]
        x = self.up4(x, x1)   # [16, H/2, W/2]

        # Final 2x Upsample to original image resolution (H, W)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
