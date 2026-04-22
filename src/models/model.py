import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)

# Each Decoder Block upsamples the feature map and concatenates it with the corresponding 
# skip connection from the encoder
# So each decoder stage contains a two-convolution block, but the decoder as a whole is a 
# sequence of several such stages.
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip=None):
        # This doubles the spatial dimensions of x (e.g. [B, 256, 32, 32]  ->  [B, 256, 64, 64]),
        # because in the decoder we want to gradually recover the original image resolution.
        # So each decoder stage upsamples the feature map
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        if skip is not None:
            # make sure shapes match before concatenation (if sizes don't match => resize x to 
            # to exactly match the skip feature size before concatenation)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

            x = torch.cat([x, skip], dim=1)

        x = self.conv(x)
        return x


class UNetResNet34(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(UNetResNet34, self).__init__()

        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet34(weights=weights)

        # -------- Encoder --------
        self.initial = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu)
        self.maxpool = backbone.maxpool

        self.encoder1 = backbone.layer1
        self.encoder2 = backbone.layer2
        self.encoder3 = backbone.layer3
        self.encoder4 = backbone.layer4

        # -------- Bottleneck --------
        self.bottleneck = ConvBlock(512, 512)

        # -------- Decoder --------
        self.decoder4 = DecoderBlock(in_channels=512, skip_channels=256, out_channels=256)
        self.decoder3 = DecoderBlock(in_channels=256, skip_channels=128, out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
        self.decoder1 = DecoderBlock(in_channels=64, skip_channels=64, out_channels=32)
        self.decoder0 = DecoderBlock(in_channels=32, skip_channels=0, out_channels=16)

        # -------- Final segmentation head --------
        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        # -------- Encoder --------
        x0 = self.initial(x)     # [B, 64, H/2, W/2]
        x1 = self.maxpool(x0)    # [B, 64, H/4, W/4]
        x1 = self.encoder1(x1)   # [B, 64, H/4, W/4]
        x2 = self.encoder2(x1)   # [B, 128, H/8, W/8]
        x3 = self.encoder3(x2)   # [B, 256, H/16, W/16]
        x4 = self.encoder4(x3)   # [B, 512, H/32, W/32]

        # -------- Bottleneck --------
        b = self.bottleneck(x4)  # [B, 512, H/32, W/32]

        # -------- Decoder --------
        d4 = self.decoder4(b, x3)   # [B, 256, H/16, W/16]
        d3 = self.decoder3(d4, x2)  # [B, 128, H/8, W/8]
        d2 = self.decoder2(d3, x1)  # [B, 64, H/4, W/4]
        d1 = self.decoder1(d2, x0)  # [B, 32, H/2, W/2]
        d0 = self.decoder0(d1)      # [B, 16, H, W]

        out = self.final_conv(d0)   # [B, 1, H, W]
        return out


if __name__ == "__main__":
    model = UNetResNet34(num_classes=1, pretrained=True)

    _input = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        output = model(_input)

    print("Input shape:", _input.shape)
    print("Output shape:", output.shape)