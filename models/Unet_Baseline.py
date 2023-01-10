import torch
import torch.nn as nn
import torch.nn.functional as F

class Unet_Baseline(torch.nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(Unet_Baseline, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, bilinear)
        self.up_conv1 = DoubleConv(1024, 512)

        self.up2 = Up(512, bilinear)
        self.up_conv2 = DoubleConv(512, 256)

        self.up3 = Up(256, bilinear)
        self.up_conv3 = DoubleConv(256, 128)

        self.up4 = Up(128, bilinear)
        self.up_conv4 = DoubleConv(128, 64)

        self.outc = OutConv(64, n_classes)


    def forward(self, img, ids, mask, token_type_ids):

        x1 = self.inc(img)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        decode1 = self.up1(x5)

        x = concatenate_layers(decode1, x4)
        x = self.up_conv1(x)

        decode2 = self.up2(x)

        x = concatenate_layers(decode2, x3)
        x = self.up_conv2(x)

        decode3 = self.up3(x)

        x = concatenate_layers(decode3, x2)
        x = self.up_conv3(x)

        decode4 = self.up4(x)

        x = concatenate_layers(decode4, x1)
        x = self.up_conv4(x)

        logits = self.outc(x)

        return logits


class Up(nn.Module):
    """Upscaling"""

    def __init__(self, in_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

    def forward(self, x1):
        x1 = self.up(x1)

        return x1


def concatenate_layers(x1, x2):
    # input is CHW
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]

    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2])

    x = torch.cat([x2, x1], dim=1)
    return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)




