import torch
import torch.nn as nn
import torch.nn.functional as F

from .LanguageCrossAttention import LangCrossAtt

class ConTEXTual_model(torch.nn.Module):
    def __init__(self, lang_model, n_channels, n_classes, bilinear=False):
        super(ConTEXTual_model, self).__init__()

        self.lang_encoder = lang_model

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        self.up1 = Up(1024, bilinear)
        self.lang_attn = LangCrossAtt(emb_dim=1024)
        self.up_conv1 = DoubleConv(1024, 512)

        self.up2 = Up(512, bilinear)
        self.up_conv2 = DoubleConv(512, 256)

        self.up3 = Up(256, bilinear)
        self.up_conv3 = DoubleConv(256, 128)

        self.up4 = Up(128, bilinear)
        self.up_conv4 = DoubleConv(128, 64)

        self.outc = OutConv(64, n_classes)

        self.lang_proj1 = nn.Linear(1024, 512)
        self.lang_attn1 = LangCrossAtt(emb_dim=512)
        self.lang_proj2 = nn.Linear(1024, 256)
        self.lang_attn2 = LangCrossAtt(emb_dim=256)
        self.lang_proj3 = nn.Linear(1024, 128)
        self.lang_attn3 = LangCrossAtt(emb_dim=128)
        self.lang_proj4 = nn.Linear(1024, 64)
        self.lang_attn4 = LangCrossAtt(emb_dim=64)

    def forward(self, img, ids, mask, token_type_ids):
        # for roberta
        #lang_output = self.lang_encoder(ids, mask, token_type_ids)
        #word_rep = lang_output[0]
        #report_rep = lang_output[1]
        #lang_rep = word_rep

        # for t5
        encoder_output = self.lang_encoder.encoder(input_ids=ids, attention_mask=mask, return_dict=True)
        pooled_sentence = encoder_output.last_hidden_state
        lang_rep = pooled_sentence

        x1 = self.inc(img)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        decode1 = self.up1(x5)

        lang_rep1 = self.lang_proj1(lang_rep)
        decode1 = self.lang_attn1(lang_rep=lang_rep1, vision_rep=decode1)

        x = concatenate_layers(decode1, x4)
        x = self.up_conv1(x)

        decode2 = self.up2(x)
        lang_rep2 = self.lang_proj2(lang_rep)
        decode2 = self.lang_attn2(lang_rep=lang_rep2, vision_rep=decode2)

        x = concatenate_layers(decode2, x3)
        x = self.up_conv2(x)

        decode3 = self.up3(x)

        lang_rep3 = self.lang_proj3(lang_rep)
        decode3 = self.lang_attn3(lang_rep=lang_rep3, vision_rep=decode3)

        x = concatenate_layers(decode3, x2)
        x = self.up_conv3(x)

        decode4 = self.up4(x)

        lang_rep4 = self.lang_proj4(lang_rep)
        decode4 = self.lang_attn4(lang_rep=lang_rep4, vision_rep=decode4)

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
            self.channelReduce = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=1)
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
