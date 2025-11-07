from torch import nn
import torch


class UpscaleAdd(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UpscaleAdd, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, f1, f2):
        f1_x2 = self.up(f1)
        return f1_x2 + self.conv(f2)


class UpscaleConcat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpscaleConcat, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, f1, f2):
        f1_x2 = self.up(f1)
        return torch.cat([f1_x2, f2], dim=1)


class FeatureFusion(nn.Module):

    def __init__(self, in_channels, out_channels, num_classes):
        super(FeatureFusion, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels[3], out_channels=out_channels[3], kernel_size=1),
            nn.BatchNorm2d(out_channels[3]),
            nn.ReLU(inplace=True),
            # nn.Upsample(scale_factor=2, mode='bicubic')
        )
        self.up_add0 = UpscaleAdd(in_channels[2], out_channels[2])
        self.up_add1 = UpscaleAdd(in_channels[1], out_channels[1])
        self.up_add2 = UpscaleAdd(in_channels[0], out_channels[0])

        self.up_concat0 = UpscaleConcat(sum(out_channels[2:]), sum(out_channels[2:]))
        self.up_concat1 = UpscaleConcat(sum(out_channels[1:]), sum(out_channels[1:]))
        self.up_concat2 = UpscaleConcat(sum(out_channels[0:]), sum(out_channels[0:]))

        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels=sum(out_channels), out_channels=num_classes, kernel_size=1),
            nn.BatchNorm2d(num_classes),
            nn.Upsample(scale_factor=2, mode='bilinear'),
        )

    def forward(self, f0, f1, f2, f3):
        _f3_ua = self.conv(f3)
        _f2_ua = self.up_add0(_f3_ua, f2)
        _f1_ua = self.up_add1(_f2_ua, f1)
        _f0_ua = self.up_add2(_f1_ua, f0)

        _f3_uc = self.up_concat0(_f3_ua, _f2_ua)
        _f2_uc = self.up_concat1(_f3_uc, _f1_ua)
        _f1_uc = self.up_concat2(_f2_uc, _f0_ua)

        return self.fusion(_f1_uc)