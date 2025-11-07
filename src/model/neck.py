from torch.nn import functional
from torch import nn
import torch

"""
Reference: 
    - https://github.com/Peachypie98/CBAM.git
"""


class SpatialAttentionModule(nn.Module):

    def __init__(self, bias=False):
        super(SpatialAttentionModule, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, bias=bias
        )

    def forward(self, x):
        _max = torch.max(x, 1)[0].unsqueeze(1)
        _avg = torch.mean(x, 1).unsqueeze(1)
        concat = torch.cat((_max, _avg), dim=1)
        output = self.conv(concat)
        return torch.sigmoid(output) * x


class ChannelAttentionModule(nn.Module):
    def __init__(self, channels, r):
        super(ChannelAttentionModule, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_features=channels, out_features=channels // r, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=channels // r, out_features=channels, bias=True)
        )

    def forward(self, x):
        _max = functional.adaptive_max_pool2d(x, output_size=1)
        _avg = functional.adaptive_avg_pool2d(x, output_size=1)
        b, c, *_ = x.size()
        linear_max = self.linear(_max.view(b, c)).view(b, c, 1, 1)
        linear_avg = self.linear(_avg.view(b, c)).view(b, c, 1, 1)
        output = linear_max + linear_avg
        return torch.sigmoid(output) * x


class FeatureDifferenceAttentionModule(nn.Module):

    def __init__(self, channels, r):
        super(FeatureDifferenceAttentionModule, self).__init__()
        self.cam = ChannelAttentionModule(channels=channels, r=r)
        self.sam = SpatialAttentionModule(bias=False)

    def forward(self, f1, f2):
        feature_difference = torch.abs(f1 - f2)
        spatial_attention = self.sam(feature_difference)
        channel_attention = self.cam(spatial_attention)
        return channel_attention
