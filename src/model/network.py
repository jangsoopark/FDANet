from torch import nn

from . import backbone
from . import neck
from . import head


class DenseNet121FDAMChangeDetection(nn.Module):

    def __init__(self, pretrained=False, num_classes=1):
        super(DenseNet121FDAMChangeDetection, self).__init__()
        self.backbone = backbone.DenseNet121(pretrained=pretrained)

        self.fdam_0 = neck.FeatureDifferenceAttentionModule(128, 16)
        self.fdam_1 = neck.FeatureDifferenceAttentionModule(256, 16)
        self.fdam_2 = neck.FeatureDifferenceAttentionModule(512, 16)
        self.fdam_3 = neck.FeatureDifferenceAttentionModule(1024, 16)

        self.recon = head.FeatureFusion(
            in_channels=[128, 256, 512, 1024],
            out_channels=[32, 32, 32, 32],
            num_classes=num_classes
        )

    def forward(self, x1, x2):
        f1_ = self.backbone(x1)
        f2_ = self.backbone(x2)

        _fa_3 = self.fdam_3(f1_[3], f2_[3])
        _fa_2 = self.fdam_2(f1_[2], f2_[2])
        _fa_1 = self.fdam_1(f1_[1], f2_[1])
        _fa_0 = self.fdam_0(f1_[0], f2_[0])

        change_map = self.recon(_fa_0, _fa_1, _fa_2, _fa_3)
        return change_map
    

class DenseNet169FDAMChangeDetection(nn.Module):

    def __init__(self, pretrained=False, num_classes=1):
        super(DenseNet169FDAMChangeDetection, self).__init__()
        self.backbone = backbone.DenseNet169(pretrained=pretrained)

        self.fdam_0 = neck.FeatureDifferenceAttentionModule(128, 16)
        self.fdam_1 = neck.FeatureDifferenceAttentionModule(256, 16)
        self.fdam_2 = neck.FeatureDifferenceAttentionModule(640, 16)
        self.fdam_3 = neck.FeatureDifferenceAttentionModule(1664, 16)

        self.recon = head.FeatureFusion(
            in_channels=[128, 256, 640, 1664],
            out_channels=[32, 32, 32, 32],
            num_classes=num_classes
        )

    def forward(self, x1, x2):
        f1_ = self.backbone(x1)
        f2_ = self.backbone(x2)

        _fa_3 = self.fdam_3(f1_[3], f2_[3])
        _fa_2 = self.fdam_2(f1_[2], f2_[2])
        _fa_1 = self.fdam_1(f1_[1], f2_[1])
        _fa_0 = self.fdam_0(f1_[0], f2_[0])

        change_map = self.recon(_fa_0, _fa_1, _fa_2, _fa_3)
        return change_map


class Resnet18FDAMChangeDetection(nn.Module):

    def __init__(self, pretrained=False, num_classes=1):
        super(Resnet18FDAMChangeDetection, self).__init__()
        self.backbone = backbone.ResNet18(pretrained=pretrained)

        self.fdam_0 = neck.FeatureDifferenceAttentionModule(64, 16)
        self.fdam_1 = neck.FeatureDifferenceAttentionModule(128, 16)
        self.fdam_2 = neck.FeatureDifferenceAttentionModule(256, 16)
        self.fdam_3 = neck.FeatureDifferenceAttentionModule(512, 16)

        self.recon = head.FeatureFusion(
            in_channels=[64, 128, 256, 512],
            out_channels=[32, 32, 32, 32],
            num_classes=num_classes
        )

    def forward(self, x1, x2):
        f1_ = self.backbone(x1)
        f2_ = self.backbone(x2)

        _fa_3 = self.fdam_3(f1_[3], f2_[3])
        _fa_2 = self.fdam_2(f1_[2], f2_[2])
        _fa_1 = self.fdam_1(f1_[1], f2_[1])
        _fa_0 = self.fdam_0(f1_[0], f2_[0])

        change_map = self.recon(_fa_0, _fa_1, _fa_2, _fa_3)
        return change_map
    

class Resnet34FDAMChangeDetection(nn.Module):

    def __init__(self, pretrained=False, num_classes=1):
        super(Resnet34FDAMChangeDetection, self).__init__()
        self.backbone = backbone.ResNet34(pretrained=pretrained)

        self.fdam_0 = neck.FeatureDifferenceAttentionModule(64, 16)
        self.fdam_1 = neck.FeatureDifferenceAttentionModule(128, 16)
        self.fdam_2 = neck.FeatureDifferenceAttentionModule(256, 16)
        self.fdam_3 = neck.FeatureDifferenceAttentionModule(512, 16)

        self.recon = head.FeatureFusion(
            in_channels=[64, 128, 256, 512],
            out_channels=[32, 32, 32, 32],
            num_classes=num_classes
        )

    def forward(self, x1, x2):
        f1_ = self.backbone(x1)
        f2_ = self.backbone(x2)

        _fa_3 = self.fdam_3(f1_[3], f2_[3])
        _fa_2 = self.fdam_2(f1_[2], f2_[2])
        _fa_1 = self.fdam_1(f1_[1], f2_[1])
        _fa_0 = self.fdam_0(f1_[0], f2_[0])

        change_map = self.recon(_fa_0, _fa_1, _fa_2, _fa_3)
        return change_map
    

class Resnet50FDAMChangeDetection(nn.Module):

    def __init__(self, pretrained=False, num_classes=1):
        super(Resnet50FDAMChangeDetection, self).__init__()
        self.backbone = backbone.ResNet50(pretrained=pretrained)

        self.fdam_0 = neck.FeatureDifferenceAttentionModule(256, 16)
        self.fdam_1 = neck.FeatureDifferenceAttentionModule(512, 16)
        self.fdam_2 = neck.FeatureDifferenceAttentionModule(1024, 16)
        self.fdam_3 = neck.FeatureDifferenceAttentionModule(2048, 16)

        self.recon = head.FeatureFusion(
            in_channels=[256, 512, 1024, 2048],
            out_channels=[32, 32, 32, 32],
            num_classes=num_classes
        )

    def forward(self, x1, x2):
        f1_ = self.backbone(x1)
        f2_ = self.backbone(x2)

        _fa_3 = self.fdam_3(f1_[3], f2_[3])
        _fa_2 = self.fdam_2(f1_[2], f2_[2])
        _fa_1 = self.fdam_1(f1_[1], f2_[1])
        _fa_0 = self.fdam_0(f1_[0], f2_[0])

        change_map = self.recon(_fa_0, _fa_1, _fa_2, _fa_3)
        return change_map
    

class VGG11BNFDAMChangeDetection(nn.Module):

    def __init__(self, pretrained=False, num_classes=1):
        super(VGG11BNFDAMChangeDetection, self).__init__()
        self.backbone = backbone.VGG11BN(pretrained=pretrained)

        self.fdam_0 = neck.FeatureDifferenceAttentionModule(128, 16)
        self.fdam_1 = neck.FeatureDifferenceAttentionModule(256, 16)
        self.fdam_2 = neck.FeatureDifferenceAttentionModule(512, 16)
        self.fdam_3 = neck.FeatureDifferenceAttentionModule(512, 16)

        self.recon = head.FeatureFusion(
            in_channels=[128, 256, 512, 512],
            out_channels=[32, 32, 32, 32],
            num_classes=num_classes
        )

    def forward(self, x1, x2):
        f1_ = self.backbone(x1)
        f2_ = self.backbone(x2)

        _fa_3 = self.fdam_3(f1_[3], f2_[3])
        _fa_2 = self.fdam_2(f1_[2], f2_[2])
        _fa_1 = self.fdam_1(f1_[1], f2_[1])
        _fa_0 = self.fdam_0(f1_[0], f2_[0])

        change_map = self.recon(_fa_0, _fa_1, _fa_2, _fa_3)
        return change_map


class VGG13BNFDAMChangeDetection(nn.Module):

    def __init__(self, pretrained=False, num_classes=1):
        super(VGG13BNFDAMChangeDetection, self).__init__()
        self.backbone = backbone.VGG13BN(pretrained=pretrained)

        self.fdam_0 = neck.FeatureDifferenceAttentionModule(128, 16)
        self.fdam_1 = neck.FeatureDifferenceAttentionModule(256, 16)
        self.fdam_2 = neck.FeatureDifferenceAttentionModule(512, 16)
        self.fdam_3 = neck.FeatureDifferenceAttentionModule(512, 16)

        self.recon = head.FeatureFusion(
            in_channels=[128, 256, 512, 512],
            out_channels=[32, 32, 32, 32],
            num_classes=num_classes
        )

    def forward(self, x1, x2):
        f1_ = self.backbone(x1)
        f2_ = self.backbone(x2)

        _fa_3 = self.fdam_3(f1_[3], f2_[3])
        _fa_2 = self.fdam_2(f1_[2], f2_[2])
        _fa_1 = self.fdam_1(f1_[1], f2_[1])
        _fa_0 = self.fdam_0(f1_[0], f2_[0])

        change_map = self.recon(_fa_0, _fa_1, _fa_2, _fa_3)
        return change_map


class VGG16BNFDAMChangeDetection(nn.Module):

    def __init__(self, pretrained=False, num_classes=1):
        super(VGG16BNFDAMChangeDetection, self).__init__()
        self.backbone = backbone.VGG16BN(pretrained=pretrained)

        self.fdam_0 = neck.FeatureDifferenceAttentionModule(128, 16)
        self.fdam_1 = neck.FeatureDifferenceAttentionModule(256, 16)
        self.fdam_2 = neck.FeatureDifferenceAttentionModule(512, 16)
        self.fdam_3 = neck.FeatureDifferenceAttentionModule(512, 16)

        self.recon = head.FeatureFusion(
            in_channels=[128, 256, 512, 512],
            out_channels=[32, 32, 32, 32],
            num_classes=num_classes
        )

    def forward(self, x1, x2):
        f1_ = self.backbone(x1)
        f2_ = self.backbone(x2)

        _fa_3 = self.fdam_3(f1_[3], f2_[3])
        _fa_2 = self.fdam_2(f1_[2], f2_[2])
        _fa_1 = self.fdam_1(f1_[1], f2_[1])
        _fa_0 = self.fdam_0(f1_[0], f2_[0])

        change_map = self.recon(_fa_0, _fa_1, _fa_2, _fa_3)
        return change_map


class VGG19BNFDAMChangeDetection(nn.Module):

    def __init__(self, pretrained=False, num_classes=1):
        super(VGG19BNFDAMChangeDetection, self).__init__()
        self.backbone = backbone.VGG19BN(pretrained=pretrained)

        self.fdam_0 = neck.FeatureDifferenceAttentionModule(128, 16)
        self.fdam_1 = neck.FeatureDifferenceAttentionModule(256, 16)
        self.fdam_2 = neck.FeatureDifferenceAttentionModule(512, 16)
        self.fdam_3 = neck.FeatureDifferenceAttentionModule(512, 16)

        self.recon = head.FeatureFusion(
            in_channels=[128, 256, 512, 512],
            out_channels=[32, 32, 32, 32],
            num_classes=num_classes
        )

    def forward(self, x1, x2):
        f1_ = self.backbone(x1)
        f2_ = self.backbone(x2)

        _fa_3 = self.fdam_3(f1_[3], f2_[3])
        _fa_2 = self.fdam_2(f1_[2], f2_[2])
        _fa_1 = self.fdam_1(f1_[1], f2_[1])
        _fa_0 = self.fdam_0(f1_[0], f2_[0])

        change_map = self.recon(_fa_0, _fa_1, _fa_2, _fa_3)
        return change_map
    

class EfficientNetB3FDAMChangeDetection(nn.Module):

    def __init__(self, pretrained=False, num_classes=1):
        super(EfficientNetB3FDAMChangeDetection, self).__init__()
        self.backbone = backbone.EfficientNetB3(pretrained=pretrained)

        self.fdam_0 = neck.FeatureDifferenceAttentionModule(24, 4)
        self.fdam_1 = neck.FeatureDifferenceAttentionModule(32, 4)
        self.fdam_2 = neck.FeatureDifferenceAttentionModule(48, 4)
        self.fdam_3 = neck.FeatureDifferenceAttentionModule(1536, 64)

        self.recon = head.FeatureFusion(
            in_channels=[24, 32, 48, 1536],
            out_channels=[16, 16, 16, 16],
            num_classes=num_classes
        )

    def forward(self, x1, x2):
        f1_ = self.backbone(x1)
        f2_ = self.backbone(x2)

        _fa_3 = self.fdam_3(f1_[3], f2_[3])
        _fa_2 = self.fdam_2(f1_[2], f2_[2])
        _fa_1 = self.fdam_1(f1_[1], f2_[1])
        _fa_0 = self.fdam_0(f1_[0], f2_[0])

        change_map = self.recon(_fa_0, _fa_1, _fa_2, _fa_3)
        return change_map


class EfficientNetB4FDAMChangeDetection(nn.Module):

    def __init__(self, pretrained=False, num_classes=1):
        super(EfficientNetB4FDAMChangeDetection, self).__init__()
        self.backbone = backbone.EfficientNetB4(pretrained=pretrained)

        self.fdam_0 = neck.FeatureDifferenceAttentionModule(24, 4)
        self.fdam_1 = neck.FeatureDifferenceAttentionModule(32, 4)
        self.fdam_2 = neck.FeatureDifferenceAttentionModule(56, 4)
        self.fdam_3 = neck.FeatureDifferenceAttentionModule(1792, 64)

        self.recon = head.FeatureFusion(
            in_channels=[24, 32, 56, 1792],
            out_channels=[16, 16, 16, 16],
            num_classes=num_classes
        )

    def forward(self, x1, x2):
        f1_ = self.backbone(x1)
        f2_ = self.backbone(x2)

        _fa_3 = self.fdam_3(f1_[3], f2_[3])
        _fa_2 = self.fdam_2(f1_[2], f2_[2])
        _fa_1 = self.fdam_1(f1_[1], f2_[1])
        _fa_0 = self.fdam_0(f1_[0], f2_[0])

        change_map = self.recon(_fa_0, _fa_1, _fa_2, _fa_3)
        return change_map


class EfficientNetV2SFDAMChangeDetection(nn.Module):

    def __init__(self, pretrained=False, num_classes=1):
        super(EfficientNetV2SFDAMChangeDetection, self).__init__()
        self.backbone = backbone.EfficientNetV2S(pretrained=pretrained)

        self.fdam_0 = neck.FeatureDifferenceAttentionModule(24, 4)
        self.fdam_1 = neck.FeatureDifferenceAttentionModule(48, 4)
        self.fdam_2 = neck.FeatureDifferenceAttentionModule(64, 4)
        self.fdam_3 = neck.FeatureDifferenceAttentionModule(1280, 64)

        self.recon = head.FeatureFusion(
            in_channels=[24, 48, 64, 1280],
            out_channels=[16, 16, 16, 16],
            num_classes=num_classes
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x1, x2):
        f1_ = self.backbone(x1)
        f2_ = self.backbone(x2)

        _fa_3 = self.fdam_3(f1_[3], f2_[3])
        _fa_2 = self.fdam_2(f1_[2], f2_[2])
        _fa_1 = self.fdam_1(f1_[1], f2_[1])
        _fa_0 = self.fdam_0(f1_[0], f2_[0])

        change_map = self.recon(_fa_0, _fa_1, _fa_2, _fa_3)
        return change_map



architecture = {
    'VGG11BN': VGG11BNFDAMChangeDetection,
    'VGG13BN': VGG13BNFDAMChangeDetection,
    'VGG16BN': VGG16BNFDAMChangeDetection,
    'VGG19BN': VGG19BNFDAMChangeDetection,
    'ResNet18': Resnet18FDAMChangeDetection,
    'ResNet34': Resnet34FDAMChangeDetection,
    
    'EfficientNetB3': EfficientNetB3FDAMChangeDetection,
    'EfficientNetB4': EfficientNetB4FDAMChangeDetection,
    'EfficientNetV2S': EfficientNetV2SFDAMChangeDetection,
    'DenseNet121': DenseNet121FDAMChangeDetection,
    'DenseNet169': DenseNet169FDAMChangeDetection,
}