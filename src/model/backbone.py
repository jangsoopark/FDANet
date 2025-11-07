from torchvision import models
from torch import nn


def _densenet121(pretrained=False):
    _weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
    _model = models.densenet121(weights=_weights).features
    # _model.conv0.stride = (1, 1)
    del _model.transition1.pool
    return _model


def _densenet169(pretrained=False):
    _weights = models.DenseNet169_Weights.DEFAULT if pretrained else None
    _model = models.densenet169(weights=_weights).features
    _model.conv0.stride = (1, 1)
    return _model


def _resnet18(pretrained=False):
    _weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    _model = models.resnet18(weights=_weights)
    _model.conv1.stride = (1, 1)
    return _model


def _resnet34(pretrained=False):
    _weights = models.ResNet34_Weights.DEFAULT if pretrained else None
    _model = models.resnet34(weights=_weights)
    _model.conv1.stride = (1, 1)
    return _model


def _resnet50(pretrained=False):
    _weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    _model = models.resnet50(weights=_weights)
    _model.conv1.stride = (1, 1)
    return _model


def _efficientnet_b3(pretrained=False):
    _weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
    _model = models.efficientnet_b3(weights=_weights).features
    _model[6][0].block[1][0].stride = (1, 1)
    return _model


def _efficientnet_b4(pretrained=False):
    _weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
    _model = models.efficientnet_b4(weights=_weights).features
    _model[6][0].block[1][0].stride = (1, 1)
    return _model


def _efficientnet_v2_s(pretrained=False):
    _weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
    _model = models.efficientnet_v2_s(weights=_weights).features
    _model[6][0].block[1][0].stride = (1, 1)
    return _model


class DenseNet121(nn.Module):
    def __init__(self, pretrained=False):
        super(DenseNet121, self).__init__()
        _model = _densenet121(pretrained=pretrained)
        self.stem = nn.Sequential(
            _model.conv0,
            _model.norm0,
            _model.relu0,
            # _model.pool0,
        )

        self.block1 = nn.Sequential(
            _model.denseblock1,
            _model.transition1,
        )

        self.block2 = nn.Sequential(
            _model.denseblock2,
            _model.transition2,
        )

        self.block3 = nn.Sequential(
            _model.denseblock3,
            _model.transition3,
        )

        self.block4 = nn.Sequential(
            _model.denseblock4,
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        f = self.stem(x)        # ?, 64, 256, 256
        f1 = self.block1(f)     # ?, 128, 128, 128
        f2 = self.block2(f1)    # ?, 256, 64, 64
        f3 = self.block3(f2)    # ?, 512, 32, 32
        f4 = self.block4(f3)    # ?, 1024, 16, 16
        return f1, f2, f3, f4


class DenseNet169(nn.Module):
    def __init__(self, pretrained=False):
        super(DenseNet169, self).__init__()
        _model = _densenet169(pretrained=pretrained)
        self.stem = nn.Sequential(
            _model.conv0,
            _model.norm0, 
            _model.relu0,
        )

        self.block1 = nn.Sequential(
            _model.denseblock1,
            _model.transition1,
            _model.pool0,
        )

        self.block2 = nn.Sequential(
            _model.denseblock2,
            _model.transition2,
        )

        self.block3 = nn.Sequential(
            _model.denseblock3,
            _model.transition3,
        )

        self.block4 = nn.Sequential(
            _model.denseblock4,
            _model.norm5,
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        f = self.stem(x)
        f1 = self.block1(f)     # ?, 128, 128, 128
        f2 = self.block2(f1)    # ?, 256, 64, 64
        f3 = self.block3(f2)    # ?, 640, 32, 32
        f4 = self.block4(f3)    # ?, 1664, 16, 16
        return f1, f2, f3, f4


class ResNet18(nn.Module):
    
    def __init__(self, pretrained=False):
        super(ResNet18, self).__init__()
        _model = _resnet18(pretrained=pretrained)
        
        self.stem = nn.Sequential(
            _model.conv1,
            _model.bn1,
            _model.relu, 
            _model.maxpool
        )

        self.block1 = _model.layer1
        self.block2 = _model.layer2
        self.block3 = _model.layer3
        self.block4 = _model.layer4

    def forward(self, x):
        f = self.stem(x)        # ?, 64,    256,    256
        f1 = self.block1(f)     # ?, 64,    128,    128
        f2 = self.block2(f1)    # ?, 128,   64,     64
        f3 = self.block3(f2)    # ?, 256,   32,     32
        f4 = self.block4(f3)    # ?, 512,   16,     16
        return f1, f2, f3, f4
    

class ResNet34(nn.Module):
    
    def __init__(self, pretrained=False):
        super(ResNet34, self).__init__()
        _model = _resnet34(pretrained=pretrained)
        
        self.stem = nn.Sequential(
            _model.conv1,
            _model.bn1,
            _model.relu, 
            _model.maxpool
        )

        self.block1 = _model.layer1
        self.block2 = _model.layer2
        self.block3 = _model.layer3
        self.block4 = _model.layer4

    def forward(self, x):
        f = self.stem(x)        # ?, 64,    256,    256
        f1 = self.block1(f)     # ?, 64,    128,    128
        f2 = self.block2(f1)    # ?, 128,   64,     64
        f3 = self.block3(f2)    # ?, 256,   32,     32
        f4 = self.block4(f3)    # ?, 512,   16,     16
        return f1, f2, f3, f4
    

class ResNet50(nn.Module):
    
    def __init__(self, pretrained=False):
        super(ResNet50, self).__init__()
        _model = _resnet50(pretrained=pretrained)
        
        self.stem = nn.Sequential(
            _model.conv1,
            _model.bn1,
            _model.relu, 
            _model.maxpool
        )

        self.block1 = _model.layer1
        self.block2 = _model.layer2
        self.block3 = _model.layer3
        self.block4 = _model.layer4

    def forward(self, x):
        f = self.stem(x)        # ?, 64,    256,    256
        f1 = self.block1(f)     # ?, 256,    128,    128
        f2 = self.block2(f1)    # ?, 512,   64,     64
        f3 = self.block3(f2)    # ?, 1024,   32,     32
        f4 = self.block4(f3)    # ?, 2048,   16,     16
        return f1, f2, f3, f4


class VGG11BN(nn.Module):

    def __init__(self, pretrained=False):
        super(VGG11BN, self).__init__()
        _weights = models.VGG11_BN_Weights.DEFAULT if pretrained else None
        _model = models.vgg11_bn(weights=_weights)

        self.stem = _model.features[0: 2]
        self.block1 = _model.features[2: 6]
        self.block2 = _model.features[6: 13]
        self.block3 = _model.features[13: 20]
        self.block4 = _model.features[20: 28]

    def forward(self, x):
        f = self.stem(x)
        f1 = self.block1(f)     # ?, 128, 128, 128
        f2 = self.block2(f1)    # ?, 256, 64, 64
        f3 = self.block3(f2)    # ?, 512, 32, 32
        f4 = self.block4(f3)    # ?, 512, 16, 16
        return f1, f2, f3, f4


class VGG13BN(nn.Module):

    def __init__(self, pretrained=False):
        super(VGG13BN, self).__init__()
        _weights = models.VGG13_BN_Weights.DEFAULT if pretrained else None
        _model = models.vgg13_bn(weights=_weights)

        
        self.stem = _model.features[0: 5]
        self.block1 = _model.features[5: 12]
        self.block2 = _model.features[12: 19]
        self.block3 = _model.features[19: 26]
        self.block4 = _model.features[26: 33]

    def forward(self, x):
        f = self.stem(x)
        f1 = self.block1(f)     # ?, 128, 128, 128
        f2 = self.block2(f1)    # ?, 256, 64, 64
        f3 = self.block3(f2)    # ?, 512, 32, 32
        f4 = self.block4(f3)    # ?, 512, 16, 16
        return f1, f2, f3, f4


class VGG16BN(nn.Module):

    def __init__(self, pretrained=False):
        super(VGG16BN, self).__init__()
        _weights = models.VGG16_BN_Weights.DEFAULT if pretrained else None
        _model = models.vgg16_bn(weights=_weights)
        
        self.stem = _model.features[0: 5]
        self.block1 = _model.features[5: 12]
        self.block2 = _model.features[12: 22]
        self.block3 = _model.features[22: 32]
        self.block4 = _model.features[32: 42]

    def forward(self, x):
        f = self.stem(x)
        f1 = self.block1(f)     # ?, 128, 128, 128
        f2 = self.block2(f1)    # ?, 256, 64, 64
        f3 = self.block3(f2)    # ?, 512, 32, 32
        f4 = self.block4(f3)    # ?, 512, 16, 16
        return f1, f2, f3, f4


class VGG19BN(nn.Module):

    def __init__(self, pretrained=False):
        super(VGG19BN, self).__init__()
        _weights = models.VGG19_BN_Weights.DEFAULT if pretrained else None
        _model = models.vgg19_bn(weights=_weights)
        
        self.stem = _model.features[0: 5]
        self.block1 = _model.features[5: 12]
        self.block2 = _model.features[12: 22]
        self.block3 = _model.features[22: 32]
        self.block4 = _model.features[32: 42]

    def forward(self, x):
        f = self.stem(x)
        f1 = self.block1(f)     # ?, 128, 128, 128
        f2 = self.block2(f1)    # ?, 256, 64, 64
        f3 = self.block3(f2)    # ?, 512, 32, 32
        f4 = self.block4(f3)    # ?, 512, 16, 16
        return f1, f2, f3, f4


class EfficientNetB3(nn.Module):

    def __init__(self, pretrained=False):
        super(EfficientNetB3, self).__init__()
        _model = _efficientnet_b3(pretrained=pretrained)
        
        self.stem = _model[0]      # ?, 40,    128,    128
        self.block1 = _model[1]    # ?, 24,    128,    128
        self.block2 = _model[2]    # ?, 32,    64,     64
        self.block3 = _model[3]    # ?, 48,    32,     32
        self.block4 = _model[4:]  # ?, 1536,   16,     16

    def forward(self, x):
        f = self.stem(x)
        f1 = self.block1(f)
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        f4 = self.block4(f3)
        
        return f1, f2, f3, f4
    

class EfficientNetB4(nn.Module):

    def __init__(self, pretrained=False):
        super(EfficientNetB4, self).__init__()
        _model = _efficientnet_b4(pretrained=pretrained)

        self.stem = _model[0]      # ?, 48,    128,    128
        self.block1 = _model[1]    # ?, 24,    128,    128
        self.block2 = _model[2]    # ?, 32,    64,     64
        self.block3 = _model[3]    # ?, 56,   32,     32
        self.block4 = _model[4:]   # ?, 1792,  16,     16

    def forward(self, x):
        f = self.stem(x)
        f1 = self.block1(f)
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        f4 = self.block4(f3)
        
        return f1, f2, f3, f4


class EfficientNetV2S(nn.Module):

    def __init__(self, pretrained=False):
        super(EfficientNetV2S, self).__init__()
        _model = _efficientnet_v2_s(pretrained=pretrained)

        self.stem = _model[0]      # ?, 24,    256,    256
        self.block1 = _model[1]    # ?, 24,    128,    128
        self.block2 = _model[2]    # ?, 48,    64,     64
        self.block3 = _model[3]    # ?, 64,   32,     32
        self.block4 = _model[4:]  # ?, 1280,  16,     16

    def forward(self, x):
        f = self.stem(x)
        f1 = self.block1(f)
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        f4 = self.block4(f3)
        
        return f1, f2, f3, f4

