import torch
import torch.nn as nn
from copy import deepcopy

class ConvBNActivation(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride, groups=1, activation=None):
        assert isinstance(activation, nn.Module), f"Expected nn.Module instance, got {type(activation)}"
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            deepcopy(activation)
        )

class InvertedResidual(nn.Module):
    def __init__(self, in_planes, out_planes, stride, expand_ratio, activation=None):
        assert isinstance(activation, nn.Module), f"Expected nn.Module instance, got {type(activation)}"
        super().__init__()
        self.stride = stride
        hidden_dim = in_planes * expand_ratio
        self.use_res_connect = self.stride == 1 and in_planes == out_planes

        layers = []
        if expand_ratio != 1:
            # Pointwise
            layers.append(ConvBNActivation(in_planes, hidden_dim, kernel_size=1, stride=1, activation=activation))
        # Depthwise
        layers.append(ConvBNActivation(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim, activation=activation))
        # Pointwise-linear
        layers.append(nn.Conv2d(hidden_dim, out_planes, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(out_planes))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)

class MobileNetV2(nn.Module):
    def __init__(self, activation, num_classes=20):
        super().__init__()
        assert isinstance(activation, nn.Module), "Activation must be an nn.Module instance"
        self.activation = activation  # no deepcopy here
        layers = []
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        # Initial Conv
        layers.append(ConvBNActivation(3, input_channel, kernel_size=3, stride=2, activation=self.activation))

        # Inverted Residual Blocks
        for t, c, n, s in interverted_residual_setting:
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedResidual(input_channel, c, stride, t, activation=self.activation))
                input_channel = c

        # Last layers
        layers.append(ConvBNActivation(input_channel, last_channel, kernel_size=1, stride=1, activation=self.activation))
        self.features = nn.Sequential(*layers)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(last_channel, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x).flatten(1)
        x = self.classifier(x)
        return x
