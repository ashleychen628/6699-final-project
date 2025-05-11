import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Dict, Any
from .activations import get_activation, replace_activations

class ConvBlock(nn.Module):
    """Basic convolutional block with batch normalization"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        activation: nn.Module = nn.ReLU()
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class CustomCNN(nn.Module):
    """Custom CNN with 3 conv blocks and 2 FC layers"""
    def __init__(
        self,
        num_classes: int = 100,
        activation: nn.Module = nn.ReLU(),
        channels: list = [64, 128, 256]
    ):
        super().__init__()
        
        # Convolutional blocks
        self.features = nn.Sequential(
            ConvBlock(3, channels[0], activation=activation),
            nn.MaxPool2d(2),
            ConvBlock(channels[0], channels[1], activation=activation),
            nn.MaxPool2d(2),
            ConvBlock(channels[1], channels[2], activation=activation),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[2], 512),
            activation,
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

def get_model(
    name: str,
    num_classes: int = 100,
    activation_name: str = "relu",
    activation_kwargs: Optional[Dict[str, Any]] = None,
    pretrained: bool = False
) -> nn.Module:
    """Factory function to get model by name with specified activation function"""
    activation_kwargs = activation_kwargs or {}
    activation = get_activation(activation_name, **activation_kwargs)
    
    if name == "custom_cnn":
        model = CustomCNN(num_classes=num_classes, activation=activation)
    else:
        # Load pretrained model from torchvision
        model_fns = {
            "mobilenetv2": models.mobilenet_v2,
            "resnet18": models.resnet18,
            "densenet121": models.densenet121
        }
        
        if name not in model_fns:
            raise ValueError(f"Model {name} not supported. Choose from: {list(model_fns.keys())} + ['custom_cnn']")
        
        model = model_fns[name](pretrained=pretrained)
        
        # Replace activation functions
        model = replace_activations(model, activation)
        
        # Modify final layer for ImageNet-100
        if name == "mobilenetv2":
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        elif name == "resnet18":
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif name == "densenet121":
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    
    return model 