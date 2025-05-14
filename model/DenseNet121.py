import torch
import torch.nn as nn
from torchvision.models import densenet121
from copy import deepcopy

def replace_activation(module, activation):
    """
    Recursively replace all ReLU activations in a module with the provided activation instance.
    Assumes `activation` is an instance of nn.Module (e.g., nn.ReLU()).
    """
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, deepcopy(activation))
        else:
            replace_activation(child, activation)

class DenseNet121(nn.Module):
    def __init__(self, activation, num_classes=10):
        super().__init__()
        self.model = densenet121(pretrained=False)
        replace_activation(self.model, activation)
        
        # Replace final classifier layer
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
