import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union

class Swish(nn.Module):
    """Swish activation function: x * sigmoid(beta * x)"""
    def __init__(self, beta: float = 1.0, trainable: bool = False):
        super().__init__()
        if trainable:
            self.beta = nn.Parameter(torch.tensor(beta))
        else:
            self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.beta * x)

def get_activation(name: str, **kwargs) -> nn.Module:
    """Factory function to get activation function by name"""
    activations = {
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(negative_slope=kwargs.get("leaky_relu_negative_slope", 0.01)),
        "elu": nn.ELU(alpha=kwargs.get("elu_alpha", 1.0)),
        "tanh": nn.Tanh(),
        "gelu": nn.GELU(),
        "swish": Swish(beta=kwargs.get("swish_beta", 1.0), trainable=False),
        "trainable_swish": Swish(beta=kwargs.get("swish_beta", 1.0), trainable=True)
    }
    
    if name not in activations:
        raise ValueError(f"Activation {name} not supported. Choose from: {list(activations.keys())}")
    
    return activations[name]

def replace_activations(model: nn.Module, activation: nn.Module) -> nn.Module:
    """Replace all ReLU activations in a model with the specified activation function"""
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU):
            setattr(model, name, activation)
        else:
            replace_activations(module, activation)
    return model 