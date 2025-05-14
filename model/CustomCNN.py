import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

class CustomCNN(nn.Module):
    def __init__(self, activation, num_classes=10):
        super().__init__()
        self.act = deepcopy(activation)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)

        # Output size after 3x MaxPool (each halves spatial dims): 224 → 112 → 56 → 28
        self.fc1 = nn.Linear(256 * 28 * 28, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(self.act(self.bn1(self.conv1(x))))  # [B, 64, 112, 112]
        x = self.pool(self.act(self.bn2(self.conv2(x))))  # [B, 128, 56, 56]
        x = self.pool(self.act(self.bn3(self.conv3(x))))  # [B, 256, 28, 28]

        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(self.act(self.fc1(x)))
        x = self.fc2(x)
        return x
