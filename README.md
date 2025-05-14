# Activation Function Comparison on CIFAR-10 and ImageNet-100

This project systematically compares several activation functions—ReLU, Leaky ReLU, ELU, Tanh, GELU, Swish (fixed β), and Swish (trainable β)—across multiple convolutional architectures including MobileNetV2, ResNet-18, DenseNet-121, WideResNet, and a custom CNN. Experiments are conducted on both the CIFAR-10 and ImageNet-100 datasets using PyTorch, with full support for reproducible training, evaluation, and visualization.

## Features
- Modular and scalable experiment pipeline built with **PyTorch Lightning**
- Support for a wide range of **activation functions** (including trainable Swish) and **CNN architectures**
- Configurable training via **Hydra** for flexible hyperparameter management
- **Reproducible results** with fixed random seeds and deterministic settings
- Implementation of **trainable Swish activation** with a learnable β parameter


## Setup

1. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Set up the ImageNet-100 dataset:
   - Download the ImageNet-100 dataset
   - Organize it in the following structure:
```
imagenet100/
├── train/
│   ├── n01440764/
│   ├── n01443537/
│   └── ...
└── val/
    ├── n01440764/
    ├── n01443537/
    └── ...
```

3. Set the environment variable for the dataset path:
```bash
export IMAGENET100_DIR=/path/to/imagenet100
```
## Code Structure
### 1. Root Files
   * CIFAR.ipynb
     * Jupyter notebook for running experiments on CIFAR-10 using different models (CustomCNN, ResNet‑20, DenseNet‑121, WideResNet‑28‑10) and activation functions. Handles training, validation, logging, and saving.
   * ImageNet.ipynb
     * Notebook for experiments on ImageNet-10 subset using models like ResNet-18 and MobileNetV2. Includes evaluation and logging.
   * README.md
     * Project description and usage instructions (should summarize setup, training, and result interpretation).
   * requirements.txt
     * Specifies Python dependencies for reproducibility (e.g., PyTorch, torchvision, tqdm).

### 2. model/
   Contains implementations of model architectures and activation modules.
   * MobileNetV2.py – MobileNetV2 wrapper supporting Swish and other activations 
   * ResNet18.py – ResNet-18 adapted to use dynamic activations
   Subdirectories (e.g., ReLU/, Swish_trainable/, etc.) represent activation-specific model checkpoints or saved results.

### 3. getPlots/
   Contains scripts or notebooks used for visualizing results, including:
   * Accuracy curves over epochs
   * Bar plots for epoch time and memory usage
   * Comparative GPU performance across activation functions
   These files are used for evaluating and interpreting model behavior post-training.

### 4. data/
   
   Scripts or assets for retrieving, preparing, or organizing datasets, e.g., downloading or formatting the ImageNet-100 subset into PyTorch ImageFolder format.

### 5. checkpoints/
   Stores intermediate and final model checkpoints and training histories in .pt format. Each activation function typically has a subfolder containing:
   * Model weights (*.pth)
   * Training history (history_*.pt)
   * esume checkpoints (resume_*.pt)
  
## Results

Training results are saved in the `logs` directory:
- Model checkpoints
- Training checkpoints
- Training metrics plots 
