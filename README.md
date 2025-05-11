# Activation Function Comparison on ImageNet-100

This project compares different activation functions (ReLU, Leaky ReLU, ELU, Tanh, GELU, Swish, Trainable Swish) across various CNN architectures (MobileNetV2, ResNet-18, DenseNet-121, Custom CNN) on the ImageNet-100 dataset using PyTorch Lightning.

## Features

- Modular and scalable experiment setup using PyTorch Lightning
- Support for multiple activation functions and CNN architectures
- Configurable training parameters using Hydra
- Wandb integration for experiment tracking
- Multi-GPU training support
- Reproducible results with fixed seeds
- Trainable Swish activation with learnable beta parameter

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

## Usage

### Basic Training

To train a model with default settings (ResNet-18 with ReLU):
```bash
python train.py
```

### Custom Configuration

You can override any configuration parameter using the command line:

```bash
python train.py model.name=mobilenetv2 activation.name=swish training.max_epochs=30
```

### Multi-GPU Training

To train using multiple GPUs:
```bash
python train.py training.devices=4
```

### Wandb Integration

The project uses Weights & Biases for experiment tracking. To disable it:
```bash
python train.py use_wandb=false
```

## Configuration

The default configuration is in `config.yaml`. Key parameters include:

- Model selection: `model.name` (mobilenetv2, resnet18, densenet121, custom_cnn)
- Activation function: `activation.name` (relu, leaky_relu, elu, tanh, gelu, swish, trainable_swish)
- Training parameters: learning rate, batch size, epochs, etc.
- Hardware settings: GPU devices, precision, etc.

## Results

Training results are saved in the `logs` directory:
- Model checkpoints
- Training metrics plots (if not using wandb)
- Wandb logs (if enabled)

## License

MIT License