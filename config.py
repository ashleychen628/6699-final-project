from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union
from omegaconf import MISSING

@dataclass
class ActivationConfig:
    name: str = MISSING  # "relu", "leaky_relu", "elu", "tanh", "swish", "trainable_swish"
    swish_beta: float = 1.0
    leaky_relu_negative_slope: float = 0.01
    elu_alpha: float = 1.0

@dataclass
class ModelConfig:
    name: str = MISSING  # "resnet18", "mobilenetv2", "densenet121", "custom_cnn"
    pretrained: bool = False
    activation: Optional[ActivationConfig] = None

    def __post_init__(self):
        if self.activation is None:
            self.activation = ActivationConfig()

@dataclass
class OptimizerConfig:
    learning_rate: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 1e-4
    nesterov: bool = True

@dataclass
class SchedulerConfig:
    warmup_epochs: int = 5
    warmup_start_lr: float = 0.01
    T_max: int = 90
    eta_min: float = 0.0

@dataclass
class DataConfig:
    # Set this to your ImageNet-100 dataset directory
    # You can either:
    # 1. Set the IMAGENET100_DIR environment variable, or
    # 2. Override this value via command line: data.data_dir=/path/to/imagenet-100
    data_dir: str = MISSING  # Required: path to ImageNet-100 dataset
    batch_size: int = 256
    num_workers: int = 8
    image_size: int = 224
    train_transform: Dict[str, Any] = None
    val_transform: Dict[str, Any] = None

@dataclass
class TrainingConfig:
    max_epochs: int = 90
    seed: int = 42
    gpus: int = 1
    precision: int = 32
    use_wandb: bool = False
    log_dir: str = "logs"
    track_convergence: bool = True
    convergence_threshold: float = 0.95
    track_gpu_memory: bool = True
    track_gradients: bool = True
    track_activations: bool = True
    track_top5: bool = True
    gradient_histogram_bins: int = 50
    activation_stats_freq: int = 100
    verbose: bool = True
    generate_plots: bool = True

@dataclass
class ExperimentConfig:
    model: ModelConfig = MISSING
    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    data: DataConfig = DataConfig()
    training: TrainingConfig = TrainingConfig()
    experiment_name: Optional[str] = None
    log_dir: str = "logs"
    use_wandb: bool = False

    def __post_init__(self):
        if self.experiment_name is None:
            self.experiment_name = f"{self.model.name}-{self.model.activation.name}" 