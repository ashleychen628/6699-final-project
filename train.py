import os
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from typing import Dict, Any

from models.lightning_module import ActivationExperiment
from data.datamodule import ImageNet100DataModule
from config import ExperimentConfig

def set_seed(seed: int) -> None:
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def plot_metrics(metrics: Dict[str, Any], save_dir: str) -> None:
    """Plot training and validation metrics"""
    plt.figure(figsize=(12, 4))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(metrics["train_loss"], label="Train Loss")
    plt.plot(metrics["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(metrics["train_acc"], label="Train Acc")
    plt.plot(metrics["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "metrics.png"))
    plt.close()

@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    # Set random seed
    set_seed(cfg.training.seed)
    
    # Create output directory
    os.makedirs(cfg.log_dir, exist_ok=True)
    
    # Initialize data module
    datamodule = ImageNet100DataModule(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        image_size=cfg.data.image_size,
        augmentations=cfg.data.augmentations
    )
    
    # Initialize model
    model = ActivationExperiment(
        model_name=cfg.model.name,
        activation_name=cfg.activation.name,
        activation_kwargs={
            "leaky_relu_negative_slope": cfg.activation.leaky_relu_negative_slope,
            "elu_alpha": cfg.activation.elu_alpha,
            "swish_beta": cfg.activation.swish_beta,
            "train_swish_beta": cfg.activation.train_swish_beta
        },
        num_classes=cfg.model.num_classes,
        pretrained=cfg.model.pretrained,
        learning_rate=cfg.optimizer.lr,
        momentum=cfg.optimizer.momentum,
        weight_decay=cfg.optimizer.weight_decay,
        nesterov=cfg.optimizer.nesterov,
        scheduler_T_max=cfg.scheduler.T_max,
        scheduler_eta_min=cfg.scheduler.eta_min,
        track_top5=cfg.metrics.track_top5,
        track_gradients=cfg.training.track_gradients,
        track_activations=cfg.training.track_activations,
        verbose=True,
        generate_plots=True
    )
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(cfg.log_dir, "checkpoints"),
            filename=f"{cfg.model.name}-{cfg.activation.name}-{{epoch:02d}}-{{val_acc:.4f}}",
            monitor="val_acc",
            mode="max",
            save_top_k=3
        ),
        EarlyStopping(
            monitor="val_acc",
            patience=5,
            mode="max"
        )
    ]
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        precision=cfg.training.precision,
        callbacks=callbacks,
        log_every_n_steps=cfg.training.log_every_n_steps,
        gradient_clip_val=cfg.training.gradient_clip_val
    )
    
    # Train model
    trainer.fit(model, datamodule)
    
    # Test model
    trainer.test(model, datamodule)
    
    # Plot metrics
    metrics = {
        "train_loss": [x.item() for x in model.trainer.callback_metrics["train_loss"]],
        "val_loss": [x.item() for x in model.trainer.callback_metrics["val_loss"]],
        "train_acc": [x.item() for x in model.trainer.callback_metrics["train_acc"]],
        "val_acc": [x.item() for x in model.trainer.callback_metrics["val_acc"]]
    }
    plot_metrics(metrics, cfg.log_dir)

if __name__ == "__main__":
    main() 