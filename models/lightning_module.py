import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import gc
import time
from collections import defaultdict
import torch.nn.utils as utils
import os
import json
from datetime import datetime
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from .models import get_model

class MetricsLogger:
    """Logger for metrics that saves to files"""
    def __init__(self, model_name: str, activation_name: str):
        # Create logs directory if it doesn't exist
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        # Create metrics file
        self.metrics_file = self.logs_dir / f"{model_name}-{activation_name}_metrics.txt"
        self.summary_file = self.logs_dir / f"{model_name}-{activation_name}_summary.txt"
        
        # Initialize metrics file with header
        with open(self.metrics_file, 'w') as f:
            f.write(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n")
            f.write("Epoch\tTrain Loss\tTrain Acc\tVal Loss\tVal Acc\tLR\t\tTime\tGPU Memory\n")
            f.write("-" * 80 + "\n")
        
        print(f"Metrics will be saved to: {self.metrics_file}")
        print(f"Summary will be saved to: {self.summary_file}")
    
    def log_epoch_metrics(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Log metrics for a single epoch"""
        with open(self.metrics_file, 'a') as f:
            f.write(f"{epoch}\t")
            f.write(f"{metrics['train_loss']:.4f}\t")
            f.write(f"{metrics['train_acc']:.4f}\t")
            f.write(f"{metrics['val_loss']:.4f}\t")
            f.write(f"{metrics['val_acc']:.4f}\t")
            f.write(f"{metrics['lr']:.6f}\t")
            f.write(f"{metrics['epoch_time']:.2f}s\t")
            f.write(f"{metrics['gpu_memory']:.1f}MB\n")
        
        # Print metrics to console for immediate feedback
        print(f"\nEpoch {epoch} metrics:")
        print(f"Train Loss: {metrics['train_loss']:.4f}")
        print(f"Train Acc: {metrics['train_acc']:.4f}")
        print(f"Val Loss: {metrics['val_loss']:.4f}")
        print(f"Val Acc: {metrics['val_acc']:.4f}")
        print(f"Learning Rate: {metrics['lr']:.6f}")
        print(f"Epoch Time: {metrics['epoch_time']:.2f}s")
        print(f"GPU Memory: {metrics['gpu_memory']:.1f}MB")
        print("-" * 50)
    
    def log_summary(self, summary: Dict[str, Any]) -> None:
        """Log final summary of the experiment"""
        with open(self.summary_file, 'w') as f:
            f.write(f"Experiment: {model_name}-{activation_name}\n")
            f.write(f"Completed at: {datetime.now()}\n")
            f.write("=" * 80 + "\n\n")
            
            # Write configuration
            f.write("Configuration:\n")
            f.write("-" * 40 + "\n")
            for key, value in summary['config'].items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # Write final metrics
            f.write("Final Metrics:\n")
            f.write("-" * 40 + "\n")
            for key, value in summary['metrics'].items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")
            
            # Write convergence info if available
            if 'convergence_epoch' in summary['metrics']:
                f.write(f"\nConvergence: Reached 95% of final accuracy at epoch {summary['metrics']['convergence_epoch']}\n")
            
            # Write performance metrics
            f.write("\nPerformance Metrics:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Average epoch time: {summary['metrics']['avg_epoch_time']:.2f}s\n")
            f.write(f"Peak GPU memory: {summary['metrics']['peak_gpu_memory_mb']:.1f}MB\n")

class WarmupCosineScheduler(SequentialLR):
    """Learning rate scheduler with warmup and cosine annealing"""
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        warmup_start_lr: float,
        T_max: int,
        eta_min: float = 0.0
    ):
        # Warmup scheduler
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,  # Start at 10% of the target learning rate
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        
        # Cosine annealing scheduler
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=T_max - warmup_epochs,
            eta_min=eta_min
        )
        
        super().__init__(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )

class PlotGenerator:
    """Utility class for generating training plots"""
    def __init__(self, log_dir: str, experiment_name: str):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.plots_dir = os.path.join(log_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn')
        sns.set_palette("husl")
    
    def plot_training_curves(self, metrics: Dict[str, List[float]], epochs: List[int]) -> None:
        """Plot training and validation curves"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # Plot loss curves
        ax1.plot(epochs, metrics['train_loss'], label='Train Loss', marker='o', markersize=2)
        ax1.plot(epochs, metrics['val_loss'], label='Val Loss', marker='o', markersize=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy curves
        ax2.plot(epochs, metrics['train_acc'], label='Train Acc', marker='o', markersize=2)
        ax2.plot(epochs, metrics['val_acc'], label='Val Acc', marker='o', markersize=2)
        if 'train_top5_acc' in metrics:
            ax2.plot(epochs, metrics['train_top5_acc'], label='Train Top-5', marker='o', markersize=2)
            ax2.plot(epochs, metrics['val_top5_acc'], label='Val Top-5', marker='o', markersize=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f"{self.experiment_name}_training_curves.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_learning_rate(self, lrs: List[float], epochs: List[int]) -> None:
        """Plot learning rate schedule"""
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, lrs, marker='o', markersize=2)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, f"{self.experiment_name}_learning_rate.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_gpu_memory(self, memory_usage: List[float], epochs: List[int]) -> None:
        """Plot GPU memory usage"""
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, memory_usage, marker='o', markersize=2)
        plt.xlabel('Epoch')
        plt.ylabel('GPU Memory (MB)')
        plt.title('GPU Memory Usage')
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, f"{self.experiment_name}_gpu_memory.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_epoch_times(self, times: List[float], epochs: List[int]) -> None:
        """Plot epoch training times"""
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, times, marker='o', markersize=2)
        plt.xlabel('Epoch')
        plt.ylabel('Time (seconds)')
        plt.title('Training Time per Epoch')
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, f"{self.experiment_name}_epoch_times.png"), dpi=300, bbox_inches='tight')
        plt.close()

class ActivationExperiment(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        activation_name: str,
        activation_kwargs: Dict[str, Any],
        num_classes: int = 100,
        pretrained: bool = True,
        learning_rate: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        nesterov: bool = True,
        scheduler_T_max: int = 10,
        scheduler_eta_min: float = 0.0,
        track_top5: bool = True,
        track_gradients: bool = True,
        track_activations: bool = True,
        verbose: bool = True,
        generate_plots: bool = True
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize model with custom activation
        self.model = get_model(
            name=model_name,
            activation_name=activation_name,
            activation_kwargs=activation_kwargs,
            num_classes=num_classes,
            pretrained=pretrained
        )
        
        # Initialize loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize best validation accuracy
        self.best_val_acc = torch.tensor(0.0)
        
        # Training parameters
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.scheduler_T_max = scheduler_T_max
        self.scheduler_eta_min = scheduler_eta_min
        
        # Initialize metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.train_top5_accs = []
        self.val_top5_accs = []
        self.epoch_times = []
        self.peak_gpu_memory = 0.0
        
        # For tracking gradients and activations
        self.track_gradients = track_gradients
        self.track_activations = track_activations
        self.gradient_stats = defaultdict(list)
        self.activation_stats = defaultdict(lambda: {"mean": [], "std": []})
        self.gradient_histogram_bins = 50
        self.activation_stats_freq = 100
        
        # Initialize metrics logger
        self.metrics_logger = MetricsLogger(model_name=model_name, activation_name=activation_name)
        
        # Register hooks for gradient and activation tracking
        if self.track_gradients or self.track_activations:
            self._register_hooks()
        
        # Initialize plot generator
        self.plot_generator = PlotGenerator(log_dir="logs", experiment_name=f"{model_name}-{activation_name}")
        
        # Set verbose mode
        self.verbose = verbose
        self.generate_plots = generate_plots

    def _register_hooks(self):
        """Register hooks to track gradients and activations"""
        self.gradients = {}
        
        def grad_hook(name):
            def hook(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    self.gradients[name] = grad_output[0].detach()
            return hook
        
        # Register hooks for each layer
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                module.register_full_backward_hook(grad_hook(name))

        def get_activation_hook(name):
            def activation_hook(module, input, output):
                if self.track_activations and self.global_step % self.activation_stats_freq == 0:
                    if isinstance(output, torch.Tensor):
                        act = output.detach()
                        self.activation_stats[name]["mean"].append(act.mean().item())
                        self.activation_stats[name]["std"].append(act.std().item())
            return activation_hook

        # Register hooks for all named modules
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                module.register_forward_hook(get_activation_hook(name))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate accuracies
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # Calculate Top-5 accuracy for ImageNet
        if self.hparams.track_top5:
            _, top5_preds = torch.topk(logits, k=5, dim=1)
            top5_acc = torch.any(top5_preds == y.unsqueeze(1), dim=1).float().mean()
            self.log("train_top5_acc", top5_acc, on_step=True, on_epoch=True, prog_bar=True)
        
        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        
        # Print batch progress if verbose
        if self.verbose and batch_idx % 10 == 0:  # Print every 10 batches
            print(f"\rEpoch {self.current_epoch} [{batch_idx}/{self.trainer.num_training_batches}] "
                  f"Loss: {loss.item():.4f} Acc: {acc.item():.4f}", end="")
            sys.stdout.flush()
        
        # Store metrics for convergence tracking
        if batch_idx == 0:
            self.train_losses.append(loss.item())
            self.train_accs.append(acc.item())
            if self.hparams.track_top5:
                self.train_top5_accs.append(top5_acc.item())
        
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate accuracies
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # Calculate Top-5 accuracy for ImageNet
        if self.hparams.track_top5:
            _, top5_preds = torch.topk(logits, k=5, dim=1)
            top5_acc = torch.any(top5_preds == y.unsqueeze(1), dim=1).float().mean()
            self.log("val_top5_acc", top5_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        
        # Print validation progress if verbose
        if self.verbose and batch_idx % 5 == 0:  # Print every 5 batches
            print(f"\rValidating [{batch_idx}/{self.trainer.num_val_batches}] "
                  f"Loss: {loss.item():.4f} Acc: {acc.item():.4f}", end="")
            sys.stdout.flush()
        
        # Store metrics for convergence tracking
        if batch_idx == 0:
            self.val_losses.append(loss.item())
            self.val_accs.append(acc.item())
            if self.hparams.track_top5:
                self.val_top5_accs.append(top5_acc.item())
        
        # Update best validation accuracy
        if acc > self.best_val_acc:
            self.best_val_acc = acc
            print(f"\nNew best validation accuracy: {acc.item():.4f}")

    def on_train_epoch_start(self) -> None:
        self.epoch_start_time = time.time()
        if self.verbose:
            print(f"\nEpoch {self.current_epoch} starting...")
            print(f"Learning rate: {self.trainer.optimizers[0].param_groups[0]['lr']:.6f}")

    def on_train_epoch_end(self) -> None:
        # Calculate epoch time
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        
        # Get current metrics
        current_metrics = {
            'train_loss': self.train_losses[-1],
            'train_acc': self.train_accs[-1],
            'val_loss': self.val_losses[-1],
            'val_acc': self.val_accs[-1],
            'lr': self.trainer.optimizers[0].param_groups[0]["lr"],
            'epoch_time': epoch_time,
            'gpu_memory': 0.0  # No GPU memory tracking in CPU mode
        }
        
        # Log metrics to file
        self.metrics_logger.log_epoch_metrics(self.current_epoch, current_metrics)
        
        # Print epoch summary if verbose
        if self.verbose:
            print(f"\nEpoch {self.current_epoch} completed in {epoch_time:.2f}s")
            print(f"Train Loss: {current_metrics['train_loss']:.4f} "
                  f"Train Acc: {current_metrics['train_acc']:.4f}")
            print(f"Val Loss: {current_metrics['val_loss']:.4f} "
                  f"Val Acc: {current_metrics['val_acc']:.4f}")
            if self.hparams.track_top5:
                print(f"Train Top-5: {self.train_top5_accs[-1]:.4f} "
                      f"Val Top-5: {self.val_top5_accs[-1]:.4f}")
            print("-" * 80)
        
        # Log gradient statistics
        if self.track_gradients and self.current_epoch == 0:
            for name, stats in self.gradient_stats.items():
                if "norm" in name:
                    # Store gradient norms in metrics
                    self.gradient_stats[f"{name}_mean"] = np.mean(stats)
                elif "hist" in name:
                    # Store gradient histograms
                    hist, bins = np.histogram(np.concatenate(stats), bins=self.gradient_histogram_bins)
                    self.gradient_stats[f"{name}_histogram"] = (hist, bins)
        
        # Log activation statistics
        if self.track_activations:
            for name, stats in self.activation_stats.items():
                # Store activation statistics
                self.activation_stats[name]["mean_epoch"] = np.mean(stats["mean"])
                self.activation_stats[name]["std_epoch"] = np.mean(stats["std"])

    def on_train_start(self) -> None:
        if self.verbose:
            print("\nStarting training...")
            print(f"Model: {self.hparams.model_name}")
            print(f"Activation: {self.hparams.activation_name}")
            
            # Get batch size from datamodule if available
            try:
                batch_size = self.trainer.datamodule.hparams.batch_size
                print(f"Batch size: {batch_size}")
            except (AttributeError, KeyError):
                print("Batch size: Not available in datamodule")
            
            print(f"Learning rate: {self.hparams.learning_rate}")
            print(f"Total epochs: {self.scheduler_T_max}")
            print(f"Warmup epochs: 3")
            print("=" * 80)
            
            # Print initial learning rates for each parameter group
            for i, param_group in enumerate(self.trainer.optimizers[0].param_groups):
                print(f"Parameter group {i} initial learning rate: {param_group['lr']:.6f}")

    def on_train_end(self) -> None:
        if self.verbose:
            print("\nTraining completed!")
            print(f"Best validation accuracy: {self.best_val_acc:.4f}")
            print(f"Average epoch time: {np.mean(self.epoch_times):.2f}s")
            print("=" * 80)
        
        # Generate plots if enabled
        if self.generate_plots:
            if self.verbose:
                print("\nGenerating training plots...")
            
            epochs = list(range(len(self.train_losses)))
            
            # Prepare metrics for plotting
            metrics = {
                'train_loss': self.train_losses,
                'val_loss': self.val_losses,
                'train_acc': self.train_accs,
                'val_acc': self.val_accs
            }
            
            if self.hparams.track_top5:
                metrics.update({
                    'train_top5_acc': self.train_top5_accs,
                    'val_top5_acc': self.val_top5_accs
                })
            
            # Generate plots
            self.plot_generator.plot_training_curves(metrics, epochs)
            self.plot_generator.plot_learning_rate(
                [self.trainer.optimizers[0].param_groups[0]["lr"] for _ in epochs],
                epochs
            )
            self.plot_generator.plot_epoch_times(self.epoch_times, epochs)
            
            if self.verbose:
                print(f"Plots saved in: {self.plot_generator.plots_dir}")
        
        # Prepare summary
        summary = {
            'config': {
                'model': self.hparams.model_name,
                'activation': self.hparams.activation_name,
                'learning_rate': self.hparams.learning_rate,
                'weight_decay': self.hparams.weight_decay,
                'batch_size': self.trainer.datamodule.hparams.batch_size,
                'epochs': self.hparams.scheduler_T_max
            },
            'metrics': {
                'final_train_loss': self.train_losses[-1],
                'final_val_loss': self.val_losses[-1],
                'final_train_acc': self.train_accs[-1],
                'final_val_acc': self.val_accs[-1],
                'best_val_acc': self.best_val_acc,
                'avg_epoch_time': np.mean(self.epoch_times)
            }
        }
        
        # Add top-5 accuracy if tracked
        if self.hparams.track_top5:
            summary['metrics'].update({
                'final_train_top5_acc': self.train_top5_accs[-1],
                'final_val_top5_acc': self.val_top5_accs[-1]
            })
        
        # Log summary to file
        self.metrics_logger.log_summary(summary)
        
        # Clear memory
        gc.collect()

    def configure_optimizers(self):
        # Use different learning rates for pretrained and new layers
        if self.hparams.pretrained:
            # Lower learning rate for pretrained layers
            pretrained_params = []
            new_params = []
            for name, param in self.named_parameters():
                if 'fc' in name or 'classifier' in name:  # Final layer
                    new_params.append(param)
                else:
                    pretrained_params.append(param)
            
            optimizer = SGD([
                {'params': pretrained_params, 'lr': self.hparams.learning_rate * 0.1},  # 10x smaller for pretrained
                {'params': new_params, 'lr': self.hparams.learning_rate}  # Full lr for new layers
            ],
                momentum=self.hparams.momentum,
                weight_decay=self.hparams.weight_decay,
                nesterov=self.hparams.nesterov
            )
        else:
            optimizer = SGD(
                self.parameters(),
                lr=self.hparams.learning_rate,
                momentum=self.hparams.momentum,
                weight_decay=self.hparams.weight_decay,
                nesterov=self.hparams.nesterov
            )
        
        # Store T_max in instance variable for easy access
        self.scheduler_T_max = 10  # Explicitly set to 10 epochs
        
        # Configure warmup scheduler
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=3,
            warmup_start_lr=self.hparams.learning_rate * 0.1,  # 10% of base lr
            T_max=self.scheduler_T_max,
            eta_min=self.hparams.scheduler_eta_min
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }

    def on_validation_epoch_end(self) -> None:
        # Print validation epoch summary
        if self.verbose:
            print("\nValidation Epoch Summary:")
            print(f"Validation Loss: {self.val_losses[-1]:.4f}")
            print(f"Validation Accuracy: {self.val_accs[-1]:.4f}")
            if self.hparams.track_top5:
                print(f"Validation Top-5 Accuracy: {self.val_top5_accs[-1]:.4f}")
            print("-" * 50) 