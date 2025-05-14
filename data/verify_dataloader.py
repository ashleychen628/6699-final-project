import torch
from data.datamodule import ImageNet100DataModule
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

def denormalize(tensor):
    """Denormalize image tensor for visualization"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return tensor * std + mean

def verify_dataloader():
    # Initialize datamodule
    datamodule = ImageNet100DataModule(
        data_dir="data/imagenet100",  # Adjust path as needed
        batch_size=32,
        num_workers=4,
        num_classes=10,  # Using 10 classes
        train_reduction_factor=10,
        val_reduction_factor=10
    )
    
    # Setup datamodule
    datamodule.setup()
    
    # Get train and val dataloaders
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    
    print("\n=== Dataset Statistics ===")
    print(f"Training set size: {len(datamodule.train_dataset)}")
    print(f"Validation set size: {len(datamodule.val_dataset)}")
    
    # Verify a few batches from training set
    print("\n=== Verifying Training Data ===")
    for i, (images, labels) in enumerate(train_loader):
        print(f"\nBatch {i}:")
        print(f"Image shape: {images.shape}")
        print(f"Label shape: {labels.shape}")
        print(f"Label range: [{labels.min()}, {labels.max()}]")
        print(f"Unique labels in batch: {torch.unique(labels).shape[0]}")
        print(f"Image value range: [{images.min():.3f}, {images.max():.3f}]")
        
        # Visualize first batch
        if i == 0:
            # Denormalize images for visualization
            denorm_images = denormalize(images)
            
            # Create grid of images
            grid = make_grid(denorm_images, nrow=8, normalize=True)
            
            # Convert to numpy and transpose for matplotlib
            grid_np = grid.cpu().numpy().transpose(1, 2, 0)
            
            # Plot
            plt.figure(figsize=(15, 15))
            plt.imshow(grid_np)
            plt.title(f"First batch of training images\nLabels: {labels[:32].tolist()}")
            plt.axis('off')
            plt.savefig('dataloader_verification.png')
            plt.close()
            
            print("\nSaved visualization to 'dataloader_verification.png'")
        
        if i >= 2:  # Check first 3 batches
            break
    
    # Verify a few batches from validation set
    print("\n=== Verifying Validation Data ===")
    for i, (images, labels) in enumerate(val_loader):
        print(f"\nBatch {i}:")
        print(f"Image shape: {images.shape}")
        print(f"Label shape: {labels.shape}")
        print(f"Label range: [{labels.min()}, {labels.max()}]")
        print(f"Unique labels in batch: {torch.unique(labels).shape[0]}")
        print(f"Image value range: [{images.min():.3f}, {images.max():.3f}]")
        
        if i >= 2:  # Check first 3 batches
            break

if __name__ == "__main__":
    verify_dataloader() 