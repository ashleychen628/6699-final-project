import os
import json
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
import pytorch_lightning as pl
from typing import Optional, Dict, Any, List, Tuple

class ImageNet100Dataset(Dataset):
    """Dataset class for ImageNet-100 with support for split directories (train.X1-4 and val.X)"""
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        split_number: Optional[int] = None,
        train_reduction_factor: int = 10,  # Keep 1/10 of training images
        val_reduction_factor: int = 10     # Keep 1/10 of validation images
    ):
        """
        Args:
            root_dir: Root directory of the dataset
            split: Either 'train' or 'val'
            transform: Optional transforms to apply
            split_number: For training, which split to use (1-4). For validation, ignored.
                        If None, uses all splits.
            train_reduction_factor: Keep 1/N of training images (default: 10)
            val_reduction_factor: Keep 1/N of validation images (default: 10)
        """
        # Convert to absolute path
        self.root_dir = Path(os.path.abspath(root_dir))
        self.split = split
        self.transform = transform or transforms.ToTensor()
        self.train_reduction_factor = train_reduction_factor
        self.val_reduction_factor = val_reduction_factor
        
        # Load class labels with better error handling
        labels_path = self.root_dir / "Labels.json"
        if not labels_path.exists():
            raise FileNotFoundError(
                f"Labels.json not found at {labels_path}. "
                f"Please ensure the file exists in the dataset directory."
            )
        
        try:
            with open(labels_path, 'r') as f:
                self.class_labels = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing Labels.json: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading Labels.json: {e}")
        
        # Create class to index mapping
        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(self.class_labels.keys()))}
        
        # Get image paths and labels
        self.images = []
        self.labels = []
        
        if split == "train":
            if split_number is not None:
                # Use specific training split (train.X1 through train.X4)
                split_dir = self.root_dir / f"train.X{split_number}"
                if not split_dir.exists():
                    raise FileNotFoundError(f"Training split directory not found: {split_dir}")
                self._load_split(split_dir)
            else:
                # Use all training splits
                for i in range(1, 5):  # train.X1 through train.X4
                    split_dir = self.root_dir / f"train.X{i}"
                    if not split_dir.exists():
                        raise FileNotFoundError(f"Training split directory not found: {split_dir}")
                    self._load_split(split_dir)
        else:  # validation
            # For validation, use val.X directory
            split_dir = self.root_dir / "val.X"
            if not split_dir.exists():
                raise FileNotFoundError(f"Validation directory not found: {split_dir}")
            self._load_split(split_dir)
        
        if len(self.images) == 0:
            raise RuntimeError(f"No images found in {split_dir}")
        
        print(f"Loaded {len(self.images)} images from {split_dir}")
    
    def _load_split(self, split_dir: Path):
        """Load images and labels from a split directory"""
        if not split_dir.exists():
            return
            
        for class_name in self.class_to_idx.keys():
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue
                
            # Get all image files
            image_files = [f for f in os.listdir(class_dir) 
                         if f.endswith(('.JPEG', '.jpeg', '.jpg', '.png'))]
            
            # Sort files for deterministic selection
            image_files.sort()
            
            # Apply reduction factor with fixed seed for reproducibility
            if self.split == "train":
                # For training, take first N images after sorting
                num_images = len(image_files) // self.train_reduction_factor
                image_files = image_files[:num_images]
            else:
                # For validation, take first N images after sorting
                num_images = len(image_files) // self.val_reduction_factor
                image_files = image_files[:num_images]
            
            # Add selected images and their labels
            for img_name in image_files:
                self.images.append(str(class_dir / img_name))
                self.labels.append(self.class_to_idx[class_name])
            
            print(f"Loaded {len(image_files)} images for class {class_name} in {self.split} split")
            print(f"First few images: {image_files[:3]}")  # Print first few images for verification
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load and transform image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image, label

class ImageNet100DataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for ImageNet-100"""
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 224,
        augmentations: Optional[Dict[str, Any]] = None,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        pin_memory: bool = True,
        train_reduction_factor: int = 10,  # Keep 1/10 of training images
        val_reduction_factor: int = 10     # Keep 1/10 of validation images
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.train_reduction_factor = train_reduction_factor
        self.val_reduction_factor = val_reduction_factor
        
        # Default augmentations if none provided
        self.augmentations = augmentations or {
            "resize_size": 256,
            "crop_size": 224,
            "hflip_prob": 0.5,
            "color_jitter": {
                "brightness": 0.4,
                "contrast": 0.4,
                "saturation": 0.4
            },
            "normalize": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            }
        }
        
        # Setup transforms
        self.train_transform = transforms.Compose([
            transforms.Resize(self.augmentations["resize_size"]),
            transforms.RandomResizedCrop(self.augmentations["crop_size"]),
            transforms.RandomHorizontalFlip(p=self.augmentations["hflip_prob"]),
            transforms.ColorJitter(
                brightness=self.augmentations["color_jitter"]["brightness"],
                contrast=self.augmentations["color_jitter"]["contrast"],
                saturation=self.augmentations["color_jitter"]["saturation"]
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.augmentations["normalize"]["mean"],
                std=self.augmentations["normalize"]["std"]
            )
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize(self.augmentations["resize_size"]),
            transforms.CenterCrop(self.augmentations["crop_size"]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.augmentations["normalize"]["mean"],
                std=self.augmentations["normalize"]["std"]
            )
        ])
    
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            # Create training datasets for each split
            self.train_datasets = [
                ImageNet100Dataset(
                    self.data_dir,
                    split="train",
                    transform=self.train_transform,
                    split_number=i,
                    train_reduction_factor=self.train_reduction_factor,
                    val_reduction_factor=self.val_reduction_factor
                ) for i in range(1, 5)  # train.X1 through train.X4
            ]
            # Combine all training splits
            self.train_dataset = ConcatDataset(self.train_datasets)
            
            # Create validation dataset
            self.val_dataset = ImageNet100Dataset(
                self.data_dir,
                split="val",
                transform=self.val_transform,
                train_reduction_factor=self.train_reduction_factor,
                val_reduction_factor=self.val_reduction_factor
            )
            
            print(f"Training set size: {len(self.train_dataset)}")
            print(f"Validation set size: {len(self.val_dataset)}")
            print(f"Using reduction factors - Training: 1/{self.train_reduction_factor}, Validation: 1/{self.val_reduction_factor}")
        
        if stage == "test" or stage is None:
            # For testing, use the same validation dataset
            self.test_dataset = ImageNet100Dataset(
                self.data_dir,
                split="val",  # This will automatically load all val.X splits
                transform=self.val_transform
            )
            print(f"Test set size: {len(self.test_dataset)}")
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            drop_last=True,
            multiprocessing_context='fork'
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            multiprocessing_context='fork'
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            multiprocessing_context='fork'
        ) 