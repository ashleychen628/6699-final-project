# import os
# import shutil
# from pathlib import Path
# import argparse
# import kagglehub
# from kagglehub import KaggleDatasetAdapter

# def prepare_imagenet100(target_dir: str = "data/imagenet-100"):
#     """
#     Download and prepare the ImageNet-100 dataset using kagglehub with Hugging Face adapter.
    
#     Args:
#         target_dir: Directory where the dataset will be organized
#     """
#     # Create target directory
#     target_dir = Path(target_dir)
#     target_dir.mkdir(parents=True, exist_ok=True)
    
#     print(f"Downloading ImageNet-100 dataset to {target_dir}...")
    
#     try:
#         # Download the dataset using kagglehub with Hugging Face adapter
#         dataset = kagglehub.load_dataset(
#             KaggleDatasetAdapter.HUGGING_FACE,
#             "ambityga/imagenet100-1",
#             str(target_dir),
#             # Additional Hugging Face dataset arguments
#             hf_kwargs={
#                 "cache_dir": str(target_dir),
#                 "use_auth_token": None,  # No authentication needed for public dataset
#             }
#         )
        
#         print("Dataset download complete!")
#         print(f"\nDataset is now available at: {target_dir}")
#         print("\nDataset structure:")
#         print(dataset)
        
#         print("\nYou can now run the training script with:")
#         print(f"python train.py model.name=resnet18 activation.name=relu data.data_dir={target_dir}")
        
#     except Exception as e:
#         print(f"Error downloading dataset: {e}")
#         print("\nMake sure you have:")
#         print("1. Installed kagglehub with Hugging Face support: pip install 'kagglehub[hf-datasets]'")
#         print("2. Set up your Kaggle credentials (kaggle.json) in ~/.kaggle/")
#         print("3. Have an active internet connection")
#         print("\nIf you still have issues, you can try downloading manually from:")
#         print("https://www.kaggle.com/datasets/ambityga/imagenet100-1")
#         raise

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Download and prepare ImageNet-100 dataset using kagglehub with Hugging Face adapter")
#     parser.add_argument("--target-dir", default="data/imagenet-100",
#                       help="Directory where the dataset will be organized (default: data/imagenet-100)")
    
#     args = parser.parse_args()
#     prepare_imagenet100(args.target_dir) 
import kagglehub

# Download the latest version.
kagglehub.dataset_download('ambityga/imagenet100')