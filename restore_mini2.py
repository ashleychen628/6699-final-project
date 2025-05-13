import os
import shutil

# Your 20-class mapping
class_to_idx = {
    'n01514859': 0, 'n01537544': 1, 'n01560419': 2, 'n01592084': 3,
    'n01608432': 4, 'n01614925': 5, 'n01632458': 6, 'n01685808': 7,
    'n01687978': 8, 'n01693334': 9, 'n01770393': 10, 'n01774384': 11,
    'n01795545': 12, 'n01828970': 13, 'n01843383': 14, 'n01910747': 15,
    'n01944390': 16, 'n01986214': 17, 'n02006656': 18, 'n02077923': 19
}

# Directories
full_train_dir = "data/imagenet-100-imagefolder/train"
full_val_dir   = "data/imagenet-100-imagefolder/val"
mini_train_dir = "data/imagenet-mini2/train"
mini_val_dir   = "data/imagenet-mini2/val"

# Create mini dirs
os.makedirs(mini_train_dir, exist_ok=True)
os.makedirs(mini_val_dir, exist_ok=True)

# Helper to copy selected folders
def copy_folders(class_list, src_root, dst_root):
    for cls in class_list:
        src = os.path.join(src_root, cls)
        dst = os.path.join(dst_root, cls)
        if os.path.exists(src):
            shutil.copytree(src, dst)
            print(f"✅ Copied {cls} from {src_root} to {dst_root}")
        else:
            print(f"❌ WARNING: {src} not found — skipping.")

# Copy matching folders
copy_folders(class_to_idx.keys(), full_train_dir, mini_train_dir)
copy_folders(class_to_idx.keys(), full_val_dir, mini_val_dir)
