import os
import random
import shutil
from tqdm import tqdm

def shrink_imagefolder_consistent(
    full_train_dir, full_val_dir,
    mini_train_dir, mini_val_dir,
    num_classes, train_imgs_per_class, val_imgs_per_class
):
    # Step 1: Get shared subset of class names
    all_classes = sorted(os.listdir(full_train_dir))
    selected_classes = random.sample(all_classes, num_classes)

    os.makedirs(mini_train_dir, exist_ok=True)
    os.makedirs(mini_val_dir, exist_ok=True)

    for cls in tqdm(selected_classes, desc="Processing selected classes"):
        # Process train
        src_train = os.path.join(full_train_dir, cls)
        dst_train = os.path.join(mini_train_dir, cls)
        os.makedirs(dst_train, exist_ok=True)

        train_images = sorted(os.listdir(src_train))
        selected_train_images = random.sample(train_images, min(train_imgs_per_class, len(train_images)))
        for img in selected_train_images:
            shutil.copy(os.path.join(src_train, img), os.path.join(dst_train, img))

        # Process val
        src_val = os.path.join(full_val_dir, cls)
        dst_val = os.path.join(mini_val_dir, cls)
        os.makedirs(dst_val, exist_ok=True)

        val_images = sorted(os.listdir(src_val))
        selected_val_images = random.sample(val_images, min(val_imgs_per_class, len(val_images)))
        for img in selected_val_images:
            shutil.copy(os.path.join(src_val, img), os.path.join(dst_val, img))

    print(f"✅ Created subset with {num_classes} classes in:")
    print(f"   → Train: {mini_train_dir}")
    print(f"   → Val:   {mini_val_dir}")

# === CONFIG ===
full_train_dir = "data/imagenet-100-imagefolder/train"
full_val_dir   = "data/imagenet-100-imagefolder/val"

mini_train_dir = "data/imagenet-1300-10/train"
mini_val_dir   = "data/imagenet-1300-10/val"

shrink_imagefolder_consistent(
    full_train_dir, full_val_dir,
    mini_train_dir, mini_val_dir,
    num_classes=10, train_imgs_per_class=1300, val_imgs_per_class=50
)
