import os
import shutil
from glob import glob
from tqdm import tqdm

original_root = "data/imagenet-100"
output_root = "data/imagenet-100-imagefolder"

train_parts = ["train.X1", "train.X2", "train.X3", "train.X4"]
val_part = "val.X"

def merge_parts(parts, split_name):
    out_dir = os.path.join(output_root, split_name)
    os.makedirs(out_dir, exist_ok=True)

    for part in parts:
        part_dir = os.path.join(original_root, part)
        class_folders = glob(os.path.join(part_dir, "*"))

        for class_dir in tqdm(class_folders, desc=f"Merging {part}"):
            class_name = os.path.basename(class_dir)
            out_class_dir = os.path.join(out_dir, class_name)
            os.makedirs(out_class_dir, exist_ok=True)

            for img_file in glob(os.path.join(class_dir, "*")):
                filename = os.path.basename(img_file)
                dest = os.path.join(out_class_dir, filename)
                if not os.path.exists(dest):
                    shutil.copy(img_file, dest)

# Merge train
merge_parts(train_parts, "train")

# Merge val
merge_parts([val_part], "val")

print("✅ Merged dataset is ready at 'data/imagenet-100-imagefolder'")
