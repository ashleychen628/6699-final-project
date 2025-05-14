import os
import torch

# Directory where all history files are stored
HISTORY_DIR = "checkpoints"

# Loop through all subdirectories (one per activation)
for activation in os.listdir(HISTORY_DIR):
    act_dir = os.path.join(HISTORY_DIR, activation)
    if not os.path.isdir(act_dir):
        continue

    # Look for files like history_ResNet18_ReLU.pt
    for filename in os.listdir(act_dir):
        if filename.startswith("history_") and filename.endswith(".pt"):
            path = os.path.join(act_dir, filename)
            try:
                history = torch.load(path)
                val_acc = history.get("val_acc", [])
                if val_acc:
                    final_acc = val_acc[-1]
                    print(f"{filename:<40} → Final Val Acc: {final_acc:.2f}%")
                else:
                    print(f"{filename:<40} → No val_acc found.")
            except Exception as e:
                print(f"{filename:<40} → Failed to load: {e}")
