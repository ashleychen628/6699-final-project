import os
import torch
import matplotlib.pyplot as plt

BASE_DIR = "checkpoints"  # or wherever your .pt files are saved
SELECT_MODEL = "ResNet18"  # or "ResNet18", "MobileNetV2", "CustomCNN" etc.

# List all activations you used in training
ACTIVATIONS = ['ReLU', 'LeakyReLU', 'Sigmoid', 'Tanh', 'ELU', 'Swish_fixed', 'Swish_trainable']

plt.figure(figsize=(10,6))

for activation in ACTIVATIONS:
    hist_path = os.path.join(BASE_DIR, activation, f"history_{SELECT_MODEL}_{activation}.pt")
    if not os.path.exists(hist_path):
        print(f"Missing: {hist_path}")
        continue

    history = torch.load(hist_path)
    acc_list = history.get('val_top5', history.get('val_top5', []))
    
    if not acc_list:
        print(f"No val/test acc found in {activation}")
        continue

    epochs = range(1, len( acc_list) + 1)
    plt.plot(epochs,  acc_list, marker='o', label=activation)

plt.title(f"Validation Accuracy (Top%) over Epochs ({SELECT_MODEL})")
plt.xlabel("Epoch")
plt.ylabel("Top 5 Accuracy (%)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"{SELECT_MODEL}_val_acc_5_plot.png", dpi=300)

plt.show()
# plt.savefig(f"{SELECT_MODEL}_val_acc_plot.png", dpi=300)
