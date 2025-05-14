import os
import torch
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = "checkpoints"
SELECT_MODEL = "MobileNetV2" # "ResNet18"

ACTIVATIONS = ['ReLU', 'LeakyReLU', 'Sigmoid', 'Tanh', 'ELU', 'Swish_fixed', 'Swish_trainable']

epoch_time_dict = {}

# Load and extract epoch times
for activation in ACTIVATIONS:
    hist_path = os.path.join(BASE_DIR, activation, f"history_{SELECT_MODEL}_{activation}.pt")
    if not os.path.exists(hist_path):
        print(f"Missing: {hist_path}")
        continue

    history = torch.load(hist_path)
    epoch_times = history.get('epoch_time', [])
    if not epoch_times:
        print(f"⚠️ No epoch_time found in {activation}")
        continue

    epoch_time_dict[activation] = epoch_times

# Compute average time per activation
avg_epoch_times = {act: np.mean(t) for act, t in epoch_time_dict.items()}

# Plot bar chart
plt.figure(figsize=(9, 5))
activations = list(avg_epoch_times.keys())
avg_values = list(avg_epoch_times.values())

bars = plt.bar(activations, avg_values, color='salmon')

# Add average value on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, f'{height:.2f}',
             ha='center', va='bottom', fontsize=9)

plt.ylabel("Average Epoch Time (seconds)")
plt.title(f"Average Epoch Time per Activation ({SELECT_MODEL})")
plt.xticks(rotation=30)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(f"{SELECT_MODEL}_avg_epoch_time_bar.png", dpi=300)
plt.show()
