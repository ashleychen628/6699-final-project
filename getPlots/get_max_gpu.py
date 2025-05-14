import os
import torch
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = "checkpoints"
SELECT_MODEL = "MobileNetV2" # "ResNet18", "MobileNetV2"

ACTIVATIONS = ['ReLU', 'LeakyReLU', 'Sigmoid', 'Tanh', 'ELU', 'Swish_fixed', 'Swish_trainable']

memory_dict = {}

# Load GPU memory history for each activation
for activation in ACTIVATIONS:
    hist_path = os.path.join(BASE_DIR, activation, f"history_{SELECT_MODEL}_{activation}.pt")
    if not os.path.exists(hist_path):
        print(f"Missing: {hist_path}")
        continue

    history = torch.load(hist_path)
    max_gpu_list = history.get('max_gpu_mem', [])
    if not max_gpu_list:
        print(f"⚠️ No max_gpu_mem found in {activation}")
        continue

    memory_dict[activation] = max_gpu_list

# Compute average memory usage per activation
avg_memory = {act: np.mean(mem) for act, mem in memory_dict.items()}

# Plot bar chart
plt.figure(figsize=(9, 5))
activations = list(avg_memory.keys())
avg_values = list(avg_memory.values())

plt.bar(activations, avg_values, color='skyblue')
plt.ylabel("Average Max GPU Memory (MB)")
plt.title(f"Average Max GPU Memory per Activation ({SELECT_MODEL})")
plt.xticks(rotation=30)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(f"{SELECT_MODEL}_avg_gpu_mem_bar.png", dpi=300)
plt.show()
