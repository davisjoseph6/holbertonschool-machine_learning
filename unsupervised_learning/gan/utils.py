#!/usr/bin/env python3

# utils.py
import matplotlib.pyplot as plt

def plot_100(images, title):
    fig, axes = plt.subplots(10, 10, figsize=(20, 20))
    fig.subplots_adjust(top=0.95)
    fig.suptitle(title)
    for i in range(100):
        axes[i // 10, i % 10].imshow(images[i], cmap='gray')
        axes[i // 10, i % 10].axis("off")
    plt.show()

