#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

# Load the pictures
array_of_pictures = np.load("faces/small_res_faces_10000.npy")
array_of_pictures = array_of_pictures.astype("float32") / 255

# Visualize the first 100 real faces
fig, axes = plt.subplots(10, 10, figsize=(10, 10))
fig.suptitle("Real Faces")
for i in range(100):
    axes[i // 10, i % 10].imshow(array_of_pictures[i, :, :], cmap='gray')
    axes[i // 10, i % 10].axis("off")
plt.show()
