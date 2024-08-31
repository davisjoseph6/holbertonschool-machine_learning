#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

# Load the pictures
array_of_pictures = np.load("faces/small_res_faces_10000.npy")
array_of_pictures = array_of_pictures.astype("float32") / 255

# Calculate and visualize the mean face
mean_face = array_of_pictures.mean(axis=0)
plt.imshow(mean_face, cmap='gray')
plt.show()

