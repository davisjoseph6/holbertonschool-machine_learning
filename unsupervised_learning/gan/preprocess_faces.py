#!/usr/bin/env python3

import numpy as np

# Load the pictures
array_of_pictures = np.load("faces/small_res_faces_10000.npy")
array_of_pictures = array_of_pictures.astype("float32") / 255

# Calculate the mean face and center the data
mean_face = array_of_pictures.mean(axis=0)
centered_array = array_of_pictures - mean_face

# Normalize the data
multiplier = np.max(np.abs(array_of_pictures), axis=0)
normalized_array = centered_array / multiplier

# Function to recover the original images
def recover(normalized):
    return normalized * multiplier + mean_face

