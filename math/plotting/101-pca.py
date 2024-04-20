#!/usr/bin/env python3
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# Load the data and labels from the.npy files
data = np.load('data.npy')
labels = np.load('labels.npy')

# Save both into a single .npz file
np.savez('pca.npz', data=data, labels=labels)

# Load the data and the labels
lib = np.load("pca.npz")
data = lib["data"]
labels = lib["labels"]

# Standard PCA processing
data_means = np.mean(data, axis=0)
norm_data = data - data_means
_, _, Vh = np.linalg.svd(norm_data)
pca_data = np.matmul(norm_data, Vh[:3].T)

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot using labels for color coding
scatter = ax.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2], 
        c=labels, cmap='plasma')

# Labeling the axes
ax.set_xlabel('U1')
ax.set_ylabel('U2')
ax.set_zlabel('U3')

# Title
ax.set_title('PCA of Iris Dataset')

# Show plot
plt.show()
