#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import norm, chi2

# Load the pictures
array_of_pictures = np.load("faces/small_res_faces_10000.npy")
array_of_pictures = array_of_pictures.astype("float32") / 255

# Reshape the data for PCA
flat = array_of_pictures.reshape(10000, 256)
pca = PCA(n_components=49)
pca.fit(flat)
X = pca.transform(flat)
X /= np.std(X, axis=0)

# Plot the distribution of principal components
x = np.linspace(-5, 5, 50)
y = norm.pdf(x)

fig, axes = plt.subplots(7, 7, figsize=(21, 11))
for i in range(49):
    axes[i // 7, i % 7].hist(X[:, i], density=True, bins=40)
    axes[i // 7, i % 7].plot(x, y, color="magenta")
plt.show()

# Plot the distribution of sum of squares
plt.hist(np.sum(np.square(X), axis=1), bins=100, density=True)
x = np.linspace(0, 200, 100)
plt.plot(x, chi2.pdf(x, 49))
plt.savefig("not_chi2.png")
plt.show()

# Shuffle the principal components to enforce independence
Y = X.copy()
for i in range(49):
    np.random.shuffle(Y[:, i])

# Plot the shuffled distribution of sum of squares
plt.hist(np.sum(np.square(Y), axis=1), bins=100, density=True)
plt.plot(x, chi2.pdf(x, 49))
plt.savefig("chi2.png")
plt.show()
