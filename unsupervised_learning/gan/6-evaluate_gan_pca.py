#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import chi2
import tensorflow as tf

# Import WGAN_GP class from the 4-wgan_gp.py file
WGAN_GP = __import__('4-wgan_gp').WGAN_GP

# Import convolutional_GenDiscr from 3-generate_faces.py
convolutional_GenDiscr = __import__('3-generate_faces').convolutional_GenDiscr

# Load the pictures from small_res_faces_10000.npy
array_of_pictures = np.load("small_res_faces_10000.npy")
array_of_pictures = array_of_pictures.astype("float32") / 255

# Center and Normalize the data
mean_face = array_of_pictures.mean(axis=0)
centered_array = array_of_pictures - mean_face
multiplier = np.max(np.abs(array_of_pictures), axis=0)
normalized_array = centered_array / multiplier

real_ex = tf.convert_to_tensor(normalized_array, dtype="float32")

# Load the pretrained WGAN_GP model and replace weights
generator, discriminator = convolutional_GenDiscr()
G = WGAN_GP(generator, discriminator, latent_generator=lambda k: np.random.randn(k, 16), real_examples=real_ex)
G.replace_weights("generator.h5", "discriminator.h5")

# Generate fake samples using the generator
H = G.get_fake_sample(10000).numpy()[:, :, :, 0]
flat = H.reshape(10000, 256)

# Perform PCA on the fake samples
pca = PCA(n_components=49)
pca.fit(flat)
X = pca.transform(flat)
X = X / np.std(X, axis=0)

# Plot the distribution of sum of squares for the fake samples
plt.hist(np.sum(np.square(X), axis=1), bins=100, density=True)
x = np.linspace(0, 200, 100)
plt.plot(x, chi2.pdf(x, 49))
plt.savefig("not_chi2.png")
plt.show()

