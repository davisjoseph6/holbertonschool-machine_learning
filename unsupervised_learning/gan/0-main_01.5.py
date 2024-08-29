#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from simple_gan import Simple_GAN
from gan_utils import fully_connected_GenDiscr, spheric_generator

# Create the generator and discriminator
generator, discriminator, latent_generator = fully_connected_GenDiscr([1, 100, 100, 2], None)

# Print summaries to inspect the model architectures
print(generator.summary())
print(discriminator.summary())

# Define a function to generate real examples (e.g., points in a 2D circle)
def circle_example(num_points, radius=0.5):
    angles = np.linspace(0, 2 * np.pi, num_points)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return np.stack([x, y], axis=1).astype(np.float32)

# Generate real examples
real_examples = circle_example(1000)  # 1000 points in a 2D circle

# Example setup and training code
generator, discriminator, latent_generator = fully_connected_GenDiscr([1, 100, 100, 2], None)
G = Simple_GAN(generator, discriminator, latent_generator, real_examples, batch_size=200, disc_iter=2, learning_rate=.005)

# Compile the GAN model (this step is necessary before training)
G.compile()

# Train the model
history = G.fit(real_examples, epochs=16, steps_per_epoch=100, verbose=1)

# Visualization
def visualize_results(G, real_examples, epoch=None):
    plt.figure(figsize=(12, 6))
    
    # Plot real examples
    plt.subplot(1, 2, 1)
    plt.title("Real Data")
    plt.scatter(real_examples[:, 0], real_examples[:, 1], color='blue', label='Real data')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    
    # Generate and plot fake examples
    fake_examples = G.get_fake_sample(training=False)
    plt.subplot(1, 2, 2)
    plt.title(f"Generated Data at Epoch {epoch+1}" if epoch is not None else "Generated Data")
    plt.scatter(fake_examples[:, 0], fake_examples[:, 1], color='orange', label='Generated data')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    
    plt.show()

# Visualize results after training
visualize_results(G, real_examples, epoch=15)

# Plot the loss over epochs
plt.figure(figsize=(8, 6))
plt.plot(history.history['discr_loss'], label='Discriminator Loss')
plt.plot(history.history['gen_loss'], label='Generator Loss')
plt.title('GAN Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

