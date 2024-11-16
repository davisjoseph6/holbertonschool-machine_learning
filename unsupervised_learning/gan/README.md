# Generative Adversarial Networks (GANs)

This directory contains various implementations of Generative Adversarial Networks (GANs) and related modules, with a focus on basic GANs, Wasserstein GANs (WGAN), and their variants. GANs are powerful tools for generating synthetic data by training two neural networks (generator and discriminator) in a competitive process.

---

## Directory Overview

### Files and Functions

1. **`0-simple_gan.py`**  
   - **`Simple_GAN` Class**: Implements a basic GAN architecture where a generator creates fake samples and a discriminator tries to differentiate between real and fake samples.
   - **Functions**:
     - `__init__(self, generator, discriminator, latent_generator, real_examples, batch_size=200, disc_iter=2, learning_rate=.005)`: Initializes the GAN model.
     - `get_real_sample(self, size=None)`: Generates a batch of real samples from the dataset.
     - `get_fake_sample(self, size=None, training=False)`: Generates a batch of fake samples from the generator.
     - `train_step(self, useless_argument)`: Performs one training step, iterating through discriminator updates and then updating the generator.

   - **Key Losses**:
     - **Generator Loss**: Mean Squared Error (MSE) with the target of ones.
     - **Discriminator Loss**: MSE with real samples labeled as ones and fake samples as negative ones.

2. **`1-wgan_clip.py`**  
   - **`WGAN_clip` Class**: Implements a Wasserstein GAN with weight clipping for the discriminator to enforce the 1-Lipschitz constraint, an important property for WGANs.
   - **Functions**:
     - `__init__(self, generator, discriminator, latent_generator, real_examples, batch_size=200, disc_iter=2, learning_rate=.005)`: Initializes the WGAN model.
     - The architecture and loss functions are designed to improve training stability and mitigate mode collapse.

---

## Key Concepts

### GAN Architecture
A Generative Adversarial Network consists of two neural networks:
- **Generator (G)**: Takes random noise as input and generates fake data.
- **Discriminator (D)**: Takes both real data and generated (fake) data as input and tries to distinguish between the two.

The two networks are trained in a zero-sum game, where the generator tries to fool the discriminator, and the discriminator tries to correctly classify real and fake data.

### Simple GAN
- A basic GAN model where the generator and discriminator are trained using standard MSE loss. The generator aims to produce samples that are indistinguishable from real samples, while the discriminator distinguishes real from fake samples.
- The training loop involves updating the discriminator multiple times for each generator update.

### Wasserstein GAN (WGAN)
- A type of GAN that uses the Wasserstein loss (Earth Mover’s Distance) instead of the traditional binary cross-entropy loss. This leads to more stable training.
- **Weight Clipping**: In WGANs, the discriminator (or critic) is trained with weight clipping to enforce a 1-Lipschitz constraint. This prevents the discriminator from becoming too powerful, ensuring stable training.

### WGAN with Weight Clipping (WGAN-clip)
- This version of WGAN uses weight clipping for the discriminator, which is essential for maintaining the 1-Lipschitz constraint necessary for WGANs.
- The loss functions are adjusted accordingly, and the discriminator is trained with weight clipping to avoid large gradient values.

---

## How to Use

1. **Simple GAN**:
   - Define your `generator` and `discriminator` models.
   - Create a `Simple_GAN` instance with these models, along with the real training data and a latent space generator.
   - Call `train_step()` to perform one training step, where both the discriminator and generator are updated.

2. **WGAN-clip**:
   - Define your `generator` and `discriminator` models.
   - Use the `WGAN_clip` class, which implements weight clipping for the discriminator and Wasserstein loss for training.
   - Follow a similar procedure as in Simple GAN for training.

---

## Applications
- **Image Generation**: GANs can generate realistic images from random noise.
- **Data Augmentation**: Generating synthetic data for training other machine learning models.
- **Art Generation**: GANs can be used to create artwork or enhance creative processes.
- **Anomaly Detection**: By learning the distribution of the training data, GANs can help detect outliers or anomalies.
- **Style Transfer and Image-to-Image Translation**: GANs are also useful in tasks where the model learns to transform one type of image to another (e.g., turning sketches into realistic images).

---

## Requirements
- Python 3.x
- TensorFlow 2.x
- Keras
- Matplotlib (for visualizations)

---

## References
- **Wasserstein GAN**: "Wasserstein GAN" by Martin Arjovsky, Soumith Chintala, and Léon Bottou (2017).
- **GANs**: "Generative Adversarial Nets" by Ian Goodfellow et al. (2014).
- TensorFlow and Keras Documentation.

---

## Author
- Davis Joseph ([LinkedIn](https://www.linkedin.com/in/davisjoseph767/))

