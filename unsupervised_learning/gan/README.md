# Generative Adversarial Networks (GANs)

This directory contains implementations of various types of GANs, including the simple GAN and Wasserstein GAN (WGAN) with clipping. These models are used for generating synthetic data, such as images, by training two neural networks: a generator and a discriminator, in an adversarial setting.

---

## Directory Overview

### Files and Functions

1. **`0-simple_gan.py`**
   - **`Simple_GAN` class**:
     - A basic implementation of a Generative Adversarial Network (GAN).
     - **Components**:
       - **Generator**: Takes random noise as input and generates fake data (e.g., images).
       - **Discriminator**: Classifies data as real or fake.
     - **Training**:
       - The `train_step` method alternates between training the discriminator and generator. The discriminator tries to differentiate real from fake data, while the generator tries to fool the discriminator.
       - The generator and discriminator are trained using the Adam optimizer with a learning rate of `0.005` and `beta_1 = 0.5`, `beta_2 = 0.9`.

     - **Key Methods**:
       - `get_real_sample(size=None)`: Generates a batch of real samples from the dataset.
       - `get_fake_sample(size=None, training=False)`: Generates a batch of fake samples from the generator.
       - `train_step(useless_argument)`: Performs one training step for both the discriminator and generator.

2. **`1-wgan_clip.py`**
   - **`WGAN_clip` class**:
     - A variant of the GAN known as Wasserstein GAN (WGAN) with weight clipping.
     - The main difference between WGAN and the standard GAN is the use of a different loss function (Wasserstein loss) and weight clipping for the discriminator.
     - **Key Features**:
       - **Wasserstein loss**: This loss function provides more stable gradients, which can improve training.
       - **Weight Clipping**: The discriminator’s weights are clipped to a fixed range to enforce the Lipschitz constraint required for the Wasserstein loss.

     - **Training**:
       - Similar to the `Simple_GAN`, the `train_step` method alternates between training the discriminator and generator, but with Wasserstein loss and weight clipping.

---

## Key Concepts

### Simple GAN
- **Generator**: Learns to create synthetic data that resembles the real data distribution.
- **Discriminator**: Learns to distinguish between real data and the generated data.
- **Adversarial Process**: The generator and discriminator are trained simultaneously, where the generator aims to deceive the discriminator, and the discriminator tries to correctly classify data as real or fake.

### WGAN with Clipping
- **Wasserstein Loss**: Unlike the traditional GAN loss (binary cross-entropy), WGAN uses a loss that is based on the Earth Mover’s distance, which provides smoother gradients for training.
- **Weight Clipping**: To ensure that the discriminator function is Lipschitz continuous, the weights of the discriminator are clipped within a fixed range after each gradient update.

---

## How to Use

1. **Simple GAN**:
   - Create a `Simple_GAN` model by initializing it with a generator, discriminator, and a latent space generator (e.g., random noise generator).
   - Use the `train_step` method for training the model. This method alternates between training the discriminator and the generator.
   - Adjust hyperparameters such as `batch_size`, `learning_rate`, and `disc_iter` for better results.

2. **WGAN with Clipping**:
   - Create a `WGAN_clip` model by initializing it similarly to the `Simple_GAN`, but with a discriminator that includes weight clipping and Wasserstein loss.
   - Train the model using the `train_step` method, which handles both discriminator and generator training steps.

---

## Applications

- **Image Generation**: GANs are widely used for generating synthetic images, including faces, objects, and scenes.
- **Data Augmentation**: Generating new data to augment training datasets.
- **Super-Resolution**: Enhancing the resolution of images using GANs.
- **Art and Creative Works**: Generating artwork, music, and other creative content.

---

## Requirements

- Python 3.x
- TensorFlow 2.x
- Keras
- Matplotlib (for visualization)
- NumPy

---

## References

- Ian Goodfellow et al., "Generative Adversarial Networks" (2014)
- Martin Arjovsky, Soumith Chintala, and Léon Bottou, "Wasserstein GAN" (2017)

---

## Author
- Davis Joseph ([LinkedIn](https://www.linkedin.com/in/davisjoseph767/))

