# Autoencoders

This project focuses on various types of autoencoders built using TensorFlow, including vanilla, sparse, and convolutional autoencoders. Autoencoders are unsupervised neural networks used for dimensionality reduction, feature extraction, and data denoising.

---

## Directory Overview

### Files and Functions

1. **`0-vanilla.py`**
   - **`autoencoder(input_dims, hidden_layers, latent_dims)`**
     - Implements a vanilla (basic) autoencoder with fully connected layers.
     - **Inputs**:
       - `input_dims`: The number of features in the input data.
       - `hidden_layers`: A list of integers representing the number of nodes in each hidden layer.
       - `latent_dims`: The size of the latent space (bottleneck).
     - **Outputs**:
       - Returns the encoder, decoder, and full autoencoder models.
     - **Key Layers**:
       - Dense layers with ReLU activation for encoding and decoding.
       - Sigmoid activation in the output layer for reconstruction.

2. **`1-sparse.py`**
   - **`autoencoder(input_dims, hidden_layers, latent_dims, lambtha)`**
     - Implements a sparse autoencoder with L1 regularization for sparsity control.
     - **Inputs**:
       - `input_dims`: The number of features in the input data.
       - `hidden_layers`: A list of integers representing the number of nodes in each hidden layer.
       - `latent_dims`: The size of the latent space (bottleneck).
       - `lambtha`: Regularization factor for L1 regularization.
     - **Outputs**:
       - Returns the encoder, decoder, and full autoencoder models.
     - **Key Layers**:
       - Sparse regularization on the latent space to encourage sparsity.

3. **`2-convolutional.py`**
   - **`build_encoder(input_dims, filters)`**
     - Builds the encoder part of the convolutional autoencoder using convolutional layers.
     - **Inputs**:
       - `input_dims`: The dimensions of the input (e.g., height, width, channels).
       - `filters`: A list of filter sizes for each convolutional layer.
     - **Outputs**:
       - Returns the encoder model.
   
   - **`build_decoder(latent_dims, filters, input_dims)`**
     - Builds the decoder part of the convolutional autoencoder using deconvolution (upsampling) layers.
     - **Inputs**:
       - `latent_dims`: The shape of the latent space (e.g., the bottleneck).
       - `filters`: A list of filter sizes for the decoder layers.
       - `input_dims`: The dimensions of the original input data.
     - **Outputs**:
       - Returns the decoder model.
   
   - **`autoencoder(input_dims, filters, latent_dims)`**
     - Combines the encoder and decoder to create a full convolutional autoencoder.
     - **Inputs**:
       - `input_dims`: The dimensions of the input (e.g., height, width, channels).
       - `filters`: A list of filter sizes for the convolutional layers.
       - `latent_dims`: The shape of the latent space.
     - **Outputs**:
       - Returns the encoder, decoder, and full autoencoder models.

---

## Key Concepts

### Vanilla Autoencoder
- A basic autoencoder with fully connected layers.
- The encoder compresses the input into a lower-dimensional latent space.
- The decoder reconstructs the input from the latent space.

### Sparse Autoencoder
- Similar to the vanilla autoencoder but with L1 regularization applied to the latent space.
- Encourages sparsity in the hidden representations by adding a penalty term to the loss function.

### Convolutional Autoencoder
- Uses convolutional layers for the encoder and decoder, making it well-suited for image data.
- The encoder uses convolutional and pooling layers to compress the input.
- The decoder uses deconvolutional (upsampling) layers to reconstruct the input.

---

## How to Use

1. **Vanilla Autoencoder**:
   - Use the `autoencoder(input_dims, hidden_layers, latent_dims)` function to create a basic autoencoder.
   - Train the model with your data and use the encoder and decoder for encoding and decoding tasks.

2. **Sparse Autoencoder**:
   - Use the `autoencoder(input_dims, hidden_layers, latent_dims, lambtha)` function to create a sparse autoencoder with L1 regularization.
   - Adjust the `lambtha` parameter to control the sparsity level.

3. **Convolutional Autoencoder**:
   - Use the `autoencoder(input_dims, filters, latent_dims)` function to create a convolutional autoencoder.
   - Pass image data with appropriate input dimensions (height, width, channels) for the model.

---

## Applications
- **Dimensionality Reduction**: Reducing the number of features in data while preserving important information.
- **Image Denoising**: Removing noise from images while maintaining important features.
- **Anomaly Detection**: Identifying outliers in data based on reconstruction error.
- **Generative Models**: Generating new data similar to the training data.

---

## Requirements
- Python 3.x
- TensorFlow 2.x
- Keras

---

## References
- "Autoencoders: A Comprehensive Review" - A deep dive into autoencoder architectures and applications.
- TensorFlow and Keras Documentation.

---

## Author
- Davis Joseph ([LinkedIn](https://www.linkedin.com/in/davisjoseph767/))

