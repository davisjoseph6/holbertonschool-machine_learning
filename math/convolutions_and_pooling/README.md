# Convolutions and Pooling in Machine Learning

This project focuses on implementing convolutional and pooling operations from scratch using NumPy. These operations are fundamental building blocks in convolutional neural networks (CNNs), commonly used for image processing tasks.

---

## Directory Overview

### Convolutions
1. **`0-convolve_grayscale_valid.py`**
   - Performs a valid convolution on grayscale images.

2. **`1-convolve_grayscale_same.py`**
   - Performs a same convolution on grayscale images, maintaining the original dimensions.

3. **`2-convolve_grayscale_padding.py`**
   - Conducts a convolution on grayscale images with custom padding.

4. **`3-convolve_grayscale.py`**
   - Implements convolution with adjustable padding (`same`, `valid`, or custom) and stride.

5. **`4-convolve_channels.py`**
   - Performs a convolution on multi-channel images (e.g., RGB).

6. **`5-convolve.py`**
   - Extends convolution to support multiple kernels for feature extraction.

---

### Pooling
7. **`6-pool.py`**
   - Implements pooling operations (`max` and `average`) for dimensionality reduction and feature selection.

---

## How to Use

### Convolution
- **Basic Convolution**: Use `0-convolve_grayscale_valid.py` for valid convolution.
- **Same Convolution**: Utilize `1-convolve_grayscale_same.py` for maintaining the original image dimensions.
- **Custom Padding**: Apply `2-convolve_grayscale_padding.py` for convolution with user-defined padding.
- **Multi-channel Input**: Leverage `4-convolve_channels.py` for images with multiple color channels.
- **Multiple Kernels**: Use `5-convolve.py` to apply multiple filters simultaneously.

### Pooling
- Apply `6-pool.py` for max or average pooling with custom kernel size and stride.

---

## Features

- **Custom Padding and Stride**: Control the behavior of convolution operations using adjustable padding and stride.
- **Multi-channel and Multi-kernel Support**: Perform convolutions on RGB images with multiple filters.
- **Pooling**: Reduce dimensionality using max or average pooling.

---

## Applications
- Image feature extraction
- Dimensionality reduction
- Preparing data for convolutional neural networks (CNNs)

---

## Requirements
- Python 3.x
- NumPy

---

## Author
- Davis Joseph ([LinkedIn](https://www.linkedin.com/in/davis-joseph/))

