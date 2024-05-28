#!/usr/bin/env python3
"""
Performs a same convolution on grayscale images.
"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate padding for height and width
    pad_h = (kh - 1) // 2
    pad_w = (kw - 1) // 2

    # Pad the images with zeros
    padded_images = np.pad(images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

    # Initialize the output array with zeros
    output = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            # Apply the kernel to each position of the padded image
            output[:, i, j] = np.sum(padded_images[:, i:i+kh, j:j+kw] * kernel, axis=(1, 2))

    return output
