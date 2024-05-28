#!/usr/bin/env python3
"""
Performs a convolution on grayscale images
"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph = (kh - 1) // 2
        pw = (kw - 1) // 2
        output_h = h
        output_w = w
    elif padding == 'valid':
        ph = 0
        pw = 0
        output_h = (h - kh) // sh + 1
        output_w = (w - kw) // sw + 1
    else:
        ph, pw = padding
        output_h = (h + 2 * ph - kh) // sh + 1
        output_w = (w + 2 * pw - kw) // sw + 1

    # Pad the images with zeros
    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant', constant_values=0)

    # Initialize the output array with zeros
    output = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            # Apply the kernel to each position of the padded image
            output[:, i, j] = np.sum(
                    padded_images[:, i*sh:i*sh+kh, j*sw:j*sw+kw] * kernel, axis=(1, 2)
                    )

    return output
