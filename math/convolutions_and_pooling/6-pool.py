#!/usr/bin/env python3
"""
Performs pooling on images.
"""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Calculate output dimensions
    output_h = (h - kh) // sh + 1
    output_w = (w - kw) // sw + 1

    # Initialize pooling output array
    pooled = np.zeros((m, output_h, output_w, c))

    for i in range(output_h):
        for j in range(output_w):
            # Extract region from images, scaling indexes by stride
            region = images[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
            if mode == 'max':
                # Max pooling
                pooled[:, i, j, :] = np.max(region, axis=(1, 2))
            elif mode == 'avg':
                # Average pooling
                pooled[:, i, j, :] = np.mean(region, axis=(1, 2))

    return pooled
