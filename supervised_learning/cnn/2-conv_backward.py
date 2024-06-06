#!/usr/bin/env python3
"""
Convolution Back Propagation
"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer of a neural network

    Parameters:
    - dZ (numpy.ndarray): shape (m, h_new, w_new, c_new) partial derivatives
    with respect to the unactivated output of the convolutional layer
    - A_prev (numpy.ndarray): shape (m, h_prev, w_prev, c_prev) output of
    the previous layer
    - W (numpy.ndarray): shape (kh, kw, c_prev, c_new) kernels for the convolution
    - b (numpy.ndarray): shape (1, 1, 1, c_new) biases applied to the convolution
    - padding (str): 'same' or 'valid', indicating the type of padding used
    - stride (tuple): (sh, sw) containing the strides for the convolution

    Returns:
    - dA_prev (numpy.ndarray): partial derivatives with respect to the
    previous layer
    - dW (numpy.ndarray): partial derivatives with respect to the kernels
    - db (numpy.ndarray): partial derivatives with respect to the biases
    """

    # Extract dimensions from the input shapes
    _, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    # Determine padding size
    if padding == 'valid':
        # No padding
        ph, pw = 0, 0
    elif padding == 'same':
        # Calculate padding to keep the output size same as input
        ph = int((((h_prev - 1) * sh + kh - h_prev) / 2 + 0.5))
        pw = int((((w_prev - 1) * sw + kw - w_prev) / 2 + 0.5))

    # Apply padding to A_prev
    A_prev_pad = np.pad(A_prev,
            [(0, 0), (ph, ph), (pw, pw), (0, 0)],
            mode='constant')

    # Calculate the bias gradient
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    # Initialize gradients for dA_pad and dW
    dA_pad = np.zeros(shape=A_prev_pad.shape)
    dW = np.zeros(shape=W.shape)

    # Loop over every example in the batch
    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                # Loop over every filter
                for f in range(c_new):
                    # Define the slice boundaries
                    v_start = h * sh
                    v_end = v_start + kh
                    h_start = w * sw
                    h_end = h_start + kw

                    # Update the gradient for the padded input
                    dA_pad[i, v_start:v_end, h_start:h_end, :] += W[:, :, :, f] * dZ[i, h, w, f]
                    # Update the gradient for the filter weights
                    dW[:, :, :, f] += A_prev_pad[i, v_start:v_end, h_start:h_end, :] * dZ[i, h, w, f]

    # Remove padding from the gradient if necessary
    if padding == "same":
        dA = dA_pad[:, ph:-ph, pw:-pw, :]
    else:
        dA = dA_pad

    return dA, dW, db
