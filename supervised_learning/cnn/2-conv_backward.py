#!/usr/bin/env python3
"""
Performs backprop over a convolutional layer of a NN
"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Perform backpropagation over a convolutional layer of a neural network

    Parameters:
    - dZ (numpy.ndarray): partial derivatives with respect to the unactivated
    output of the convolutional layer, of shape (m, h_new, w_new, c_new)
    - A_prev (numpy.ndarray): output of the previous
    layer of shape (m, h_prev, w_prev, c_prev)
    - W (numpy.ndarray): kernels for the convolution,
    of shape (kh, kw, c_prev, c_new)
    - padding (str): 'same' or 'valid', indicating the type of padding used
    - stride (tuple): (sh, sw) containing the strides for the convolution

    Returns:
    - dA_prev (numpy.ndarray): partial derivatives with respect
    to the previous layer
    - dW (numpy.ndarray): partial derivatives with respect to the kernels
    - db (numpy.ndarray): partial derivatives with respect to the biases
    """
    # Convert all inputs to float64 for consistency
    dZ = dZ.astype(np.float64)
    A_prev = A_prev.astype(np.float64)
    W = W.astype(np.float64)
    b = b.astype(np.float64)

    # Get dimensions from the input shapes
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    # Initialize derivatives with zeros
    dA_prev = np.zeros_like(A_prev, dtype=np.float64)
    dW = np.zeros_like(W, dtype=np.float64)
    db = np.zeros_like(b, dtype=np.float64)

    # Determine padding values
    if padding == "same":
        ph = int(((h_prev - 1) * sh + kh - h_prev) / 2)
        pw = int(((w_prev - 1) * sw + kw - w_prev) / 2)
    elif padding == "valid":
        ph, pw = 0, 0
    else:
        raise ValueError("Padding must be 'same' or 'valid'")

    # Pad A_prev and dA_prev
    A_prev_padded = np.pad(
            A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
            mode='constant', constant_values=0)
    dA_prev_padded = np.pad(
            dA_prev, (
                (0, 0), (ph, ph), (pw, pw), (0, 0)),
            mode='constant', constant_values=0)

    # Compute the bias gradient
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    # Perform the backpropagation
    for i in range(h_new):
        for j in range(w_new):
            for k in range(c_new):
                # Define the slice for the current position
                vert_start = i * sh
                vert_end = vert_start + kh
                horiz_start = j * sw
                horiz_end = horiz_start + kw

                # Slice the padded input
                A_slice = A_prev_padded[
                        :, vert_start:vert_end, horiz_start:horiz_end, :]

                # Compute the gradients
                dW[:, :, :, k] += np.sum(A_slice * dZ[:, i, j, k][
                    :, None, None, None], axis=0)
                dA_prev_padded[:, vert_start:vert_end,
                               horiz_start:horiz_end, :] += W[
                                       :, :, :, k] * dZ[:, i, j, k][
                                               :, None, None, None]

    # Remove padding from dA_prev
    if padding == "same":
        if ph > 0:
            dA_prev = dA_prev_padded[:, ph:-ph, pw:-pw, :]
        else:
            dA_prev = dA_prev_padded
    elif padding == "valid":
        dA_prev = dA_prev_padded

    return dA_prev, dW, db
