#!/usr/bin/env python3
"""
A function that performs forward propagation over a convolutional neural network.
"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    A function that performs forward propagation over a convolutional neural network.
    """
    # Get dimensions from A_prev's shape
    (m, h_prev, w_prev, c_prev) = A_prev.shape

    # Get dimensions from W's shape
    (kh, kw, _, c_new) = W.shape

    # Get strides
    (sh, sw) = stride

    # Determine padding dimensions
    if padding == "same":
        ph = int(((h_prev - 1) * sh + kh - h_prev) / 2)
        pw = int(((w_prev - 1) * sw + kw - w_prev) / 2)
    elif padding == "valid":
        ph = pw = 0
    else:
        raise ValueError("Padding must be 'same' or 'valid'")

    # Pad A_prev if necessary
    A_prev_pad = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')

    # Determine the dimensions of the output
    h_new = int((h_prev - kh + 2 * ph) / sh) + 1
    w_new = int((w_prev - kw + 2 * pw) / sw) + 1

    # Initialize the output volume Z
    Z = np.zeros((m, h_new, w_new, c_new))

    # Perform the convolution
    for i in range(m):  # loop over the batch of training examples
        for h in range(h_new):  # loop over the output height
            for w in range(w_new):  # loop over the output width
                for c in range(c_new):  # loop over the output channels
                    h_start = h * sh
                    h_end = h_start + kh
                    w_start = w * sw
                    w_end = w_start + kw

                    # Slice the input volume
                    A_slice = A_prev_pad[i, h_start:h_end, w_start:w_end, :]

                    # Perform the convolution
                    Z[i, h, w, c] = np.sum(A_slice * W[:, :, :, c]) + float(b[:, :, :, c])

    # Apply the activation function
    A = activation(Z)

    return A
