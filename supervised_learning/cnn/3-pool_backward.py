#!/usr/bin/env python3
"""
Pooling Back propagation
"""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Perform back propagation over a pooling layer of a neural network
    """
    # Extract dimensions from the input shapes
    m, h_new, w_new, c = dA.shape
    h_prev, w_prev = A_prev.shape[1], A_prev.shape[2]
    kh, kw = kernel_shape
    sh, sw = stride

    # Initialize the gradient for the previous layer with zeros
    dA_prev = np.zeros_like(A_prev)

    # Perform backpropagation through the pooling layer
    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for channel in range(c):
                    # Define the slice boundaries
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    if mode == 'max':
                        # Find the maximum value in the slice and its mask
                        A_slice = A_prev[i, vert_start:vert_end,
                                         horiz_start:horiz_end, channel]
                        mask = (A_slice == np.max(A_slice))
                        dA_prev[i, vert_start:vert_end,
                                horiz_start:horiz_end, channel] += (
                                        mask * dA[i, h, w, channel]
                                        )

                    elif mode == 'avg':
                        # Compute the average gradient distribution
                        da = dA[i, h, w, channel]
                        shape = (kh, kw)
                        average_dA = da / (kh * kw)
                        dA_prev[i, vert_start:vert_end,
                                horiz_start:horiz_end, channel] += (
                                        np.ones(shape) * average_dA
                                        )

    return dA_prev
