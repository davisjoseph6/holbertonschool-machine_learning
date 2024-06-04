#!/usr/bin/env python3
"""
Performs forward propagation over a pooling layer of a neural network.
"""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Perform forward propagation over a pooling layer of a neural network

    Parameters:
    - A_prev (numpy.ndarray): output of the previous layer
    with shape (m, h_prev, w_prev, c_prev)
       - m is the number of examples
       - h_prev is the height of the previous layer
       - w_prev is the width of the previous layer
       - c_prev is the number of channels in the previous layer
    - kernel_shape (tuple): (kh, kw) containing the size of the kernel for
    the pooling
       - kh is the kernel height
       - kw is the kernel width
    - stride (tuple): (sh, sw) containing the strides for the pooling
       - sh is the stride for the height
       - sw is the stride for the width
    - mode (str): 'max' or 'avg', indicating whether to perform maximum or
    average pooling

    Returns:
    - numpy.ndarray: the output of the pooling layer
    """
    # Get dimensions from the input shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Calculate the dimensions of the output
    h_new = int((h_prev - kh) / sh + 1)
    w_new = int((w_prev - kw) / sw + 1)

    # Initialize the output volume with zeros
    Z = np.zeros((m, h_new, w_new, c_prev))

    # Perform the pooling operation
    for i in range(h_new):
        for j in range(w_new):
            # Define the vertical and horizontal start and
            # end points for the slice
            vert_start = i * sh
            vert_end = vert_start + kh
            horiz_start = j * sw
            horiz_end = horiz_start + kw

            # Extract the slice from the input array
            A_slice = A_prev[:, vert_start:vert_end, horiz_start:horiz_end, :]

            # Perform the pooling operation based on the mode
            if mode == 'max':
                # Maximum pooling
                Z[:, i, j, :] = np.max(A_slice, axis=(1, 2))
            elif mode == 'avg':
                # Average pooling
                Z[:, i, j, :] = np.mean(A_slice, axis=(1, 2))
            else:
                raise ValueError("Mode must be 'max' or 'avg'")

    return Z
