#!/usr/bin/env python3
import numpy as np

def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Perform forward propagation over a convolutional layer of a neural network.
    
    Parameters:
    - A_prev (numpy.ndarray): output of the previous layer with shape (m, h_prev, w_prev, c_prev)
    - W (numpy.ndarray): kernels for the convolution with shape (kh, kw, c_prev, c_new)
    - b (numpy.ndarray): biases applied to the convolution with shape (1, 1, 1, c_new)
    - activation (function): activation function applied to the convolution
    - padding (str): 'same' or 'valid', indicating the type of padding used
    - stride (tuple): (sh, sw) containing the strides for the convolution
    
    Returns:
    - numpy.ndarray: the output of the convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    if padding == "same":
        ph = int(((h_prev - 1) * sh + kh - h_prev) / 2)
        pw = int(((w_prev - 1) * sw + kw - w_prev) / 2)
    elif padding == "valid":
        ph, pw = 0, 0
    else:
        raise ValueError("Padding must be 'same' or 'valid'")

    h_new = int((h_prev - kh + 2 * ph) / sh + 1)
    w_new = int((w_prev - kw + 2 * pw) / sw + 1)

    A_prev_padded = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant', constant_values=0)
    
    Z = np.zeros((m, h_new, w_new, c_new))

    for i in range(h_new):
        for j in range(w_new):
            for k in range(c_new):
                vert_start = i * sh
                vert_end = vert_start + kh
                horiz_start = j * sw
                horiz_end = horiz_start + kw

                A_slice = A_prev_padded[:, vert_start:vert_end, horiz_start:horiz_end, :]
                Z[:, i, j, k] = np.sum(A_slice * W[:, :, :, k], axis=(1, 2, 3))
    
    Z = Z + b
    A = activation(Z)
    
    return A
