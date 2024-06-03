#!/usr/bin/env python3
"""
    Convolutional Forward Propagation
"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
        function that performs forward propagation over a conv layer of NN

        :param A_prev: ndarray, shape(m,h_prev,w_prev,c_prev) output layer
        :param W: ndarray, shape(kh,kw,c_prev,c_new) kernel
        :param b: ndarray, shape(1,1,1,c_new) biases
        :param activation: activation function
        :param padding: string 'same' or 'valid'
        :param stride: tuple (sh,sw)

        :return: output of the convolutional layer
    """
    # size output layer, kernel, stride
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    # output size and padding
    if padding == 'valid':
        # no padding
        ph, pw = 0, 0
    elif padding == 'same':
        ph = int((((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int((((w_prev - 1) * sw + kw - w_prev) / 2))

    # output size
    output_height = int((h_prev - kh + 2 * ph) / sh + 1)
    output_width = int((w_prev - kw + 2 * pw) / sw + 1)

    # initialize output
    convolved_images = np.zeros((m, output_height, output_width, c_new))

    # pad image
    image_pad = np.pad(A_prev,
                       ((0, 0), (ph, ph),
                        (pw, pw), (0, 0)), mode='constant')

    # convolution
    for k in range(c_new):
        for h in range(output_height):
            for w in range(output_width):
                # extract region from each image
                image_zone = image_pad[:, h * sh:h * sh + kh,
                                       w * sw:w * sw + kw, :]

                # element wize multiplication
                convolved_images[:, h, w, k] = np.sum(image_zone
                                                      * W[:, :, :, k],
                                                      axis=(1, 2, 3))

    # add bias
    Z = convolved_images + b

    # apply activation function
    Z = activation(Z)

    return Z
