#!/usr/bin/env python3
"""
Inception block module
"""

from tensorflow import keras as K


def inception_block(A_prev, filters):
    """
    Builds an inception block as described in
    'Going Deeper with Convolutions' (2014).

    Arguments:
    - A_prev is the output from the previous layer
    - filters is a tuple or list containing F1, F3R, F3, F5R, F5, FPP,
    respectively:
        - F1 is the number of filters in the 1x1 convolution
        - F3R is the number of filters in the 1x1 convolution before the
        3x3 convolution
        - F3 is the number of filters in the 3x3 convolution
        - F5R is the number of filters in the 1x1 convolution before the
        5x5 convolution
        - F5 is the number of filters in the 5x5 convolution
        - FPP is the number of filters in the 1x1 convolution after the max
        pooling

    - Returns:
        - The concatenated output of the inception block
    """

    F1, F3R, F3, F5R, F5, FPP = filters

    # 1x1 convolution branch
    conv_1x1 = K.layers.Conv2D(
            F1, (1, 1), padding='same', activation='relu')(A_prev)

    # 1x1 convolution before 3x3 convolution branch
    conv_3x3_reduce = K.layers.Conv2D(
            F3R, (1, 1), padding='same', activation='relu')(A_prev)
    conv_3x3 = K.layers.Conv2D(
            F3, (3, 3), padding='same', activation='relu')(conv_3x3_reduce)

    # 1x1 convolution before 5x5 convolution branch
    conv_5x5_reduce = K.layers.Conv2D(
            F5R, (1, 1), padding='same', activation='relu')(A_prev)
    conv_5x5 = K.layers.Conv2D(
            F5, (5, 5), padding='same', activation='relu')(conv_5x5_reduce)

    # 3x3 max pooling before 1x1 convolution branch
    max_pool = K.layers.MaxPooling2D(
            (3, 3), strides=(1, 1), padding='same')(A_prev)
    max_pool_conv = K.layers.Conv2D(
            FPP, (1, 1), padding='same', activation='relu')(max_pool)

    # Concatenate all branches
    output = K.layers.Concatenate(axis=-1)(
            [conv_1x1, conv_3x3, conv_5x5, max_pool_conv])

    return output
