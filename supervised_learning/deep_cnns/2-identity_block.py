#!/usr/bin/env python3
"""
Identity block module
"""

import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
    Builds an identity block as described in
    'Deep Residual Learning for Image Recognition' (2015).

    Arguments:
    - A_prev is the output from the previous layer
    - filters is a tuple or list containing F11, F3, F12, respectively:
       - F11 is the number of filters in the first 1x1 convolution
       - F3 is the number of filters in the 3x3 convolution
       - F12 is the number of filters in the second 1x1 convolution

    Returns:
    - The activated output of the identity block
    """
    F11, F3, F12 = filters
    he_normal = K.initializers.he_normal(seed=0)

    # First component of the main path
    conv1 = K.layers.Conv2D(
            F11, (1, 1), padding='same', kernel_initializer=he_normal)(A_prev)
    bn1 = K.layers.BatchNormalization(axis=3)(conv1)
    act1 = K.layers.Activation('relu')(bn1)

    # Second component of the main path
    conv2 = K.layers.Conv2D(
            F3, (3, 3), padding='same', kernel_initializer=he_normal)(act1)
    bn2 = K.layers.BatchNormalization(axis=3)(conv2)
    act2 = K.layers.Activation('relu')(bn2)

    # Third component of the main path
    conv3 = K.layers.Conv2D(
            F12, (1, 1), padding='same', kernel_initializer=he_normal)(act2)
    bn3 = K.layers.BatchNormalization(axis=3)(conv3)

    # Add shortcut value to the main path,
    # and pass it through a RELU activation
    add = K.layers.Add()([bn3, A_prev])
    act3 = K.layers.Activation('relu')(add)

    return act3
