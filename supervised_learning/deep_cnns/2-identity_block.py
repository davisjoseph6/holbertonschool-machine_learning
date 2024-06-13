#!/usr/bin/env python3
"""
    Identity Block
"""

import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
        build an identity block as described in
        'Deep Residual Learning for Image Recognition' (2015)

    :param A_prev: output from previous layer
    :param filters: tuple or list containing
        * F11 : number of filters in first 1x1 conv
        * F3: number of filters in 3x3 conv
        * F12: number of filters in second 1x1 conv
    Each conv layer followed by batch normalization along channels axis
    and ReLu
    He Normal initialization

    :return: activated output of the identity block
    """
    # filters extraction
    F11, F3, F12 = filters

    # initializer
    initializer = K.initializers.HeNormal()

    # First layer
    conv1 = K.layers.Conv2D(F11,
                            kernel_initializer=initializer,
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            padding='same')(A_prev)
    batchN1 = K.layers.BatchNormalization(axis=3)(conv1)
    relu1 = K.layers.Activation(activation='relu')(batchN1)

    # second layer
    conv2 = K.layers.Conv2D(F3,
                            kernel_initializer=initializer,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same')(relu1)

    batchN2 = K.layers.BatchNormalization(axis=3)(conv2)
    relu2 = K.layers.Activation(activation='relu')(batchN2)

    # third layer
    conv3 = K.layers.Conv2D(F12,
                            kernel_size=(1, 1),
                            kernel_initializer=initializer,
                            strides=(1, 1),
                            padding='same')(relu2)
    batchN3 = K.layers.BatchNormalization(axis=3)(conv3)

    # add input (Residual Network)
    resnet = K.layers.add([batchN3, A_prev])

    # last activation
    output = K.layers.Activation(activation='relu')(resnet)

    return output
