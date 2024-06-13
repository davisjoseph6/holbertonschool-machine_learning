#!/usr/bin/env python3
"""
Projection Block
"""

from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """
    Builds a projection block as described in
    'Deep Residual Learning for Image Recognition' (2015).

    Parameters:
    A_prev : tensor
        The output of the previous layer.
    filters : tuple or list
        Contains F11, F3, F12 respectively:
            F11 : int
                Number of filters in the first 1x1 convolution.
            F3 : int
                Number of filters in the 3x3 convolution.
            F12 : int
                Number of filters in the second 1x1 convolution as well
                as the 1x1 convolution in the shortcut connection.
    s : int
        Stride of the first convolution in both the main path and the shortcut
        connection.

    Returns:
    tensor
        The activated output of the projection block.
    """
    F11, F3, F12 = filters

    # Initializer he_normal with seed 0
    init = K.initializers.HeNormal(seed=0)

    # First layer of main path
    conv1 = K.layers.Conv2D(filters=F11,
                            kernel_size=(1, 1),
                            strides=(s, s),
                            padding="same",
                            kernel_initializer=init)(A_prev)
    norm1 = K.layers.BatchNormalization(axis=-1)(conv1)
    relu1 = K.layers.Activation(activation="relu")(norm1)

    # Second layer of main path
    conv2 = K.layers.Conv2D(filters=F3,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding="same",
                            kernel_initializer=init)(relu1)
    norm2 = K.layers.BatchNormalization(axis=-1)(conv2)
    relu2 = K.layers.Activation(activation="relu")(norm2)

    # Final layer of main path
    conv3 = K.layers.Conv2D(filters=F12,
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            padding="same",
                            kernel_initializer=init)(relu2)
    norm3 = K.layers.BatchNormalization(axis=-1)(conv3)

    # Shortcut path
    conv_shortcut = K.layers.Conv2D(filters=F12,
                                    kernel_size=(1, 1),
                                    strides=(s, s),
                                    padding="same",
                                    kernel_initializer=init)(A_prev)
    norm_shortcut = K.layers.BatchNormalization(axis=-1)(conv_shortcut)

    # Merge output of main path and shortcut path
    merged = K.layers.Add()([norm3, norm_shortcut])

    # Return activated output of merge, using ReLU.
    return K.layers.Activation(activation="relu")(merged)
