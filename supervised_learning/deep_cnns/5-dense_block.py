#!/usr/bin/env python3
"""
Dense Block
"""

from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Builds a dense block as described in 'Densely Connected Convolutional Networks' (2016)'

    Parameters:
    X : tensor
        The output of the previous layer.
    nb_filters : int
        The number of filters in X.
    growth_rate : int
        The growth rate for the dense block.
    layers : int
        The number of layers in the dense block.

    Returns:
    tensor, int
        The concatenated output of each layer within the Dense Block and the number of filters within the concatenated outputs, respectively.
    """
    init = K.initializers.HeNormal(seed=0)
    concat_features = X

    for _ in range(layers):
        # Batch Normalization and ReLU
        bn1 = K.layers.BatchNormalization(axis=-1)(concat_features)
        relu1 = K.layers.Activation('relu')(bn1)

        # 1x1 Convolution (Bottleneck layer)
        conv1 = K.layers.Conv2D(4 * growth_rate, (1, 1), padding='same', kernel_initializer=init)(relu1)

        # Batch Normalization and ReLU
        bn2 = K.layers.BatchNormalization(axis=-1)(conv1)
        relu2 = K.layers.Activation('relu')(bn2)

        # 3x3 Convolution
        conv2 = K.layers.Conv2D(growth_rate, (3, 3), padding='same', kernel_initializer=init)(relu2)

        # Concatenate input with output of the 3x3 convolution
        concat_features = K.layers.Concatenate(axis=-1)([concat_features, conv2])
        nb_filters += growth_rate

    return concat_features, nb_filters
