#!/usr/bin/env python3
"""
Transition Layer
"""

from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer as described in
    'Densely Connected Convolutional Networks' (2016).

    Parameters:
    X : tensor
        The output of the previous layer.
    nb_filters : int
        The number of filters in X.
    compression : float
        The compression factor for the transition layer.

    Returns:
    tensor, int
        The output of the transition layer and the number of filters
        within the output, respectively.
    """
    init = K.initializers.HeNormal(seed=0)
    compressed_filters = int(nb_filters * compression)

    # Batch Normalization and ReLU
    bn = K.layers.BatchNormalization(axis=-1)(X)
    relu = K.layers.Activation('relu')(bn)

    # 1x1 Convolution
    conv = K.layers.Conv2D(compressed_filters,
                           (1, 1), padding='same',
                           kernel_initializer=init)(relu)

    # Average Pooling
    avg_pool = K.layers.AveragePooling2D((2, 2),
                                         strides=(2, 2),
                                         padding='same')(conv)

    return avg_pool, compressed_filters
