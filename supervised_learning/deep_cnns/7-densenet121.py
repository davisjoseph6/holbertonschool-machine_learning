#!/usr/bin/env python3
"""
DenseNet-121 Implementation
"""

from tensorflow import keras as K

dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Builds the DenseNet-121 architecture as described in 'Densely Connected Convolutional Networks'.

    Parameters:
    growth_rate (int): The growth rate.
    compression (float): The compression factor.

    Returns:
    keras.Model: The keras model.
    """
    init = K.initializers.HeNormal(seed=0)
    input_shape = (224, 224, 3)

    X_input = K.Input(shape=input_shape)

    # Initial Convolution and Pooling
    X = K.layers.BatchNormalization(axis=-1)(X_input)
    X = K.layers.Activation('relu')(X)
    X = K.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', kernel_initializer=init)(X)
    X = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(X)

    # Dense Block 1
    X, nb_filters = dense_block(X, 64, growth_rate, 6)

    # Transition Layer 1
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 2
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 12)

    # Transition Layer 2
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 3
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 24)

    # Transition Layer 3
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 4
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 16)

    # Final Batch Norm and ReLU
    X = K.layers.BatchNormalization(axis=-1)(X)
    X = K.layers.Activation('relu')(X)

    # Global Average Pooling
    X = K.layers.GlobalAveragePooling2D()(X)

    # Fully Connected Layer
    X = K.layers.Dense(1000, activation='softmax', kernel_initializer=init)(X)

    model = K.Model(inputs=X_input, outputs=X)
    return model
