#!/usr/bin/env python3
"""
Builds a neural network using the Keras library with the Functional API
"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network using the Keras library with the Functional API

    Args:
        nx (int): The number of input features to the network
        layers (list): A list containing the number of nodes in each layer
        of the network.
        activations (list): A list containing the activation functions used
        for each layer of the network
        lambtha (float): The L2 regularization parameter.
        keep_prob (float): The probability that a node will be kept for
        dropout

    Returns:
    K.Model: The constructed Keras model
    """
    inputs = K.Input(shape=(nx,))
    x = inputs

    for i in range(len(layers)):
        x = K.layers.Dense(layers[i],
                           activation=activations[i],
                           kernel_regularizer=K.regularizers.l2(lambtha))(x)
        if i < len(layers) - 1:
            x = K.layers.Dropout(1 - keep_prob)(x)

    model = K.Model(inputs=inputs, outputs=x)
    return model
