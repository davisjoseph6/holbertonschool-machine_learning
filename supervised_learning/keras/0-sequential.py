#!/usr/bin/env python3
"""
Builds a neural netowrk using the keras library
"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network using the Keras library

    Args:
        nx (int): The number of nput features to the network
        layers (list): A list containing the number of nodes in each layer
        of the network.
        activations (list): A list containing the activation functions used
        for each layerof the network
        lambtha (float): The L2 regularization parameter.
        keep_prob (float): The probability that node will be kept for droppout

    Returns:
    K.Sequential: The constructed Keras Sequential model
    """
    model = K.Sequential()
    for i in range(len(layers)):
        if i == 0:
            model.add(K.layers.Dense(layers[i], input_shape=(nx,),
                      activation=activations[i],
                      kernel_regularizer=K.regularizers.l2(lambtha)))
        else:
            model.add(K.layers.Dense(layers[i], activation=activations[i],
                      kernel_regularizer=K.regularizers.l2(lambtha)))
        if i < len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))
    return model
