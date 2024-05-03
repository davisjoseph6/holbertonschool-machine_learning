#!/usr/bin/env python3
"""
This script defines a Deep Neural Network for binary classification.
"""

import numpy as np


class DeepNeuralNetwork:
    """
    Define a deep neural network that does binary classification.
    """
    def __init__(self, nx, layers):
        """
        Initialize a deep neural network with given number of input features
        and layers.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")

        # Check if all layers are positive integers
        if not all(map(lambda x: isinstance(x, int) and x > 0, layers)):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)  # number of layers
        self.__cache = {}  # to store all intermediary values of the network
        self.__weights = {}  # to hold all weights and biases of the network

        # Initialize weights and biases using He et al. method 4 each layer
        for layer_index in range(1, self.__L + 1):
            layer_size = layers[layer_index - 1]
            prev_layer_size = nx if layer_index == 1 else layers[
                    layer_index - 2
                    ]
            self.__weights[f'W{layer_index}'] = (
                    np.random.randn(layer_size, prev_layer_size) * np.sqrt(
                        2 / prev_layer_size
                        )
                    )
            self.__weights[f'b{layer_index}'] = np.zeros((layer_size, 1))

    @property
    def L(self):
        """
        Getter 4 number of layers.
        """
        return self.__L

    @property
    def cache(self):
        """
        Getter 4 cache.
        """
        return self.__cache

    @property
    def weights(self):
        """
        Getter 4 weights.
        """
        return self.__weights
