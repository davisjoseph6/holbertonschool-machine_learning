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

    def sigmoid(self, Z):
        """
        Sigmoid activation function.
        """
        return 1 / (1 + np.exp(-Z))

    def forward_prop(self, X):
        """
        Perform forward propagation for the neural network.
        """
        self.__cache['A0'] = X
        for layer_index in range(1, self.__L + 1):
            W = self.__weights[f'W{layer_index}']
            b = self.__weights[f'b{layer_index}']
            A_prev = self.__cache[f'A{layer_index-1}']
            Z = np.dot(W, A_prev) + b
            self.__cache[f'A{layer_index}'] = self.sigmoid(Z)

        AL = self.__cache[f'A{self.__L}']
        return AL, self.__cache

    def cost(self, Y, A):
        """
        Calculate the cost using the logistic regression 'cross-entropy'
        cost function.
        """
        m = Y.shape[1]  # number of examples
        cost = -(1 / m) * np.sum(Y * np.log(A) + (
            1 - Y
            ) * np.log(1.0000001 - A)
            )
        return cost

    def evaluate(self, X, Y):
        """
        Evaluate the neural network's predictions against the true labels.
        """
        # Perform forward propagation to get outputs
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)  # Compute the cost with the actual labels
        # Convert probabilities to binary output
        predictions = (A >= 0.5).astype(int)
        return predictions, cost
