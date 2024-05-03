#!/usr/bin/env python3

import numpy as np

class DeepNeuralNetwork:
    """
    Define a deep neural network performing binary classification.
    """
    def __init__(self, nx, layers):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(layer, int) and layer > 0 for layer in layers):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)  # number of layers
        self.cache = {}  # to store all intermediary values of the network
        self.weights = {}  # to hold all weights and biases of the network

        # Initialize weights and biases using He et al. method for each layer
        layer_dims = [nx] + layers  # layer dimensions from input to last layer
        for l in range(1, self.L + 1):
            self.weights[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2 / layer_dims[l-1])
            self.weights[f'b{l}'] = np.zeros((layer_dims[l], 1))
