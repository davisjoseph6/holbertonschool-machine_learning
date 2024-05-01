#!/usr/bin/env python3

"""
This module defines the NeuralNetwork class for binary classification with one hidden layer.
"""

import numpy as np


class NeuralNetwork:
    """
    Represents a neural network with one hidden layer performing binary classification.

    Attributes:
        W1 (numpy.ndarray): The weights vector for the hidden layer.
        b1 (numpy.ndarray): The bias for the hidden layer, initialized to zeros.
        A1 (float): The activated output for the hidden layer, initialized to zero.
        W2 (numpy.ndarray): The weights vector for the output neuron.
        b2 (float): The bias for the output neuron, initialized to zero.
        A2 (float): The activated output for the output neuron (prediction), initialized to zero.
    """

    def __init__(self, nx, nodes):
        """
        Initializes a NeuralNetwork instance.

        Args:
            nx (int): The number of input features.
            nodes (int): The number of nodes in the hidden layer.

        Raises:
            TypeError: If nx or nodes is not an integer.
            ValueError: If nx or nodes is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Initialize weights and biases for the hidden layer and output neuron
        self.W1 = np.random.randn(nodes, nx)  # Weights for hidden layer
        self.b1 = np.zeros((nodes, 1))        # Biases for hidden layer
        self.A1 = 0                           # Activation for hidden layer

        self.W2 = np.random.randn(1, nodes)   # Weights for output neuron
        self.b2 = 0                           # Bias for output neuron
        self.A2 = 0                           # Activation for output neuron
