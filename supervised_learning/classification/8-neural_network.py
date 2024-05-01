#!/usr/bin/env python3
"""
This is the NeuralNetwork class module for binary classification.
"""

import numpy as np

class NeuralNetwork:
    """
    Represents a neural network with one hidden layer performing binary classification.
    
    Attributes:
        W1 (numpy.ndarray): Weights vector for the hidden layer.
        b1 (numpy.ndarray): Bias for the hidden layer, initialized as zeros.
        A1 (float): Activated output for the hidden layer, initialized to 0.
        W2 (numpy.ndarray): Weights vector for the output neuron.
        b2 (float): Bias for the output neuron, initialized to 0.
        A2 (float): Activated output for the output neuron (prediction), initialized to 0.
    """

    def __init__(self, nx, nodes):
        """
        Initializes a NeuralNetwork with one hidden layer performing binary classification.

        Args:
            nx (int): Number of input features.
            nodes (int): Number of nodes in the hidden layer.

        Raises:
            TypeError: If nx or nodes are not integers.
            ValueError: If nx or nodes are less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Initialize weights, biases, and activations for both layers
        self.W1 = np.random.randn(nodes, nx) * 0.01  # Small random weights
        self.b1 = np.zeros((nodes, 1))  # Bias vector of zeros
        self.A1 = 0  # Activation initialized to 0

        self.W2 = np.random.randn(1, nodes) * 0.01  # Small random weights for output layer
        self.b2 = 0  # Bias initialized to 0
        self.A2 = 0  # Output activation initialized to 0
