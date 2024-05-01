#!/usr/bin/env python3
"""
NeuralNetwork class that defines a neural network with one hidden layer
performing binary classification.
"""

import numpy as np

class NeuralNetwork:
    """
    Defines a neural network with one hidden layer performing binary classification.
    """
    def __init__(self, nx, nodes):
        """
        Constructor for the NeuralNetwork class.
        
        Parameters:
        nx (int): The number of input features.
        nodes (int): The number of nodes in the hidden layer.

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

        # Initialize the weights, biases, and activations for the hidden layer
        self.W1 = np.random.randn(nodes, nx)  # Weight matrix for the hidden layer
        self.b1 = np.zeros((nodes, 1))        # Bias vector for the hidden layer
        self.A1 = np.zeros((nodes, 1))        # Activation output for the hidden layer

        # Initialize the weights, biases, and activation for the output layer
        self.W2 = np.random.randn(1, nodes)   # Weight matrix for the output layer
        self.b2 = np.zeros((1, 1))            # Bias for the output layer
        self.A2 = np.zeros((1, 1))            # Activation output for the output layer
