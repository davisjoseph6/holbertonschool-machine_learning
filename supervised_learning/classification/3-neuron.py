#!/usr/bin/env python3
"""
Wrote a class Neuron that defines a single neuron performing binary
classification
"""

import numpy as np


class Neuron:
    """
    Class that defines a single neuron performing binary classification
    """
    def __init__(self, nx):
        """
        Constructor for the neuron class.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Weight vector initialized using a random normal distribution
        self.__W = np.random.randn(1, nx)
        # Bias initialized to 0.
        self.__b = 0
        # Activation output initialized to 0.
        self.__A = 0

    @property
    def W(self):
        """
        Getter for the private attribute __W.
        """
        return self.__W

    @property
    def b(self):
        """
        Getter for the private attribute __b.
        """
        return self.__b

    @property
    def A(self):
        """
        Getter for the private attribute __A
        """
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron.
        """
        # Calculate the linear part of the neuron (Z = W.X + b)
        Z = np.dot(self.__W, X) + self.__b
        # Apply the sigmoid activation function
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        """
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(Y * np.log(A) +
                                 (1 - Y) * np.log(1.0000001 - A))
        return cost
