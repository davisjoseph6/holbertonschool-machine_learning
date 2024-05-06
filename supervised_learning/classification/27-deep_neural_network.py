#!/usr/bin/env python3
"""
This script defines a Deep Neural Network for multiclass classification.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle

class DeepNeuralNetwork:
    """
    Represents a deep neural network performing multiclass classification.
    """

    def __init__(self, nx, layers):
        """
        Class constructor

        Parameters:
        nx (int): number of input features
        layers (list of int): number of nodes in each layer of the network

        Exceptions:
        TypeError: If nx is not an integer or layers is not a list of the positive integers.
        ValueError: If nx is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")
        if any(not isinstance(x, int) or x <= 0 for x in layers):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for l in range(1, self.__L + 1):
            layer_size = layers[l - 1]
            prev_layer_size = nx if l == 1 else layers[l - 2]

            self.__weights['W' + str(l)] = np.random.randn(layer_size, prev_layer_size) * np.sqrt(2 / prev_layer_size)
            self.__weights['b' + str(l)] = np.zeros((layer_size, 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        """
        self.__cache['A0'] = X
        for l in range(1, self.__L + 1):
            W = self.__weights['W' + str(l)]
            b = self.__weights['b' + str(l)]
            A_prev = self.__cache['A' + str(l - 1)]
            Z = np.dot(W, A_prev) + b
            if l == self.__L:  # Apply softmax activation function at the output layer
                t = np.exp(Z - np.max(Z, axis=0, keepdims=True))
                self.__cache['A' + str(l)] = t / np.sum(t, axis=0, keepdims=True)
            else:  # Apply sigmoid activation function to hidden layers
                self.__cache['A' + str(l)] = 1 / (1 + np.exp(-Z))
        return self.__cache['A' + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using categorical cross-entropy
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A + 1e-8)) / m
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network’s predictions
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        predictions = np.argmax(A, axis=0)
        return predictions, cost

    def gradient_descent(self, Y, alpha=0.05):
        """
        Performs one pass of gradient descent on the neural network
        """
        m = Y.shape[1]
        A = self.__cache['A' + str(self.__L)]
        dA = -(Y - A)

        for l in reversed(range(1, self.__L + 1)):
            A_prev = self.__cache['A' + str(l - 1)]
            W = self.__weights['W' + str(l)]
            dZ = dA
            dW = np.dot(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            if l > 1:
                dA = np.dot(W.T, dZ) * A_prev * (1 - A_prev)
            self.__weights['W' + str(l)] -= alpha * dW
            self.__weights['b' + str(l)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """
        Trains the neural network
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if not isinstance(verbose, bool):
            raise TypeError("verbose must be a boolean")
        if not isinstance(graph, bool):
            raise TypeError("graph must be a boolean")
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        for i in range(iterations):
            A, _ = self.forward_prop(X)
            self.gradient_descent(Y, alpha)
            cost = self.cost(Y, A)
            if verbose and (i % step == 0 or i == iterations - 1):
                print(f"Cost after {i} iterations: {cost}")
            if graph:
                costs.append(cost)

        if graph:
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per hundreds)')
            plt.title("Learning rate =" + str(alpha))
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """
        Save the instance to a file in pickle format.
        """
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        """
        Load a saved instance from a file.
        """
        try:
            with open(filename, 'rb') as file:
                return pickle.load(file)
        except FileNotFoundError:
            return None

