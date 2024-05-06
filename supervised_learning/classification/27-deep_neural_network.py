#!/usr/bin/env python3
"""
This script defines a Deep Neural Network for multiclass classification.
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle

class DeepNeuralNetwork:
    """
    Represents a deep neural network for multiclass classification.
    """
    def __init__(self, nx, layers):
        """
        Initialize the deep neural network.

        Parameters:
            nx (int): Number of input features.
            layers (list of int): Number of nodes in each layer.

        Raises:
            TypeError: If nx is not an integer or layers is not a list of the correct format.
            ValueError: If nx is less than 1 or layers contains non-positive integers.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")
        if any(type(layer) != int or layer <= 0 for layer in layers):
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
        Perform forward propagation of the neural network.

        Args:
            X (np.ndarray): The input data.

        Returns:
            tuple: The output of the last layer and the cache.
        """
        self.__cache['A0'] = X
        for l in range(1, self.__L + 1):
            A_prev = self.__cache['A' + str(l - 1)]
            W = self.__weights['W' + str(l)]
            b = self.__weights['b' + str(l)]
            Z = np.dot(W, A_prev) + b
            if l == self.__L:
                # Softmax activation function on the output layer
                t = np.exp(Z - np.max(Z, axis=0, keepdims=True))
                self.__cache['A' + str(l)] = t / np.sum(t, axis=0, keepdims=True)
            else:
                # Sigmoid activation function for hidden layers
                self.__cache['A' + str(l)] = 1 / (1 + np.exp(-Z))
        return self.__cache['A' + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """
        Calculate the cross-entropy cost of the network.

        Args:
            Y (np.ndarray): Correct labels for input data.
            A (np.ndarray): Output from the forward propagation.

        Returns:
            float: Cost.
        """
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(Y * np.log(A + 1e-8))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluate the neural network's predictions.

        Args:
            X (np.ndarray): Input data.
            Y (np.ndarray): True labels for input data.

        Returns:
            tuple: Predictions and the cost.
        """
        A, _ = self.forward_prop(X)
        predictions = np.argmax(A, axis=0)
        one_hot_predictions = np.eye(Y.shape[0])[predictions]
        cost = self.cost(Y, A)
        return one_hot_predictions, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Perform one pass of gradient descent on the neural network.
        """
        m = Y.shape[1]
        A = cache[f'A{self.__L}']

        # Start with the gradient of cost with respect to the output activation
        dZ = A - Y

        for layer_index in reversed(range(1, self.__L + 1)):
            A_prev = cache[f'A{layer_index-1}']
            A_curr = cache[f'A{layer_index}']
            W = self.__weights[f'W{layer_index}']

            # Gradient computation
            dW = np.dot(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            if layer_index > 1:
                # Compute dZ for the next layer
                dZ = np.dot(W.T, dZ) * A_prev * (1 - A_prev)

            # Update weights and biases
            self.__weights[f'W{layer_index}'] -= alpha * dW
            self.__weights[f'b{layer_index}'] -= alpha * db


    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """
        Train the neural network.

        Args:
            X (np.ndarray): Input data.
            Y (np.ndarray): True labels.
            iterations (int): Number of iterations.
            alpha (float): Learning rate.
            verbose (bool): Whether to print progress.
            graph (bool): Whether to plot a graph.

        Returns:
            tuple: The output predictions and cost of the network.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        costs = []
        for i in range(iterations + 1):
            A, _ = self.forward_prop(X)
            cost = self.cost(Y, A)
            if i % step == 0 or i == iterations:
                if verbose:
                    print(f"Cost after {i} iterations: {cost}")
                costs.append(cost)
            self.gradient_descent(Y, self.__cache, alpha)

        if graph:
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per hundreds)')
            plt.title("Learning rate =" + str(alpha))
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """
        Save the neural network to a file.

        Args:
            filename (str): The file path where the object should be saved.
        """
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        """
        Load a neural network from a file.

        Args:
            filename (str): The file path to load the neural network from.

        Returns:
            DeepNeuralNetwork: The loaded neural network or None if file does not exist.
        """
        try:
            with open(filename, 'rb') as file:
                loaded_network = pickle.load(file)
            return loaded_network
        except FileNotFoundError:
            return None

