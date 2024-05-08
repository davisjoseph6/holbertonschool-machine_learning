#!/usr/bin/env python3
"""
This script defines a Deep Neural Network for multi-class classification.
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle

class DeepNeuralNetwork:
    """
    Define a deep neural network performing multi-class classification.
    """
    def __init__(self, nx, layers):
        """
        Initialize a deep neural network with given number of input features
        and layers.
        """
        np.random.seed(0)
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(x, int) and x > 0 for x in layers):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)  # number of layers
        self.__cache = {}  # to store all intermediary values of the network
        self.__weights = {}  # to hold all weights and biases of the network

        # Initialize weights and biases using He et al. method for each layer
        for layer_index in range(1, self.__L + 1):
            layer_size = layers[layer_index - 1]
            prev_layer_size = nx if layer_index == 1 else layers[layer_index - 2]
            self.__weights[f'W{layer_index}'] = (
                np.random.randn(layer_size, prev_layer_size) * np.sqrt(2 / prev_layer_size)
            )
            self.__weights[f'b{layer_index}'] = np.zeros((layer_size, 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def softmax(self, Z):
        """Softmax activation function for output layer"""
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # Numerical stability
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

    def forward_prop(self, X):
        """Calculate the forward propagation of the neural network, including softmax for the final layer"""
        self.__cache['A0'] = X
        for i in range(1, self.__L):
            W = self.__weights[f'W{i}']
            b = self.__weights[f'b{i}']
            Z = np.dot(W, self.__cache[f'A{i-1}']) + b
            self.__cache[f'A{i}'] = 1 / (1 + np.exp(-Z))  # Sigmoid activation for hidden layers

        # Softmax activation for the output layer
        Z_final = np.dot(self.__weights[f'W{self.__L}'], self.__cache[f'A{self.__L-1}']) + self.__weights[f'b{self.__L}']
        self.__cache[f'A{self.__L}'] = self.softmax(Z_final)
        return self.__cache[f'A{self.__L}'], self.__cache

    def cost(self, Y, A):
        """Calculate the cost of the model using softmax output"""
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A + 1e-8)) / m  # Adding a small value to avoid log(0)
        return cost

    def evaluate(self, X, Y):
        """Evaluate the neural network's predictions"""
        A, _ = self.forward_prop(X)
        predictions = np.argmax(A, axis=0)
        one_hot_predictions = np.eye(Y.shape[0])[predictions].T  # Convert predictions to one-hot
        cost = self.cost(Y, A)
        return one_hot_predictions, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Perform one pass of gradient descent on the neural network.
        """
        m = Y.shape[1]  # Number of examples
        L = self.__L  # Number of layers

        dZ = cache[f'A{L}'] - Y  # Difference at output layer

        for layer_index in reversed(range(1, L + 1)):
            A_prev = cache[f'A{layer_index-1}']
            A_curr = cache[f'A{layer_index}']
            W = self.__weights[f'W{layer_index}']

            dW = np.dot(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            if layer_index > 1:
                # Prepare dZ for the next layer (element-wise)
                dZ = np.dot(W.T, dZ) * A_prev * (1 - A_prev)

            # Update weights and biases
            self.__weights[f'W{layer_index}'] -= alpha * dW
            self.__weights[f'b{layer_index}'] -= alpha * db

        return self.__weights

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """
        Trains the deep neural network using forward propagation and gradient descent.
        """
        # List to store costs for potential plotting
        costs = []
        count = []

        for i in range(iterations + 1):
            # Forward propagation
            A, cache = self.forward_prop(X)

            # Gradient descent on all iterations except the last
            if i != iterations:
                self.gradient_descent(Y, self.cache, alpha)

                # Calculate cost
                cost = self.cost(Y, A)

                # Store costs for plotting
                costs.append(cost)
                count.append(i)

                # Verbose condition to print the cost periodically
                if verbose and (i % step == 0 or i == 0 or i == iterations):
                    print("Cost after {} iterations: {}".format(i, cost))

        # Plotting the cost graph if required
        if graph:
            plt.plot(count, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        # Evaluate and return the final performance after training
        return self.evaluate(X, Y)

    def save(self, filename):
        """
        Save the model to a file in pickle format.
        """
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        """
        Load a saved deep neural network model.
        """
        try:
            with open(filename, 'rb') as file:
                loaded = pickle.load(file)
            return loaded
        except FileNotFoundError:
            return None

