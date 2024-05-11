#!/usr/bin/env python3

"""
This is the DeepNeuralNetwork class module.
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt

class DeepNeuralNetwork:
    """
    Represents a deep neural network to perform classification.
    """

    def __init__(self, nx, layers, activation='sig'):
        """
        Initialize the deep neural network object.

        Args:
            nx (int): The number of input features.
            layers (list): A list of positive integers representing the
                number of nodes in each layer.
            activation (str): Type of activation function ('sig' for sigmoid, 'tanh' for tanh).

        Raises:
            TypeError: If nx is not an integer or layers is not a list of
                positive integers.
            ValueError: If nx is not a positive integer or activation is not 'sig' or 'tanh'.

        Attributes:
            L (int): The number of layers in the neural network.
            cache (dict): A dictionary to hold intermediate values.
            weights (dict): A dictionary to hold the weights and biases.
            activation (str): The activation function used in the hidden layers.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0 or not all(isinstance(x, int) and x > 0 for x in layers):
            raise TypeError("layers must be a list of positive integers")
        if activation not in ('sig', 'tanh'):
            raise ValueError("activation must be 'sig' or 'tanh'")
        
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation

        for i in range(self.__L):
            layer_size = layers[i]
            prev_layer_size = nx if i == 0 else layers[i-1]
            weight_key = f"W{i+1}"
            bias_key = f"b{i+1}"
            self.__weights[weight_key] = np.random.randn(layer_size, prev_layer_size) * np.sqrt(2 / prev_layer_size)
            self.__weights[bias_key] = np.zeros((layer_size, 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    @property
    def activation(self):
        return self.__activation

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network.
        """
        self.__cache["A0"] = X
        for i in range(1, self.__L + 1):
            prev_A = self.__cache[f"A{i-1}"]
            Z = np.matmul(self.__weights[f"W{i}"], prev_A) + self.__weights[f"b{i}"]
            if i < self.__L:
                if self.__activation == 'sig':
                    self.__cache[f"A{i}"] = 1 / (1 + np.exp(-Z))
                elif self.__activation == 'tanh':
                    self.__cache[f"A{i}"] = np.tanh(Z)
            else:
                exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
                self.__cache[f"A{i}"] = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
        return self.__cache[f"A{self.__L}"], self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.
        """
        m = Y.shape[1]
        return -(1 / m) * np.sum(Y * np.log(A + 1e-8))

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions.
        """
        A, _ = self.forward_prop(X)
        prediction = np.eye(Y.shape[0])[np.argmax(A, axis=0)].T
        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Performs one pass of gradient descent on the network.
        """
        m = Y.shape[1]
        weights_copy = self.__weights.copy()
        for i in reversed(range(1, self.__L + 1)):
            A = cache[f"A{i}"]
            A_prev = cache[f"A{i-1}"]
            if i == self.__L:
                dZ = A - Y
            else:
                if self.__activation == 'sig':
                    dZ = np.dot(weights_copy[f"W{i+1}"].T, dZ) * A * (1 - A)
                elif self.__activation == 'tanh':
                    dZ = np.dot(weights_copy[f"W{i+1}"].T, dZ) * (1 - np.square(A))
            dW = np.dot(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            self.__weights[f"W{i}"] -= alpha * dW
            self.__weights[f"b{i}"] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """
        Trains the deep neural network.
        """
        if not isinstance(iterations, int) or iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float) or alpha <= 0:
            raise ValueError("alpha must be a positive float")

        costs = []
        for i in range(iterations):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
            cost = self.cost(Y, A)
            costs.append(cost)
            if verbose and i % step == 0:
                print(f"Cost after {i} iterations: {cost}")

        if graph:
            plt.plot(range(iterations), costs)
            plt.xlabel('Iterations')
            plt.ylabel('Cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """
        Saves the model to a file in pickle format.
        """
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        """
        Loads a model from a pickle file.
        """
        try:
            with open(filename, "rb") as file:
                return pickle.load(file)
        except FileNotFoundError:
            return None

