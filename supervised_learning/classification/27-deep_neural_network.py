#!/usr/bin/env python3
"""
This script defines a Deep Neural Network 4 binary classification.
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt

class DeepNeuralNetwork:
    def __init__(self, nx, layers):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")
        if any(isinstance(layer, int) and layer <= 0 for layer in layers):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i in range(self.__L):
            layer_size = layers[i]
            weight_key = 'W' + str(i + 1)
            bias_key = 'b' + str(i + 1)
            prev_layer_size = nx if i == 0 else layers[i - 1]
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

    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return expZ / expZ.sum(axis=0, keepdims=True)

    def forward_prop(self, X):
        self.__cache['A0'] = X
        for i in range(1, self.__L + 1):
            W = self.__weights['W' + str(i)]
            b = self.__weights['b' + str(i)]
            A_prev = self.__cache['A' + str(i - 1)]
            Z = np.dot(W, A_prev) + b
            if i == self.__L:  # Apply softmax on the output layer
                self.__cache['A' + str(i)] = self.softmax(Z)
            else:
                self.__cache['A' + str(i)] = 1 / (1 + np.exp(-Z))  # Sigmoid for hidden layers
        return self.__cache['A' + str(self.__L)], self.__cache

    def cost(self, Y, A):
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A + 1e-8)) / m  # Avoid log(0) with epsilon
        return cost

    def evaluate(self, X, Y):
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        predictions = np.argmax(A, axis=0)
        labels = np.argmax(Y, axis=0)
        accuracy = np.mean(predictions == labels)
        return predictions, cost, accuracy

    def gradient_descent(self, Y, cache, alpha=0.05):
        m = Y.shape[1]
        A = cache['A' + str(self.__L)]
        dZ = A - Y
        for i in reversed(range(1, self.__L + 1)):
            A_prev = cache['A' + str(i - 1)]
            W = self.__weights['W' + str(i)]
            dW = np.dot(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            if i > 1:
                dZ = np.dot(W.T, dZ) * (A_prev * (1 - A_prev))
            self.__weights['W' + str(i)] -= alpha * dW
            self.__weights['b' + str(i)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
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

        cost_list = []
        for i in range(iterations):
            A, _ = self.forward_prop(X)
            self.gradient_descent(Y, self.__cache, alpha)
            cost = self.cost(Y, A)
            if verbose and (i % step == 0 or i == iterations - 1):
                print(f"Cost after {i} iterations: {cost}")
            if graph:
                cost_list.append(cost)
        if graph:
            plt.plot(range(0, iterations, step), cost_list)
            plt.xlabel('Iterations')
            plt.ylabel('Cost')
            plt.title('Training Cost')
            plt.show()
        return self.evaluate(X, Y)

