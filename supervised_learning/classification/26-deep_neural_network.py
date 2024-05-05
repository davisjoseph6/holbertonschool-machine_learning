#!/usr/bin/env python3
"""
This script defines a Deep Neural Network 4 binary classification.
"""

import matplotlib.pyplot as plt
import numpy as np


class DeepNeuralNetwork:
    """
    Define a deep neural network that does binary classification.
    """
    def __init__(self, nx, layers):
        """
        Initialize a deep neural network with given number of input features
        and layers.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")

        # Check if all layers are positive integers
        if not all(map(lambda x: isinstance(x, int) and x > 0, layers)):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)  # number of layers
        self.__cache = {}  # to store all intermediary values of the network
        self.__weights = {}  # to hold all weights and biases of the network

        # Initialize weights and biases using He et al. method 4 each layer
        for layer_index in range(1, self.__L + 1):
            layer_size = layers[layer_index - 1]
            prev_layer_size = nx if layer_index == 1 else layers[
                    layer_index - 2
                    ]
            self.__weights[f'W{layer_index}'] = (
                    np.random.randn(layer_size, prev_layer_size) * np.sqrt(
                        2 / prev_layer_size
                        )
                    )
            self.__weights[f'b{layer_index}'] = np.zeros((layer_size, 1))

    @property
    def L(self):
        """
        Getter 4 number of layers.
        """
        return self.__L

    @property
    def cache(self):
        """
        Getter 4 cache.
        """
        return self.__cache

    @property
    def weights(self):
        """
        Getter 4 weights.
        """
        return self.__weights

    def sigmoid(self, Z):
        """
        Sigmoid activation function.
        """
        return 1 / (1 + np.exp(-Z))

    def forward_prop(self, X):
        """
        Perform 4ward propagation for the neural network.
        """
        self.__cache['A0'] = X
        for layer_index in range(1, self.__L + 1):
            W = self.__weights[f'W{layer_index}']
            b = self.__weights[f'b{layer_index}']
            A_prev = self.__cache[f'A{layer_index-1}']
            Z = np.dot(W, A_prev) + b
            self.__cache[f'A{layer_index}'] = self.sigmoid(Z)

        AL = self.__cache[f'A{self.__L}']
        return AL, self.__cache

    def cost(self, Y, A):
        """
        Calculate the cost using the logistic regression 'cross-entropy'
        cost function.
        """
        m = Y.shape[1]  # number of examples
        cost = -(1 / m) * np.sum(Y * np.log(A) + (
            1 - Y
            ) * np.log(1.0000001 - A)
            )
        return cost

    def evaluate(self, X, Y):
        """
        Evaluate the neural network's predictions against the true labels.
        """
        # Per4m 4ward propagation to get outputs
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)  # Compute the cost with the actual labels
        # Convert probabilities to binary output
        predictions = (A >= 0.5).astype(int)
        return predictions, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Per4m one pass of gradient descent on the neural network.
        """
        m = Y.shape[1]  # Number of examples
        L = self.__L  # Number of layers

        A = cache[f'A{L}']  # Output of the last layer
        # Derivative of cost with respect to A
        dA = - (np.divide(Y, A) - np.divide(1 - Y, 1 - A))

        for layer_index in reversed(range(1, L + 1)):
            A_prev = cache[f'A{layer_index-1}']
            A_curr = cache[f'A{layer_index}']
            W = self.__weights[f'W{layer_index}']

            # Element-wise product assumes sigmoid activation
            dZ = dA * A_curr * (1 - A_curr)
            dW = np.dot(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            if layer_index > 1:
                dA = np.dot(W.T, dZ)  # Prepare dA 4 the next layer

            # Update weights and biases
            self.__weights[f'W{layer_index}'] -= alpha * dW
            self.__weights[f'b{layer_index}'] -= alpha * db

        return self.__weights

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
            Method to train deep neural network

            :param X: ndarray, shape(nx,m), input data
            :param Y: ndarray, shapte(1,m), correct labels
            :param iterations: number of iterations to train over
            :param alpha: learning rate
            :param verbose: boolean print or not information
            :param graph: boolean print or not graph
            :param step: int

            :return: evaluation of training after iterations
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
        if verbose is True or graph is True:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        # list to store cost /iter
        costs = []
        count = []

        for i in range(iterations + 1):
            # run forward propagation
            A, cache = self.forward_prop(X)

            # run gradient descent for all iterations except the last one
            if i != iterations:
                self.gradient_descent(Y, self.cache, alpha)

            cost = self.cost(Y, A)

            # store cost for graph
            costs.append(cost)
            count.append(i)

            # verbose TRUE, every step + first and last iteration
            if verbose and (i % step == 0 or i == 0 or i == iterations):
                # run evaluate
                print("Cost after {} iterations: {}".format(i, cost))

        # graph TRUE after training complete
        if graph:
            plt.plot(count, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """
        Saves the instance object to a file in pickle format.
        filename is the file to which the object will be saved.
        If filename does not have the extension .pkl, adds it.
        """
        if not filename.endswith(".pkl"):
            filename += ".pkl"

        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        """
        Loads a pickled DeepNeuralNetwork object.
        filename is the file from which the object should be loaded.
        Returns: the loaded object, or None if filename doesn't exist.
        """

        try:
            with open(filename, "rb") as file:
                unpickled_obj = pickle.load(file)
            return unpickled_obj

        except FileNotFoundError:
            return None
