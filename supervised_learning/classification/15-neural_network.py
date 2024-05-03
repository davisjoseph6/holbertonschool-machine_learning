#!/usr/bin/env python3
"""
Defines a class NeuralNetwork that defines a neural network with one hidden
layer performing binary classification
"""

import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    """
    Class that defines a neural network with one hidden layer performing
    binary classification
    """
    def __init__(self, nx, nodes):
        """
        Constructor for NeuralNetwork.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """
        Getter for W1.
        """
        return self.__W1

    @property
    def b1(self):
        """
        Getter for b1.
        """
        return self.__b1

    @property
    def A1(self):
        """
        Getter for A1.
        """
        return self.__A1

    @property
    def W2(self):
        """
        Getter for W2.
        """
        return self.__W2

    @property
    def b2(self):
        """
        Getter for b2.
        """
        return self.__b2

    @property
    def A2(self):
        """
        Getter for A2.
        """
        return self.__A2

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network.
        """
        Z1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))  # Sigmoid activation function

        Z2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))  # Sigmoid activation function

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.
        """
        m = Y.shape[1]
        cost = -(1 / m) * (
                np.sum(
                    Y * np.log(A) +
                    (1 - Y) * np.log(1.0000001 - A)
                    )
                )
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions.
        """
        _, A2 = self.forward_prop(X)
        predictions = (A2 >= 0.5).astype(int)
        cost = self.cost(Y, A2)
        return predictions, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Performs one pass of gradient descent on the neural network
        """
        m = Y.shape[1]

        # Output layer gradients
        dZ2 = A2 - Y
        dW2 = np.dot(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        # Hidden layer gradients
        dA1 = np.dot(self.__W2.T, dZ2)
        dZ1 = dA1 * A1 * (1 - A1)
        dW1 = np.dot(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        # Update weights and biases
        self.__W1 -= alpha * dW1
        self.__b1 -= alpha * db1
        self.__W2 -= alpha * dW2
        self.__b2 -= alpha * db2

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """
        Trains the neural network using gradient descent for a number
        of iterations.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        for i in range(iterations + 1):
            A1, A2 = self.forward_prop(X)
            cost = self.cost(Y, A2)
            if i % step == 0 or i == iterations:
                if verbose:
                    print(f"Cost after {i} iterations: {cost}")
                if graph:
                    costs.append(cost)
            self.gradient_descent(X, Y, A1, A2, alpha)

        if graph:
            plt.plot(range(0, iterations + 1, step), costs, 'b')
            plt.xlabel("Iteration")
            plt.ylabel("Cost")
            plt.title("Training Cost")
            plt.show()
            
        return self.evaluate(X, Y)
