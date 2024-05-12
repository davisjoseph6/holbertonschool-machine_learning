#!/usr/bin/env python3
"""
Creates the training operation for the neural network
"""

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def create_train_op(loss, alpha):
    """
    Creates the training operation for the neural network.
    """
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
    train_op = optimizer.minimize(loss)
    return train_op
