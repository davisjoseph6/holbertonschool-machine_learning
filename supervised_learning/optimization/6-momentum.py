#!/usr/bin/env python3
"""
Sets up the gradient descent with momentum optimization algorithm in
TensorFlow
"""

import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    Sets up the gradient descent with momentum optimization
    algorithm
    """
    optimizer = tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
    return optimizer
