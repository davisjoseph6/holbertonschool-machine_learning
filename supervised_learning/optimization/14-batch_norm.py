#!/usr/bin/env python3
"""
Creates a batch normalization layer for a neural network in TensorFlow.
"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in TensorFlow.

    Args:
        prev (tensorflow.Tensor: The activated output of the previpus layer.
        n (init): The number of nodes in the layer to be created.
        activation: The activation function that should be used on the output
        of the layer.

    Returns:
        tensorflow.Tensor: A tensor of the activated output for the layers.
    Autor: davisjoseph6
    """
    # He et. al initialization
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    # Creating a new Dense layer
    dense_layer = tf.keras.layers.Dense(units=n,
                                        kernel_initializer=initializer)

    # Pass input through the layer
    Z = dense_layer(prev)

    # Create trainable parameters gamma & beta
    # NOTE initialized as trainable vectors of 1 and 0 respectively
    gamma = tf.Variable(initial_value=tf.ones([n]), name='gamma',
                        trainable=True)
    beta = tf.Variable(initial_value=tf.zeros([n]), name='beta',
                       trainable=True)

    # Given epsilon value
    epsilon = 1e-7

    mean, variance = tf.nn.moments(Z, axes=[0])

    # apply batch normalization to Z
    Z_batch_norm = tf.nn.batch_normalization(
            x=Z,
            mean=mean,
            variance=variance,
            offset=beta,
            scale=gamma,
            variance_epsilon=epsilon)

    return activation(Z_batch_norm)
