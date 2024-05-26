#!/usr/bin/env python3
"""
Creates a layer of a neural netowrk using dropout.
"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Creates a layer of a neural network using dropout.
    """
    # Create a dense layer with n units and the specified activation function
    dense_layer = tf.keras.layers.Dense(units=n, activation=activation)(prev)

    # Apply dropout to the dense layer
    dropout_layer = tf.keras.layers.Dropout(rate=1-keep_prob)(dense_layer, training=training)

    return dropout_layer
