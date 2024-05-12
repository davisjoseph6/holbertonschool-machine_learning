#!/usr/bin/env python3
"""
Creates the forward propagation graph for the neural network.
"""

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

# Importing the create_layer function from another script file
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    creates the forward propagation graph for the neural network
    """
    current_output = x
    for i, (size, activation) in enumerate(zip(layer_sizes, activations)):
        current_output = create_layer(current_output, size, activation)

    return current_output
