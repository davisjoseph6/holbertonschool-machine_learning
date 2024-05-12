#!/usr/bin/env python3
"""
Creating placeholders for x, y.
"""


import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def create_placeholders(nx, classes):
    """
    Creates two placeholders, x and y, for the neural network
    """
    x = tf.placeholder(dtype=tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(dtype=tf.float32, shape=[None, classes], name='y')

    return x, y
