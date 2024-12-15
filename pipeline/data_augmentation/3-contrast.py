#!/usr/bin/env python3
"""
This module provides a function to randomly adjust the contrast of an image.
"""

import tensorflow as tf


def change_contrast(image, lower, upper):
    """
    Randomly adjusts the contrast of an image.
    """
    return tf.image.random_contrast(image, lower, upper)
