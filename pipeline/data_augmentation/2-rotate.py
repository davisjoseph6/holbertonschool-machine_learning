#!/usr/bin/env python3
"""
This module provides a function to rotate an image by 90 degrees
counter-clockwise.
"""

import tensorflow as tf


def rotate_image(image):
    """
    Rotates an image by 90 degrees counter-clockwise.
    """
    return tf.image.rot90(image)
