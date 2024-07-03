#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

class NST:
    # Define the style and content layers as class attributes
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        # Validate inputs
        if not isinstance(style_image, np.ndarray) or style_image.shape[-1] != 3:
            raise TypeError("style_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(content_image, np.ndarray) or content_image.shape[-1] != 3:
            raise TypeError("content_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        # Preprocess the images
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        # Validate the image input
        if not isinstance(image, np.ndarray) or image.shape[-1] != 3:
            raise TypeError("image must be a numpy.ndarray with shape (h, w, 3)")

        # Convert the image to a TensorFlow tensor and add a batch dimension
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.expand_dims(image, axis=0)

        # Get the height and width of the image
        h, w, _ = image.shape[1:]

        # Calculate the new height and width to maintain aspect ratio
        if h > w:
            new_h = 512
            new_w = int(w * 512 / h)
        else:
            new_w = 512
            new_h = int(h * 512 / w)

        # Resize the image using bicubic interpolation
        resized_image = tf.image.resize(image, (new_h, new_w), method='bicubic')

        # Rescale pixel values from [0, 255] to [0, 1]
        scaled_image = tf.clip_by_value(resized_image, 0.0, 1.0)

        return scaled_image
