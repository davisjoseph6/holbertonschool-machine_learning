#!/usr/bin/env python3
"""
Neural style transfer
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model


class NST:
    """
    Class that performs tasks for neural style transfer
    """

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Class constructor neural style transfer
        """

        if (not isinstance(style_image, np.ndarray)
                or style_image.shape[-1] != 3):
            raise TypeError("style_image must be a numpy.ndarray"
                            " with shape (h, w, 3)")
        else:
            self.style_image = self.scale_image(style_image)

        if (not isinstance(content_image, np.ndarray)
                or content_image.shape[-1] != 3):
            raise TypeError("content_image must be a numpy.ndarray"
                            " with shape (h, w, 3)")
        else:
            self.content_image = self.scale_image(content_image)

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        else:
            self.alpha = alpha

        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")
        else:
            self.beta = beta

        self.model = self.load_model()

    @staticmethod
    def scale_image(image):
        """
        rescales an image such that its pixel values are between 0 and 1
        and its largest side is 512 px
        """
        if not isinstance(image, np.ndarray) or image.shape[-1] != 3:
            raise (TypeError
                   ("image must be a numpy.ndarray with shape (h, w, 3)"))

        h, w, _ = image.shape

        if w > h:
            w_new = 512
            h_new = int((h * 512) / w)
        else:
            h_new = 512
            w_new = int((w * 512) / h)

        resized_image = tf.image.resize(image,
                                        size=[h_new, w_new],
                                        method='bicubic')

        # Normalize
        resized_image = resized_image / 255.0

        # Limit pixel values between 0 and 1
        resized_image = tf.clip_by_value(resized_image, 0, 1)

        tf_resize_image = tf.expand_dims(resized_image, 0)

        return tf_resize_image

    def load_model(self):
        """
        creates the model used to calculate cost
        """
        vgg = VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False

        outputs = [vgg.get_layer(name).output for name in self.style_layers]
        outputs.append(vgg.get_layer(self.content_layer).output)

        model = Model([vgg.input], outputs)

        return model
