#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

def cloud_in_the_corner_numpy(N, sigma, S):
    arr = np.random.randn(S * N).reshape(S, N)
    return arr * sigma + 0.8

def cloud_in_the_corner(C, sigma, S):
    return tf.convert_to_tensor(cloud_in_the_corner_numpy(C, sigma, S), dtype="float32")

