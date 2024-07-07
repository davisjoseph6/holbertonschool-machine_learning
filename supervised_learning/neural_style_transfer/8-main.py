#!/usr/bin/env python3

import matplotlib.image as mpimg
import os
import random
import numpy as np
import tensorflow as tf

NST = __import__('8-neural_style').NST


if __name__ == '__main__':
    style_image = mpimg.imread("starry_night.jpg")
    content_image = mpimg.imread("golden_gate.jpg")

    # Reproducibility
    SEED = 0
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['TF_ENABLE_ONEDNN_OPTS']= '0'
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    nst = NST(style_image, content_image)
    
    print("Scaled content image shape:", nst.content_image.shape)
    
    # Initialize generated_image with the same shape as content_image
    generated_image = tf.Variable(nst.content_image, dtype=tf.float32)
    print("Generated image shape:", generated_image.shape)
    
    grads, J_total, J_content, J_style = nst.compute_grads(generated_image)
    print(J_total)
    print(J_content)
    print(J_style)
    print(grads)

