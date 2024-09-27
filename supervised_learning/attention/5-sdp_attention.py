#!/usr/bin/env python3
"""
Calculates the scaled dot product attention.
"""

import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    Calculates the scaled dot product attention.
    """
    # Calculate the dot product between Q and the transpose of K
    matmul_qk = tf.matmul(Q, K, transpose_b=True)

    # Scale the dot product by the square root of dk
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # Apply the mask (if provided) by adding a large negative value to masked positions
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # Apply softmax to obtain attention weights
    weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # Multiply the weights with the value matrix V
    output = tf.matmul(weights, V)

    return output, weights
