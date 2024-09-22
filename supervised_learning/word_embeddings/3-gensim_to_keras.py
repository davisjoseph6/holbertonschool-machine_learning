#!/usr/bin/env python3
"""
Convert Gensim Word2Vec model to a Keras Embedding layer.
"""

import tensorflow as tf


def gensim_to_keras(model):
    """
    Converts a trained gensim Word2vec model to a Keras Embedding layer.
    """
    weights = model.wv.vectors  # Get the weights from the Gensim model
    embedding_layer = tf.keras.layers.Embedding(
            input_dim=weights.shape[0],
            output_dim=weights.shape[1],
            weights=[weights],
            trainable=True  # Set to True to allow further training
            )
    return embedding_layer
