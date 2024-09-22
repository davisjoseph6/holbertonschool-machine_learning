#!/usr/bin/env python3
"""
Convert Gensim Word2Vec model to Keras Embedding layer
"""

import numpy as np
from keras.layers import Embedding


def gensim_to_keras(model):
    """
    Converts a trained gensim Word2vec model to a Keras Embedding layer.
    """
    # Get the word vectors from the Gensim model
    weights = model.wv.vectors
    # Get the size of the vocabulary
    vocab_size = weights.shape[0]
    # Get the size of the embedding vectors
    embedding_dim = weights.shape[1]

    # Create a Keras Embedding layer
    embedding_layer = Embedding(input_dim=vocab_size,
                                output_dim=embedding_dim,
                                weights=[weights],
                                trainable=True)

    return embedding_layer
