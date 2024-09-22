#!/usr/bin/env python3
"""
Word2Vec Model Creation and Training
"""
import gensim


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Creates, builds, and trains a Word2Vec model.
    """

    # Build and train the Word2Vec model
    model = gensim.models.Word2Vec(sentences, vector_size=vector_size,
                     min_count=min_count, window=window,
                     negative=negative, sg=0 if cbow else 1,
                     seed=seed, workers=workers)

    # Train the model for the specified number of epochs
    model.train(sentences, total_examples=model.corpus_count, epochs=epochs)

    return model
