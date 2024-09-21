#!/usr/bin/env python3
"""
Word2Vec Model Creation and Training
"""
from gensim.models import Word2Vec


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Creates, builds, and trains a Word2Vec model.
    """
    # Determine training method: CBOW (if True) or Skip-gram (if False)
    sg = 0 if cbow else 1

    # Build and train the Word2Vec model
    model = Word2Vec(sentences=sentences,
                     vector_size=vector_size,
                     window=window,
                     min_count=min_count,
                     workers=workers,
                     sg=sg,
                     negative=negative,
                     seed=seed)

    # Train the model for the specified number of epochs
    model.train(sentences, total_examples=model.corpus_count, epochs=epochs)

    return model
