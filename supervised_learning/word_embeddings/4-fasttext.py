#!/usr/bin/env python3
"""
FastText model training
"""

import gensim


def fasttext_model(sentences, vector_size=100, min_count=5, negative=5,
                   window=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Creates, builds, and trains a Gensim model.
    """
    # Determine training algorithm: CBOW (if True) or Skip-gram (if False)
    sg = 0 if cbow else 1

    # Build and train the FastText model
    model = gensim.models.FastText(sentences=sentences,
                                   vector_size=vector_size,
                                   window=window,
                                   min_count=min_count,
                                   negative=negative,
                                   sg=sg,
                                   epochs=epochs,
                                   seed=seed,
                                   workers=workers)

    # Prepare the model's vocabulary and train it
    model.build_vocab(sentences)
    model.train(sentences,
                total_examples=model.corpus_count,
                epochs=model.epochs)

    return model
