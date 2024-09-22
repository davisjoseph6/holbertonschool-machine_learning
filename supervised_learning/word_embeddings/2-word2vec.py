#!/usr/bin/env python3
"""
    Train Word2Vec
"""
from gensim.models import Word2Vec


def word2vec_model(sentences, size=100, min_count=5, window=5,
                   negative=5, cbow=True, iterations=5, seed=0, workers=1):
    """
        creates and trains a gensim word2vec model

    :param sentences: list of sentences to be trained on
    :param size: dimensionality of the embedding layer
    :param min_count: minimum number of occurrences of a word
        for use in training
    :param window: maximum distance between the current and predicted
    word within a sentence
    :param negative: size of negative sampling
    :param cbow: boolean to determine training type: True=CBOW, False=Skip-gram...........
    :param iterations: number of iterations to train over
    :param seed: seed for the random number generator
    :param workers: number of worker threads to train the model

    :return: trained model
    """
    if cbow is True:
        sg = 0
    else:
        sg = 1
    model = Word2Vec(sentences=sentences,
                     size=size,
                     window=window,
                     min_count=min_count,
                     negative=negative,
                     seed=seed,
                     workers=workers,
                     iter=iterations,
                     sg=cbow)

    return model
