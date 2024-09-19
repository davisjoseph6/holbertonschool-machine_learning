#!/usr/bin/env python3
"""
This module provides a funtion to create a bag-of-words embeding matrix
from a list of sentences.
"""

import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix.
    """
    word_set = set()
    processed_sentences = []

    for sentence in sentences:
        # Remove non-alphabetic characters and convert to lowercase
        words = re.findall(r'\b\w+\b', sentence.lower())
        processed_sentences.append(words)
        if vocab is None:
            word_set.update(words)

    # Use vocab if provided, otherwise use all unique words
    if vocab is None:
        vocab = sorted(word_set)

    # Initialize the embedding matrix
    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)

    # Create embeddings
    for i, sentence in enumerate(processed_sentences):
        for word in sentence:
            if word in vocab:
                embeddings[i, vocab.index(word)] += 1

    return embeddings, vocab
