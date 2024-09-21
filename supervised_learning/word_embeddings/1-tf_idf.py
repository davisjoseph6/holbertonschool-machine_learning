#!/usr/bin/env python3
"""
This module provides a function to create a TF-IDF embedding matrix
from a list of sentences.
"""

import numpy as np
import re
from math import log


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding matrix from the given sentences.
    """
    # Tokenize and clean sentences
    word_set = set()
    processed_sentences = []

    for sentence in sentences:
        # Remove non-alphabetic characters and convert to lowercase
        words = re.findall(r'\b[a-zA-Z]{2,}\b', sentence.lower())
        processed_sentences.append(words)
        if vocab is None:
            word_set.update(words)

    # Use vocab if provided, otherwise use all unique words
    if vocab is None:
        vocab = sorted(word_set)

    # Initialize the embedding matrix
    s = len(sentences)
    f = len(vocab)
    embeddings = np.zeros((s, f))

    # Compute term frequency (TF)
    def compute_tf(words, vocab):
        tf = np.zeros(len(vocab))
        word_count = len(words)
        if word_count == 0:
            return tf
        for word in words:
            if word in vocab:
                tf[vocab.index(word)] += 1
        return tf / word_count  # Normalize by total word count

    # Compute inverse document frequency (IDF)
    def compute_idf(sentences, vocab):
        idf = np.zeros(len(vocab))
        total_docs = len(sentences)
        for i, word in enumerate(vocab):
            count = sum(1 for sentence in sentences if word in sentence)
            if count > 0:
                idf[i] = log(total_docs / count)  # No smoothing factor here
            else:
                idf[i] = 0  # If a word is not in any sentence, IDF is 0
        return idf

    # Compute TF-IDF for each sentence
    idf = compute_idf(processed_sentences, vocab)
    for i, sentence in enumerate(processed_sentences):
        tf = compute_tf(sentence, vocab)
        embeddings[i] = tf * idf

    # Normalize rows of the embedding matrix
    row_sums = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = np.divide(embeddings, row_sums, where=row_sums != 0)

    return embeddings, vocab
