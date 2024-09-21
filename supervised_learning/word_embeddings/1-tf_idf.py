#!/usr/bin/env python3
"""
    TF-IDF
"""

import re
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding.
    """
    if not isinstance(sentences, list):
        raise TypeError("sentences should be a list.")

    # Preprocess sentences by making them lowercase and removing "'s" suffix
    preprocessed_sentences = [
            re.sub(r"\b(\w+)'s\b", r"\1", sentence.lower()) for sentence in sentences
            ]

    # If vocab is None, generate vocabulary from the sentences
    if vocab is None:
        list_words = []
        for sentence in preprocessed_sentences:
            words = re.findall(r'\w+', sentence)
            list_words.extend(words)
        vocab = sorted(set(list_words))

    # Create a TF-IDF vectorizer with the given vocabulary
    tfidf_vect = TfidfVectorizer(vocabulary=vocab)

    # Fit and transform the sentences to produce the TF-IDF matrix
    tfidf_matrix = tfidf_vect.fit_transform(sentences)

    # Extract the features (vocabulary used)
    features = tfidf_vect.get_feature_names_out()

    return tfidf_matrix.toarray(), features
