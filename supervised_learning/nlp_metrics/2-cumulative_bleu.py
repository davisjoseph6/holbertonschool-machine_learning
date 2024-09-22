#!/usr/bin/env python3
"""
This module contains a function that calculates the cumulative n-gram BLEU
score for a given sentence compared to reference translations.
"""

import numpy as np


def cumulative_bleu(references, sentence, n):
    """
    Calculates the cumulative n-gram BLEU score for a given sentence.
    """
    def get_ngrams(sequence, n):
        """Helper function to generate n-grams from a given sequence."""
        return [
            ' '.join(sequence[i:i + n]) for i in range(len(sequence) - n + 1)
            ]

    precisions = []

    for i in range(1, n + 1):
        sentence_ngrams = get_ngrams(sentence, i)
        sentence_len = len(sentence_ngrams)

        # Count the n-grams in the sentence
        sentence_counts = {}
        for ngram in sentence_ngrams:
            if ngram in sentence_counts:
                sentence_counts[ngram] += 1
            else:
                sentence_counts[ngram] = 1

        max_counts = {}
        for ref in references:
            ref_ngrams = get_ngrams(ref, i)
            ref_counts = {}
            for ngram in ref_ngrams:
                if ngram in ref_counts:
                    ref_counts[ngram] += 1
                else:
                    ref_counts[ngram] = 1
            for ngram in ref_counts:
                if ngram in max_counts:
                    max_counts[ngram] = max(
                            max_counts[ngram], ref_counts[ngram]
                            )
                else:
                    max_counts[ngram] = ref_counts[ngram]

        clipped_count = 0
        for ngram in sentence_counts:
            clipped_count += min(
                    sentence_counts[ngram], max_counts.get(ngram, 0)
                    )

        precision = clipped_count / sentence_len if sentence_len > 0 else 0
        precisions.append(precision)

    # Calculate the geometric mean of the precisions
    geometric_mean = (
            np.exp(np.mean(np.log(precisions))) if all(p > 0 for p in precisions)
            else 0
            )

    # Brevity penalty
    ref_lengths = [len(ref) for ref in references]
    closest_ref_len = min(
            ref_lengths,
            key=lambda ref_len: (abs(ref_len - len(sentence)), ref_len)
            )

    if len(sentence) > closest_ref_len:
        brevity_penalty = 1
    else:
        brevity_penalty = np.exp(1 - closest_ref_len / len(sentence))

    cumulative_bleu_score = brevity_penalty * geometric_mean
    return cumulative_bleu_score
