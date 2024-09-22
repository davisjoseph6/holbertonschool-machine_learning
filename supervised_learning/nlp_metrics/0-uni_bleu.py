#!/usr/bin/env python3
"""
This module contains a function that calculates the unigram BLEU score
for a given sentence compared to reference translations.
"""

import numpy as np


def uni_bleu(references, sentence):
    """
    Calculates the unigram BLEU score for a given sentence.
    """
    sentence_len = len(sentence)

    # Count the clipped count
    word_counts = {}
    for word in sentence:
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1

    max_counts = {}
    for ref in references:
        ref_counts = {}
        for word in ref:
            if word in ref_counts:
                ref_counts[word] += 1
            else:
                ref_counts[word] = 1
        for word in ref_counts:
            if word in max_counts:
                max_counts[word] = max(max_counts[word], ref_counts[word])
            else:
                max_counts[word] = ref_counts[word]

    clipped_count = 0
    for word in word_counts:
        clipped_count += min(word_counts[word], max_counts.get(word, 0))

    precision = clipped_count / sentence_len

    # Brevity penalty
    ref_lengths = [len(ref) for ref in references]
    closest_ref_len = min(
            ref_lengths,
            key=lambda ref_len: (abs(ref_len - sentence_len), ref_len)
            )

    if sentence_len > closest_ref_len:
        brevity_penalty = 1

    else:
        brevity_penalty = np.exp(1 - closest_ref_len / sentence_len)

    bleu_score = brevity_penalty * precision
    return bleu_score
