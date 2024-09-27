#!/usr/bin/env python3
"""
Calculates the positional encoding for a transformer.
"""

import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Calculates the positional encoding for a transformer.
    """
    # Initialize the positional encoding matrix with zeros
    position = np.arange(max_seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, dm, 2) * -(np.log(10000.0) / dm))

    # Calculate the positional encoding using sine and cosine functions
    pos_encoding = np.zeros((max_seq_len, dm))
    pos_encoding[:, 0::2] = np.sin(position * div_term)
    pos_encoding[:, 1::2] = np.cos(position * div_term)

    return pos_encoding
