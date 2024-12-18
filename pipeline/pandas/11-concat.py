#!/usr/bin/env python3
"""
Module to concatenate two pandas DataFrames with specific conditions.
"""

import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """
    Concatenates two DataFrames with indexing and specific conditions.

    Args:
        df1 (pd.DataFrame): The first DataFrame (coinbase).
        df2 (pd.DataFrame): The second DataFrame (bitstamp).

    Returns:
        pd.DataFrame: The concatenated DataFrame with labeled keys.
    """
    # Set Timestamp as index for both DataFrames
    df1 = index(df1)
    df2 = index(df2)

    # Filter df2 to include rows with Timestamp <= 1417411920
    df2_filtered = df2[df2.index <= 1417411920]

    # Concatenate df2_filtered above df1 with keys for labeling
    concatenated = pd.concat([df2_filtered, df1], keys=["bitstamp", "coinbase"]
                             )

    return concatenated
