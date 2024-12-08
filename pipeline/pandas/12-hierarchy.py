#!/usr/bin/env python3
"""
Module to concatenate two pandas DataFrames and rearrange MultiIndex.
"""

import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    Rearranges the MultiIndex and concatenates DataFrames with
    specific conditions.
    """
    # Set Timestamp as index for both DataFrames
    df1 = index(df1)
    df2 = index(df2)

    # Filter both DataFrames by Timestamp range [1417411980, 1417417980]
    start, end = 1417411980, 1417417980
    df1_filtered = df1[(df1.index >= start) & (df1.index <= end)]
    df2_filtered = df2[(df2.index >= start) & (df2.index <= end)]

    # Concatenate df2_filtered and df1_filtered with keys for labeling
    concatenated = pd.concat([df2_filtered, df1_filtered], keys=["bitstamp",
                                                                 "coinbase"])

    # Swap the levels of the MultiIndex to make Timestamp the first level
    concatenated = concatenated.swaplevel(0, 1)

    # Sort by the new MultiIndex (Timestamp as the primary level)
    concatenated = concatenated.sort_index(level=0)

    return concatenated
