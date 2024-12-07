#!/usr/bin/env python3
"""
Module to sort a DataFrame in reverse chronological order and transpose it.
"""


def flip_switch(df):
    """
    Sorts the DataFrame in reverse chronological order and transposes it.
    """
    # Sort the DataFrame in reverse chronological order by index
    sorted_df = df.sort_index(ascending=False)
    # Transpose the sorted DataFrame
    return sorted_df.transpose()
