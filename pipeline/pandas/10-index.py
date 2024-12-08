#!/usr/bin/env python3
"""
Module to set the Timestamp column as the index of a pandas DataFrame.
"""


def index(df):
    """
    Sets the Timestamp column as the index of the DataFrame.
    """
    # Set the Timestamp column as the index
    df.set_index("Timestamp", inplace=True)
    return df
