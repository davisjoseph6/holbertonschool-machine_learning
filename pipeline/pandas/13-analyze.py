#!/usr/bin/env python3
"""
Module to compute descriptive statistics for a pandas DataFrame.
"""


def analyze(df):
    """
    Computes descriptive statistics for all columns except the Timestamp
    column.
    """
    # Compute descriptive statistics for all columns except "Timestamp"
    stats = df.loc[:, df.columns != "Timestamp"].describe()
    return stats
