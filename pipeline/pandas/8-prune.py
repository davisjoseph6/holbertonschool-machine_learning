#!/usr/bin/env python3
"""
Module to remove rows with NaN values in the Close column from a pandas
DataFrame.
"""


def prune(df):
    """
    Removes rows where the Close column has NaN values.
    """
    return df[df["Close"].notna()]
