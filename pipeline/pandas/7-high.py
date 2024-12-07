#!/usr/bin/env python3
"""
Module to sort a pandas DataFrame by the High column in descending order.
"""


def high(df):
    """
    Sorts the DataFrame by the High column in descending order.
    """
    return df.sort_values(by="High", ascending=False)
