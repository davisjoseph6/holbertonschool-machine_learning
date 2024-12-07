#!/usr/bin/env python3
"""
Module to slice specific columns and rows from a pandas DataFrame.
"""


def slice(df):
    """
    Extracts specific columns and selects every 60th row from a DataFrame.
    """
    # Extract the specified columns
    sliced_df = df[["High", "Low", "Close", "Volume_(BTC)"]]
    # Select every 60th row
    return sliced_df.iloc[::60]
