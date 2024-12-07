#!/usr/bin/env python3
"""
Module to rename and modify a pandas DataFrame.
"""

import pandas as pd


def rename(df):
    """
    Renames the Timestamp column to Datetime, converts timestamp values to datetime,
    and displays only the Datetime and Close columns.
    """
    # Rename the column
    df.rename(columns={"Timestamp": "Datetime"}, inplace=True)
    # Convert timestamp values to datetime
    df["Datetime"] = pd.to_datetime(df["Datetime"], unit="s")
    # Select only the specified columns
    return df[["Datetime", "Close"]]
