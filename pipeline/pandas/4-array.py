#!/usr/bin/env python3
"""
Module to extract data from a pandas DataFrame and convert it to a numpy array.
"""

import pandas as pd
import numpy as np


def array(df):
    """
    Selects the last 10 rows of the High and Close columns from a DataFrame
    and converts them into a numpy.ndarray.
    """
    # Select the last 10 rows of the High and Close columns
    selected_data = df[["High", "Close"]].tail(10)
    # Convert the selected data to a numpy array
    return selected_data.to_numpy()
