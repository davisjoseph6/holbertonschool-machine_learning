#!/usr/bin/env python3
"""
Module to load data from a file into a pandas DataFrame.
"""

import pandas as pd


def from_file(filename, delimiter):
    """
    Loads data from a file as a pandas DataFrame.
    """
    return pd.read_csv(filename, delimiter=delimiter)
