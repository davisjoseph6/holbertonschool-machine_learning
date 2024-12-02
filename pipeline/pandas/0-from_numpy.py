#!/usr/bin/env python3
"""
This module contains the function `from_numpy`
that converts a numpy array into a pandas DataFrame.
"""

import pandas as pd
import numpy as np

def from_numpy(array):
    """
    Creates a pandas DataFrame from a numpy ndarray.

    Parameters:
    - array (np.ndarray): The numpy array to convert.

    Returns:
    - pd.DataFrame: The resulting DataFrame with columns labeled
      in alphabetical order and capitalized.
    """
    # Generate column labels (A, B, C, ...)
    num_columns = array.shape[1]
    column_labels = [chr(65 + i) for i in range(num_columns)]  # 65 is ASCII for 'A'

    # Create and return the DataFrame
    return pd.DataFrame(array, columns=column_labels)

