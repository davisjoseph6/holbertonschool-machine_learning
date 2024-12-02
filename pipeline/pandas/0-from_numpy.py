#!/usr/bin/env python3
"""
Module to create a pandas DataFrame from a numpy ndarray.
"""

import pandas as pd

def from_numpy(array):
    """
    Creates a pandas DataFrame from a numpy ndarray.
    
    Args:
        array (np.ndarray): The input numpy array.
    
    Returns:
        pd.DataFrame: The resulting pandas DataFrame with alphabetical column labels.
    """
    columns = [chr(65 + i) for i in range(array.shape[1])]
    return pd.DataFrame(array, columns=columns)

