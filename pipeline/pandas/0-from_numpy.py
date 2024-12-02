#!/usr/bin/env python3
import pandas as pd

def from_numpy(array):
    """Creates a pandas DataFrame from a np.ndarray with capitalized alphabetical columns.

    Args:
        array (np.ndarray): The numpy array to convert into a DataFrame.

    Returns:
        pd.DataFrame: The resulting DataFrame with columns labeled A-Z.
    """
    import numpy as np
    num_cols = array.shape[1]
    columns = [chr(ord('A') + i) for i in range(num_cols)]
    df = pd.DataFrame(array, columns=columns)
    return df

