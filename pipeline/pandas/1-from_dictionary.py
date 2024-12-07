#!/usr/bin/env python3
"""
Module to create a pandas DataFrame from a dictionary
"""

import pandas as pd

# Creating the DataFrame from a dictionary
data = {
        "First": [0.0, 0.5, 1.0, 1.5],
        "Second": ["one", "two", "three", "four"]
        }
df = pd.DataFrame(data, index=["A", "B", "C", "D"])
