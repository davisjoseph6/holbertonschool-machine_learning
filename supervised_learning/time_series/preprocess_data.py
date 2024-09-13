#!/usr/bin/env python3
"""Preprocess BTC data for RNN model"""

import pandas as pd
import numpy as np

def load_and_preprocess(file_path):
    """
    Loads and preprocesses raw BTC data.
    """
    # Load data
    data = pd.read_csv(file_path)

    # Drop rows with missing values
    data.dropna(inplace=True)

    # Select relevant features
    features = ['timestamp', 'close']
    data = data[features]

    # Convert timestamp to datetime for resampling
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')

    # Resample to hourly intervals
    data = data.set_index('timestamp').resample('H').agg({'close': 'last'}).dropna()

    # Normalize the 'close' column
    data['close'] = (data['close'] - data['close'].min()) / (data['close'].max() - data['close'].min())

    # Create rolling windows
    sequence_length = 24  # 24 hours of data
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data['close'].iloc[i:i + sequence_length].values)
        y.append(data['close'].iloc[i + sequence_length])

    # Save processed data
    np.save('X.npy', np.array(X))
    np.save('y.npy', np.array(y))

    return data

if __name__ == "__main__":
    # Example file path
    file_path = "coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv"
    load_and_preprocess(file_path)
