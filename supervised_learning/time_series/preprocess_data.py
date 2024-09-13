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

    # Print the columns to inspect
    print("Columns in the dataset:", data.columns)

    # Drop rows with missing values
    data.dropna(inplace=True)

    # Select relevant features
    features = ['Timestamp', 'Close']
    data = data[features]

    # Convert 'Timestamp' to datetime for resampling
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], unit='s')

    # Resample to hourly intervals
    data = data.set_index('Timestamp').resample('h').agg({'Close': 'last'}).dropna()

    # Normalize the 'Close' column
    data['Close'] = (data['Close'] - data['Close'].min()) / (data['Close'].max() - data['Close'].min())

    # Create rolling windows
    sequence_length = 24  # 24 hours of data
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data['Close'].iloc[i:i + sequence_length].values)
        y.append(data['Close'].iloc[i + sequence_length])

    # Save processed data
    np.save('X.npy', np.array(X))
    np.save('y.npy', np.array(y))

    return data

if __name__ == "__main__":
    # Example file path
    file_path = "coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv"
    load_and_preprocess(file_path)
