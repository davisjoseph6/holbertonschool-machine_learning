#!/usr/bin/env python3
"""
Module to clean and fill missing values in a pandas DataFrame.
"""


def fill(df):
    """
    Cleans and fills missing values in a DataFrame.
    """
    # Remove the Weighted_Price column
    df = df.drop(columns=["Weighted_Price"])

    # Fill missing values in the Close column with the previous row's value
    df["Close"] = df["Close"].fillna(method="ffill")

    # Fill missing values in High, Low, and Open columns with
    # the corresponding Close value
    for col in ["High", "Low", "Open"]:
        df[col] = df[col].fillna(df["Close"])

    # Set missing values in Volume_(BTC) and Volume_(Currency) to 0
    df["Volume_(BTC)"] = df["Volume_(BTC)"].fillna(0)
    df["Volume_(Currency)"] = df["Volume_(Currency)"].fillna(0)

    return df
