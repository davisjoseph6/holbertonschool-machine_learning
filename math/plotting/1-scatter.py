#!/usr/bin/env python3
"""
Code to plot x -> y as a scatter plot
"""
import numpy as np
import matplotlib.pyplot as plt


def scatter():
    """
    scatter plot
    """

    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x, y = np.random.multivariate_normal(mean, cov, 2000).T
    y += 180
    plt.figure(figsize=(6.4, 4.8))

    # Plot data as magenta points
    plt.scatter(x, y, color='magenta')

    # Label x-axis and y-axis
    plt.xlabel('Height (in)')
    plt.ylabel('Weight (lbs)')

    # Set plot title
    plt.title("Men's Height vs Weight")

    # Display the plot
    plt.show()
