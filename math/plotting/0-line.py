#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
"""
Code to plot y as a line graph
"""


def line():
    """
    code to plot y
    """

    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    # Plot y as a solid red line
    plt.plot(y, 'r-')  # 'r-' is the style option for a solid red line

    # Set the limits of th x-axis
    plt.xlim(0, 10)

    # Show the plot
    plt.show()
