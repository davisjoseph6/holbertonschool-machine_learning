#!/usr/bin/env python3
"""
code to plot x -> y1 and x -> y2 as line graphs
"""
import numpy as np
import matplotlib.pyplot as plt


def two():
    """
    code to plot x -> y1 and x -> as line graphs
    """

    x = np.arange(0, 21000, 1000)
    r = np.log(0.5)
    t1 = 5730  # Half-life of C-14
    t2 = 1600  # Half-life of Ra-226
    y1 = np.exp((r / t1) * x)
    y2 = np.exp((r / t2) * x)
    plt.figure(figsize=(6.4, 4.8))

    # Plotting x against y1 and y2
    plt.plot(x, y1, 'r--', label='C-14')  # Dashed red line for C-14
    plt.plot(x, y2, 'g-', label='Ra-226')  # Solid green line for Ra-226

    # Setting labels for axes
    plt.xlabel('Time (years)')
    plt.ylabel('Fraction Remaining')

    # Setting plot title
    plt.title('Exponential Decay of Radioactive Elements')

    # Setting x and y axis limits
    plt.xlim(0, 20000)
    plt.ylim(0, 1)

    # Adding a legend in the upper right corner
    plt.legend(loc='upper right')

    # Display the plot
    plt.show()
