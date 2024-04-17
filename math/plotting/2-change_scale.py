#!/usr/bin/env python3
"""
code to plot x->y as a line graph
"""
import numpy as np
import matplotlib.pyplot as plt


def change_scale():
    """
    code to plot x->y
    """
    x = np.arange(0, 28651, 5730)
    r = np.log(0.5)
    t = 5730
    y = np.exp((r / t) * x)
    plt.figure(figsize=(6.4, 4.8))

    # Plot data as a line graph
    plt.plot(x, y, label='C-14 decay')

    # Set the labels for x and y axes
    plt.xlabel('Time (years)')
    plt.ylabel('Fraction Remaining')

    # Set the title of the graph
    plt.title('Exponential Decay of C')

    # Set y-axis to logarithmic scale
    plt.yscale('log')

    # Set the limits for the x-axis
    plt.xlim(0, 28650)

    # Optional: Add a grid for better readability
    plt.grid(True)

    # Display the plot
    plt.show()
