#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def gradient():

    # Set the seed for reproducibility
    np.random.seed(5)

    # Generate random data for x and y coordinates
    x = np.random.randn(2000) * 10
    y = np.random.randn(2000) * 10

    # Calculate elevation z
    z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))
    plt.figure(figsize=(6.4, 4.8))  # Set the figure size

    # Create the scatter plot
    sc = plt.scatter(x, y, c=z)  # Scatter plot of x and y
    
    # Add a colorbar with a label
    plt.colorbar(sc, label='elevation (m)')
    
    # Add labels and title to the plot
    plt.title('Mountain Elevation')
    plt.xlabel('x coordinate (m)')
    plt.ylabel('y coordinate (m)')

    # Display the plot
    plt.show()
