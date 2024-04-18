#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def gradient():

    np.random.seed(5)

    x = np.random.randn(2000) * 10
    y = np.random.randn(2000) * 10
    z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))

    # Generate coordinates and elevation data
    x = np.random.randn(2000) * 10
    y = np.random.randn(2000) * 10
    # Elevation 
    z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))

    # Create a scatter plot
    plt.figure(figsize=(6.4, 4.8))
    scatter = plt.scatter(x, y, c=z, cmap='viridis')  # Use the virids colormap

    # Create a colorbar with a label
    cbar = plt.colorbar(scatter)
    cbar.set_label('elevation (m)')

    # Labeling the axes and setting a title
    plt.xlabel('x coordinate (m)')
    plt.ylabel('y coordinate (m)')
    plt.title('Mountain Elevation')

    # Show the plot
    plt.show()



