#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def gradient():

    np.random.seed(5)

    x = np.random.randn(2000) * 10
    y = np.random.randn(2000) * 10
    z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))

    plt.figure(figsize=(10, 6))  # Set the figure size
    sc = plt.scatter(x, y, c=z, s=40, cmap='terrain')  # use a terrain colormap
    cbar = plt.colorbar(sc, label="elevation (m")
    cbar.set_label('elevation (m)', rotation=270, labelpad=20)
    plt.title("Mountain Elevation")
    plt.xlabel("x coordinate (m)")
    plt.ylabel("y coordinate (m)")
    plt.show()
