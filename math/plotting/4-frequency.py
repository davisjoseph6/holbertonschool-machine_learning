#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def frequency():

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    # Plot histogram with bins every 10 units
    bins = range(0, 101, 10)  # Create bins from 0 to 100 with a step of 10
    n, bins, patches = plt.hist(student_grades, bins=bins, edgecolor='black')  # edgecolor outlines the bars

    # Setting labels and title
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')

    # Set y-axis ticks
    max_height = n.max()  # Get the maximum bar height
    plt.yticks(np.arange(0, max_height + 1, 5))  # Set the y-ticks to go from 0 to max_height with steps of 5

    # Show the plot
    plt.show()
