#!/usr/bin/env python3
"""
This script generates and plots a histogram of student grades for "Project A".
It is designed to help visualize the distribution of grades among students
based on
simulated data. The grades are assumed to be normally distributed around a mean
with a given standard deviation.

The histogram displays the frequency of grades across bins that span the full
range
of possible grades from 0 to 100, with each bin representing a range of 10
grades.
Each bar in the histogram is outlined in black to enhance visual distinction.
import numpy as np
import matplotlib.pyplot as plt

Usage:
This script is intended to be run from the command line and doesn't require any
arguments.
It can be executed with `./4-frequency.py` if it's made executable or `python3
4-frequency.py`.
"""


def frequency():
    """
    Attributes:
    student_grades (numpy.ndarray): An array of randomly generated student
    grades.
    bins (list): A list of integers defining the edges of the bins for the
    histogram.

    Functions:
    main: Sets up the plot with appropriate labels, axes, and visual style, and
    displays the histogram.
    """

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)  # Normal distribution
    plt.figure(figsize=(6.4, 4.8))

    # Define bins starting from 0 to 100
    bins = np.arange(0, 101, 10)  # Bins from 0 to 100 inclusive

    # Plot settings
    plt.xlabel('Grades')
    plt.ylim(0, 30)  # Ensure the y-axis starts at 0 up to 30
    plt.xlim(0, 100)  # Ensure the x-axis spans from 0 to 100
    plt.ylabel('Number of Students')
    plt.title('Project A')
    plt.hist(student_grades, bins, edgecolor='black')  # Histogram with bars
    plt.xticks(np.arange(0, 110, 10))  # Set the x-ticks to match the bin edges

    # Show the plot
    plt.show()
