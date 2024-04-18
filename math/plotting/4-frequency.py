#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def frequency():

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)  # Normal distribution of grades
    plt.figure(figsize=(6.4, 4.8))

    # Define bins starting from 0 to 100
    bins = np.arange(0, 101, 10)  # Bins from 0 to 100 inclusive
    
    # Plot settings
    plt.xlabel('Grades')
    plt.ylim(0, 30)  # Ensure the y-axis starts at 0 up to 30
    plt.xlim(0, 100)  # Ensure the x-axis spans from 0 to 100
    plt.ylabel('Number of Students')
    plt.title('Project A')
    plt.hist(student_grades, bins, edgecolor='black')  # Histogram with black-edged bars
    plt.xticks(np.arange(0, 110, 10))  # Set the x-ticks to match the bin edges

    # Show the plot
    plt.show()
