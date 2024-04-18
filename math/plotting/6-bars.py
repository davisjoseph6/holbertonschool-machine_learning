#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def bars():
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    
    # Create a figure
    plt.figure(figsize=(6.4, 4.8))

    # Names of people
    people = ['Farrah', 'Fred', 'Felicia']
    # Colors for each fruit
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']  # Red, Yellow, Orange, Peach
    # Fruit names
    fruits = ['apples', 'bananas', 'oranges', 'peaches']

    # Bottom offset for each bar
    bottom = np.zeros(3)

    # Create stacked bars
    for idx, row in enumerate(fruit):
        plt.bar(people, row, color=colors[idx], label=fruits[idx], bottom=bottom, width=0.5)
        bottom += row

    # Labeling and setting limits
    plt.ylabel('Quantity of Fruit')
    plt.ylim(0, 80)
    plt.yticks(np.arange(0, 81, 10))
    plt.title('Number of Fruit per Person')

    # Adding a legend
    plt.legend()

    # Show the plot
    plt.show()
