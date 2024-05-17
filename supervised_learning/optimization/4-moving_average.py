#!/usr/bin/env python3


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set with bias correction.
    """
    moving_averages = []
    v = 0
    for t in range(len(data)):
        v = beta * v + (1 - beta) * data[t]
        bias_corrected_v = v / (1 - beta ** (t + 1))
        moving_averages.append(bias_corrected_v)
    return moving_averages
