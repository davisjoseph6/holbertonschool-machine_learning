# Probability Distributions

This project implements several probability distributions: **Poisson**, **Exponential**, **Normal**, and **Binomial**. These are essential for understanding statistical models, machine learning, and data analysis.

---

## Directory Overview

### Files and Implementations

1. **`poisson.py`**  
   - Implements the **Poisson distribution**.
   - **Parameters**:
     - `lambtha`: The rate parameter (λ) for the distribution.
   - **Methods**:
     - `pmf(k)`: Calculates the Probability Mass Function (PMF) for k successes.
     - `cdf(k)`: Calculates the Cumulative Distribution Function (CDF) for k successes.

2. **`exponential.py`**  
   - Implements the **Exponential distribution**.
   - **Parameters**:
     - `lambtha`: The rate parameter (λ) for the distribution.
   - **Methods**:
     - `pdf(x)`: Calculates the Probability Density Function (PDF) for a given x.
     - `cdf(x)`: Calculates the Cumulative Distribution Function (CDF) for a given x.

3. **`normal.py`**  
   - Implements the **Normal (Gaussian) distribution**.
   - **Parameters**:
     - `mean`: The mean (μ) of the distribution.
     - `stddev`: The standard deviation (σ) of the distribution.
   - **Methods**:
     - `z_score(x)`: Calculates the z-score of a given x.
     - `x_value(z)`: Calculates the x-value for a given z-score.
     - `pdf(x)`: Calculates the PDF for a given x.
     - `cdf(x)`: Calculates the CDF for a given x.

4. **`binomial.py`**  
   - Implements the **Binomial distribution**.
   - **Parameters**:
     - `n`: The number of trials.
     - `p`: The probability of success in each trial.
   - **Methods**:
     - `pmf(k)`: Calculates the PMF for k successes.
     - `cdf(k)`: Calculates the CDF for k successes.

---

## Features and Applications

### Features
- **Robust Validation**: Ensures valid parameters and input data for each distribution.
- **Statistical Metrics**: Computes PMF, PDF, CDF, and other related functions.
- **Flexible Initialization**: Distributions can be initialized with data or parameters.

### Applications
1. **Poisson Distribution**:
   - Modeling events that occur at a constant rate, e.g., call arrivals at a call center.
2. **Exponential Distribution**:
   - Modeling time between events in a Poisson process, e.g., time between customer arrivals.
3. **Normal Distribution**:
   - Describing natural phenomena, e.g., heights, weights, or test scores.
4. **Binomial Distribution**:
   - Modeling the number of successes in a fixed number of trials, e.g., flipping a coin.

---

## Usage

### Example: Poisson Distribution
```python
from poisson import Poisson

poisson = Poisson(lambtha=5)
print("PMF for k=3:", poisson.pmf(3))
print("CDF for k=3:", poisson.cdf(3))
```

### Example: Exponential Distribution
```python
from exponential import Exponential

exponential = Exponential(lambtha=2)
print("PDF for x=1:", exponential.pdf(1))
print("CDF for x=1:", exponential.cdf(1))
```

### Example: Normal Distribution
```python
from normal import Normal

normal = Normal(mean=0, stddev=1)
print("PDF for x=0:", normal.pdf(0))
print("CDF for x=1:", normal.cdf(1))
```

### Example: Binomial Distribution
```python
from binomial import Binomial

binomial = Binomial(n=10, p=0.5)
print("PMF for k=5:", binomial.pmf(5))
print("CDF for k=5:", binomial.cdf(5))
```

## Requirements
- Python 3.x

## Author
- Davis Joseph
