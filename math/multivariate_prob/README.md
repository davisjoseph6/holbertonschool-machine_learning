# Multivariate Probability

This project focuses on multivariate probability, including the computation of mean and covariance, correlation matrices, and the probability density function (PDF) of a multivariate normal distribution. These concepts are fundamental in multivariate statistics and machine learning.

---

## Directory Overview

### Files and Functions

1. **`0-mean_cov.py`**
   - **`mean_cov(X)`**
     - Calculates the mean and covariance of a given data set.
     - **Inputs**:
       - `X`: A 2D `numpy.ndarray` where each row represents a data point and each column a feature.
     - **Outputs**:
       - `mean`: A 1D array containing the mean of each feature.
       - `cov`: The covariance matrix of the data.

2. **`1-correlation.py`**
   - **`correlation(C)`**
     - Computes the correlation matrix from a covariance matrix.
     - **Inputs**:
       - `C`: A square `numpy.ndarray` representing the covariance matrix.
     - **Outputs**:
       - The correlation matrix, derived by normalizing the covariance matrix using the standard deviations.

3. **`multinormal.py`**
   - **`MultiNormal` Class**
     - Represents a multivariate normal distribution.
     - **Attributes**:
       - `mean`: Mean vector of the distribution.
       - `cov`: Covariance matrix of the distribution.
     - **Methods**:
       - **`__init__(data)`**:
         - Initializes the distribution with a given data set.
         - Computes the mean and covariance matrix.
       - **`pdf(x)`**:
         - Calculates the PDF value for a given data point.
         - **Inputs**:
           - `x`: A 1D `numpy.ndarray` representing a data point.
         - **Outputs**:
           - The probability density value for `x`.

---

## Key Concepts

### Mean and Covariance
- The **mean vector** provides the average of each feature in the dataset.
- The **covariance matrix** describes the relationships between different features:
  - Diagonal elements represent the variance of each feature.
  - Off-diagonal elements represent the covariance between features.

### Correlation Matrix
- Normalizes the covariance matrix to provide a dimensionless measure of linear relationships between variables.
- Values range between `-1` (perfect negative correlation) and `1` (perfect positive correlation).

### Multivariate Normal Distribution
- Extends the univariate normal distribution to multiple dimensions.
- Defined by:
  - A mean vector (`mean`).
  - A covariance matrix (`cov`).
- The **PDF**:
  - Provides the likelihood of observing a specific data point.
  - Utilizes the determinant and inverse of the covariance matrix.

---

## How to Use

1. **Calculate Mean and Covariance**
   - Use `mean_cov(X)` to compute the statistical properties of your data.

2. **Generate a Correlation Matrix**
   - Use `correlation(C)` to analyze the relationships between features in your dataset.

3. **Work with Multivariate Normal Distributions**
   - Create an instance of the `MultiNormal` class using a dataset.
   - Use the `pdf(x)` method to calculate the likelihood of specific data points.

---

## Applications
- Statistical modeling and hypothesis testing.
- Principal Component Analysis (PCA).
- Multivariate data analysis in machine learning and finance.
- Probabilistic inference in Bayesian networks.

---

## Requirements
- Python 3.x
- NumPy

---

## References
- Multivariate Statistics and Machine Learning
- Probability and Statistics in Data Science

---

## Author
- Davis Joseph ([LinkedIn](https://www.linkedin.com/in/davisjoseph767/))

