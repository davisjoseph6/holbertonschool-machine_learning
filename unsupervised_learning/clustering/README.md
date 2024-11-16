# Clustering Algorithms and Techniques

This repository contains implementations of various clustering algorithms and techniques, including **K-Means**, **Gaussian Mixture Models (GMMs)**, **Bayesian Information Criterion (BIC)**, and **Agglomerative Clustering**. These methods are widely used for unsupervised learning tasks in data analysis and machine learning.

---

## Directory Overview

### Files and Functions

1. **`0-initialize.py`**
   - **`initialize(X, k)`**:
     - Initializes cluster centroids for K-Means clustering.
     - **Inputs**: Dataset `X`, number of clusters `k`.
     - **Outputs**: Centroids as a numpy array of shape `(k, d)`.

2. **`1-kmeans.py`**
   - **`kmeans(X, k, iterations)`**:
     - Performs K-Means clustering.
     - **Inputs**: Dataset `X`, number of clusters `k`, max iterations.
     - **Outputs**: Tuple containing centroids `(C)` and cluster assignments `(clss)`.

3. **`2-variance.py`**
   - **`variance(X, C)`**:
     - Calculates the total intra-cluster variance for a dataset.
     - **Inputs**: Dataset `X`, centroids `C`.
     - **Outputs**: Variance as a float.

4. **`3-optimum.py`**
   - **`optimum_k(X, kmin, kmax, iterations)`**:
     - Finds the optimal number of clusters by testing variances.
     - **Inputs**: Dataset `X`, range of clusters (`kmin`, `kmax`), iterations.
     - **Outputs**: Tuple containing clustering results and variance differences.

5. **`4-initialize.py`**
   - **`initialize(X, k)`**:
     - Initializes parameters for GMMs, including priors, centroids, and covariance matrices.
     - **Inputs**: Dataset `X`, number of clusters `k`.
     - **Outputs**: Tuple `(pi, m, S)` for priors, means, and covariances.

6. **`5-pdf.py`**
   - **`pdf(X, m, S)`**:
     - Computes the probability density function of a Gaussian distribution.
     - **Inputs**: Dataset `X`, mean `m`, covariance matrix `S`.
     - **Outputs**: PDF values as a numpy array.

7. **`6-expectation.py`**
   - **`expectation(X, pi, m, S)`**:
     - Performs the expectation step in the EM algorithm for GMMs.
     - **Inputs**: Dataset `X`, priors `pi`, means `m`, covariances `S`.
     - **Outputs**: Posterior probabilities and log-likelihood.

8. **`7-maximization.py`**
   - **`maximization(X, g)`**:
     - Performs the maximization step in the EM algorithm for GMMs.
     - **Inputs**: Dataset `X`, posterior probabilities `g`.
     - **Outputs**: Updated priors, means, and covariances.

9. **`8-EM.py`**
   - **`expectation_maximization(X, k, iterations, tol, verbose)`**:
     - Full implementation of the EM algorithm for GMMs.
     - **Inputs**: Dataset `X`, number of clusters `k`, iterations, tolerance, verbosity.
     - **Outputs**: Parameters `(pi, m, S)`, posterior probabilities `g`, log-likelihood.

10. **`9-BIC.py`**
    - **`BIC(X, kmin, kmax, iterations, tol, verbose)`**:
      - Determines the optimal number of clusters for GMMs using the Bayesian Information Criterion.
      - **Inputs**: Dataset `X`, cluster range, iterations, tolerance, verbosity.
      - **Outputs**: Optimal number of clusters, best GMM parameters, log-likelihoods, and BIC values.

11. **`10-kmeans.py`**
    - **`kmeans(X, k)`**:
      - K-Means clustering using scikit-learn.
      - **Inputs**: Dataset `X`, number of clusters `k`.
      - **Outputs**: Centroids and cluster labels.

12. **`11-gmm.py`**
    - **`gmm(X, k)`**:
      - Gaussian Mixture Model clustering using scikit-learn.
      - **Inputs**: Dataset `X`, number of clusters `k`.
      - **Outputs**: Priors, means, covariances, cluster labels, and BIC.

13. **`12-agglomerative.py`**
    - **`agglomerative(X, dist)`**:
      - Performs agglomerative clustering and visualizes the dendrogram.
      - **Inputs**: Dataset `X`, maximum cophenetic distance `dist`.
      - **Outputs**: Cluster indices.

---

## How to Use

1. **K-Means Clustering**
   - Run K-Means on a dataset:
     ```python
     from 1-kmeans import kmeans
     centroids, labels = kmeans(X, k=3)
     ```

2. **Gaussian Mixture Models**
   - Fit GMM to a dataset:
     ```python
     from 8-EM import expectation_maximization
     pi, m, S, g, log_likelihood = expectation_maximization(X, k=3)
     ```

3. **Find Optimal Clusters with BIC**
   - Use BIC to determine the best number of clusters:
     ```python
     from 9-BIC import BIC
     best_k, best_model, log_likelihoods, bics = BIC(X, kmin=1, kmax=10)
     ```

4. **Agglomerative Clustering**
   - Perform hierarchical clustering and plot the dendrogram:
     ```python
     from 12-agglomerative import agglomerative
     cluster_indices = agglomerative(X, dist=10)
     ```

---

## Requirements

- Python 3.x
- NumPy
- SciPy
- scikit-learn
- Matplotlib

---

## Applications

- **Market Segmentation**: Grouping customers based on purchase history.
- **Anomaly Detection**: Identifying unusual patterns in datasets.
- **Data Compression**: Reducing dimensionality for efficient storage and retrieval.
- **Bioinformatics**: Grouping genes or proteins with similar behavior.

---

## References

- K-Means: "Lloyd's Algorithm (1957)"
- Gaussian Mixture Models: "Expectation-Maximization Algorithm (Dempster et al., 1977)"
- Agglomerative Clustering: "Hierarchical Clustering (Ward, 1963)"
- Bayesian Information Criterion: "Schwarz, Gideon E. (1978)"

---

## Author

- **Davis Joseph**  
  Artificial Intelligence Research Engineer  
  [LinkedIn](https://www.linkedin.com/in/davisjoseph767/)

