# Dimensionality Reduction: t-SNE and PCA

This project implements dimensionality reduction techniques including **Principal Component Analysis (PCA)** and **t-Distributed Stochastic Neighbor Embedding (t-SNE)**. These techniques are widely used for visualization and analysis of high-dimensional datasets.

---

## Directory Overview

### Files and Functions

1. **`1-pca.py`**
   - **`pca(X, ndim)`**
     - Performs PCA on a dataset to reduce its dimensionality.
     - **Inputs**:
       - `X`: The dataset (numpy array).
       - `ndim`: Target number of dimensions.
     - **Outputs**:
       - `T`: Transformed dataset in reduced dimensionality.

2. **`2-P_init.py`**
   - **`P_init(X, perplexity)`**
     - Initializes variables required to compute P affinities in t-SNE.
     - **Inputs**:
       - `X`: Dataset.
       - `perplexity`: Desired perplexity for t-SNE.
     - **Outputs**:
       - Pairwise distance matrix `D`.
       - Initialized P affinities matrix `P`.
       - Precision values `betas`.
       - Shannon entropy target `H`.

3. **`3-entropy.py`**
   - **`HP(Di, beta)`**
     - Computes the Shannon entropy and P affinities for a given point.
     - **Inputs**:
       - `Di`: Distances from a point to others.
       - `beta`: Precision for the Gaussian distribution.
     - **Outputs**:
       - `Hi`: Shannon entropy.
       - `Pi`: P affinities.

4. **`4-P_affinities.py`**
   - **`P_affinities(X, tol, perplexity)`**
     - Computes symmetric P affinities for t-SNE.
     - **Inputs**:
       - `X`: Dataset.
       - `tol`: Tolerance for perplexity calculations.
       - `perplexity`: Desired perplexity.
     - **Outputs**:
       - Symmetric P affinities.

5. **`5-Q_affinities.py`**
   - **`Q_affinities(Y)`**
     - Computes Q affinities for the low-dimensional representation.
     - **Inputs**:
       - `Y`: Low-dimensional dataset.
     - **Outputs**:
       - `Q`: Q affinities matrix.
       - `num`: Numerator values for Q calculation.

6. **`6-grads.py`**
   - **`grads(Y, P)`**
     - Computes gradients for updating Y in t-SNE.
     - **Inputs**:
       - `Y`: Low-dimensional dataset.
       - `P`: Symmetric P affinities.
     - **Outputs**:
       - `dY`: Gradients for Y.
       - `Q`: Q affinities.

7. **`7-cost.py`**
   - **`cost(P, Q)`**
     - Computes the Kullback-Leibler divergence (cost) between P and Q.
     - **Inputs**:
       - `P`: Symmetric P affinities.
       - `Q`: Q affinities.
     - **Outputs**:
       - `C`: Cost value.

8. **`8-tsne.py`**
   - **`tsne(X, ndims, idims, perplexity, iterations, lr)`**
     - Performs a t-SNE transformation on a dataset.
     - **Inputs**:
       - `X`: Dataset.
       - `ndims`: Target dimensionality for t-SNE.
       - `idims`: Initial dimensionality after PCA.
       - `perplexity`: Desired perplexity.
       - `iterations`: Number of optimization iterations.
       - `lr`: Learning rate for optimization.
     - **Outputs**:
       - `Y`: Low-dimensional representation of the dataset.

---

## How t-SNE Works

1. **PCA Initialization**:
   - Reduces the dataset to a lower dimension (`idims`) to improve computational efficiency.

2. **P Affinities Calculation**:
   - Converts high-dimensional distances to probabilities using a Gaussian distribution and a target perplexity.

3. **Q Affinities Calculation**:
   - Computes low-dimensional affinities using a t-distribution to handle crowding effects.

4. **Optimization**:
   - Iteratively updates the low-dimensional representation (`Y`) by minimizing the Kullback-Leibler divergence between `P` and `Q`.

---

## How to Use

1. Use **`pca(X, ndim)`** for simple dimensionality reduction.
2. For visualization or advanced reduction, use **`tsne(X, ndims, idims, perplexity, iterations, lr)`**:
   - Example:
     ```python
     import numpy as np
     from 8-tsne import tsne

     # Load your dataset
     X = np.loadtxt("mnist2500_X.txt")
     
     # Perform t-SNE
     Y = tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500)

     # Visualize results
     import matplotlib.pyplot as plt
     plt.scatter(Y[:, 0], Y[:, 1])
     plt.show()
     ```

---

## Applications
- **Data Visualization**: Reducing high-dimensional data to 2D or 3D for better interpretability.
- **Clustering**: Enhancing clustering algorithms by transforming data to a lower-dimensional space.
- **Feature Reduction**: Simplifying datasets while preserving important information.

---

## Requirements
- Python 3.x
- NumPy

---

## References
- van der Maaten, Laurens, and Geoffrey Hinton. "Visualizing data using t-SNE." *Journal of machine learning research* 9.Nov (2008): 2579-2605.
- Principal Component Analysis (PCA).

---

## Author
- Davis Joseph ([LinkedIn](https://www.linkedin.com/in/davisjoseph767/))

