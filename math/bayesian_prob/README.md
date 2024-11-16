# Bayesian Probability

This project explores key concepts in Bayesian statistics and their applications, focusing on calculating likelihood, intersection, marginal, and posterior probabilities. It includes both discrete and continuous implementations, providing a comprehensive toolkit for Bayesian inference.

---

## Directory Overview

### Files and Functions

1. **`0-likelihood.py`**
   - **`likelihood(x, n, P)`**: 
     - Computes the likelihood of obtaining the observed data given various hypothetical probabilities.
     - Inputs:
       - `x`: Number of successes.
       - `n`: Total number of trials.
       - `P`: Array of probabilities.
     - Returns:
       - Likelihood values for each probability.

2. **`1-intersection.py`**
   - **`intersection(x, n, P, Pr)`**: 
     - Calculates the intersection of the likelihood and the prior probabilities.
     - Inputs:
       - `x`, `n`, `P`: As described above.
       - `Pr`: Array of prior probabilities for each hypothesis.
     - Returns:
       - Intersection values.

3. **`2-marginal.py`**
   - **`marginal(x, n, P, Pr)`**: 
     - Computes the marginal probability of observing the data by summing over all hypotheses.
     - Inputs:
       - `x`, `n`, `P`, `Pr`: As described above.
     - Returns:
       - Marginal probability.

4. **`3-posterior.py`**
   - **`posterior(x, n, P, Pr)`**: 
     - Calculates the posterior probabilities using Bayes' theorem.
     - Inputs:
       - `x`, `n`, `P`, `Pr`: As described above.
     - Returns:
       - Posterior probabilities for each hypothesis.

5. **`100-continuous.py`**
   - **`posterior(x, n, p1, p2)`**: 
     - Computes the posterior probability that the probability of success lies within a specific range `[p1, p2]`.
     - Inputs:
       - `x`, `n`: As described above.
       - `p1`, `p2`: Range of probabilities.
     - Returns:
       - Posterior probability.

---

## Key Concepts

### Bayesian Inference
Bayesian inference provides a framework to update the probability of a hypothesis based on observed data and prior beliefs. This project implements core calculations such as:
- **Likelihood**: How well the data supports each hypothesis.
- **Intersection**: Combines the likelihood and prior probabilities.
- **Marginal Probability**: Normalizing factor for posterior calculations.
- **Posterior Probability**: Updated probabilities after observing the data.

### Continuous Bayesian Inference
In addition to discrete probabilities, this project includes continuous posterior probability calculations using the Beta distribution and its cumulative density function (CDF).

---

## How to Use

1. **Likelihood**
   - Use `likelihood(x, n, P)` to calculate the probability of observing data for each hypothesis.

2. **Intersection**
   - Use `intersection(x, n, P, Pr)` to combine likelihoods with prior probabilities.

3. **Marginal**
   - Use `marginal(x, n, P, Pr)` to calculate the total probability of the observed data.

4. **Posterior**
   - Use `posterior(x, n, P, Pr)` to update your beliefs based on observed data.

5. **Continuous Posterior**
   - Use `posterior(x, n, p1, p2)` to compute the posterior probability for a continuous range of probabilities.

---

## Applications
- Hypothesis testing and decision-making.
- Bayesian updating in real-world scenarios, such as clinical trials or A/B testing.
- Understanding uncertainty and probabilistic reasoning.

---

## Requirements
- Python 3.x
- NumPy
- SciPy (for Beta distribution CDF)

---

## References
- Bayesian Probability Theory
- Beta Distribution for Posterior Analysis

---

## Author
- Davis Joseph ([LinkedIn](https://www.linkedin.com/in/davis-joseph/))

