# Hidden Markov Models (HMM)

This project implements various algorithms and functionalities for **Hidden Markov Models (HMMs)**, including Markov Chains, Forward-Backward Algorithms, Viterbi Algorithm, Baum-Welch Algorithm, and Markov Chain properties.

---

## Directory Overview

### Files and Functions

1. **`0-markov_chain.py`**
   - **`markov_chain(P, s, t)`**
     - Calculates the probability of a Markov chain being in a particular state after `t` transitions.
     - **Inputs**:
       - `P`: Transition matrix.
       - `s`: Initial state probabilities.
       - `t`: Number of transitions.
     - **Outputs**:
       - State probabilities after `t` transitions.

2. **`1-regular.py`**
   - **`regular(P)`**
     - Determines the steady-state probabilities of a regular Markov chain.
     - **Inputs**:
       - `P`: Transition matrix.
     - **Outputs**:
       - Steady-state probabilities or `None` if the chain is not regular.

3. **`2-absorbing.py`**
   - **`absorbing(P)`**
     - Determines if a Markov chain is absorbing.
     - **Inputs**:
       - `P`: Transition matrix.
     - **Outputs**:
       - `True` if the chain is absorbing, `False` otherwise.

4. **`3-forward.py`**
   - **`forward(Observation, Emission, Transition, Initial)`**
     - Implements the forward algorithm for an HMM.
     - **Inputs**:
       - `Observation`: Sequence of observations.
       - `Emission`: Emission probability matrix.
       - `Transition`: Transition probability matrix.
       - `Initial`: Initial state probabilities.
     - **Outputs**:
       - Total likelihood of the observations and forward path probabilities.

5. **`4-viterbi.py`**
   - **`viterbi(Observation, Emission, Transition, Initial)`**
     - Implements the Viterbi algorithm to find the most likely sequence of states.
     - **Inputs**:
       - `Observation`: Sequence of observations.
       - `Emission`: Emission probability matrix.
       - `Transition`: Transition probability matrix.
       - `Initial`: Initial state probabilities.
     - **Outputs**:
       - Most likely sequence of hidden states and its probability.

6. **`5-backward.py`**
   - **`backward(Observation, Emission, Transition, Initial)`**
     - Implements the backward algorithm for an HMM.
     - **Inputs**:
       - `Observation`: Sequence of observations.
       - `Emission`: Emission probability matrix.
       - `Transition`: Transition probability matrix.
       - `Initial`: Initial state probabilities.
     - **Outputs**:
       - Total likelihood of the observations and backward path probabilities.

7. **`6-baum_welch.py`**
   - **`baum_welch(Observations, Transition, Emission, Initial, iterations)`**
     - Implements the Baum-Welch algorithm to estimate HMM parameters.
     - **Inputs**:
       - `Observations`: Sequence of observations.
       - `Transition`: Initial transition matrix.
       - `Emission`: Initial emission matrix.
       - `Initial`: Initial state probabilities.
       - `iterations`: Number of iterations for optimization.
     - **Outputs**:
       - Updated `Transition` and `Emission` matrices.

---

## How Hidden Markov Models Work

1. **Markov Chains**:
   - Describes state transitions using a transition matrix `P`.

2. **Forward Algorithm**:
   - Computes the likelihood of observing a sequence of events given an HMM.

3. **Backward Algorithm**:
   - Calculates probabilities of observing future events.

4. **Viterbi Algorithm**:
   - Determines the most likely sequence of hidden states given observed events.

5. **Baum-Welch Algorithm**:
   - Optimizes HMM parameters (`Transition`, `Emission`) using observed data.

---

## How to Use

### Example: Forward Algorithm
```python
import numpy as np
from forward import forward

# Define HMM parameters
Observation = np.array([0, 1, 0])
Emission = np.array([[0.9, 0.1], [0.2, 0.8]])
Transition = np.array([[0.7, 0.3], [0.4, 0.6]])
Initial = np.array([[0.6], [0.4]])

# Compute forward probabilities
P, F = forward(Observation, Emission, Transition, Initial)
print("Likelihood of Observation:", P)
```

### Example: Viterbi Algorithm
```python
from viterbi import viterbi

path, P = viterbi(Observation, Emission, Transition, Initial)
print("Most likely states:", path)
print("Probability:", P)
```

### Example: Baum-Welch Algorithm
```python
from baum_welch import baum_welch

Transition, Emission = baum_welch(Observation, Transition, Emission, Initial)
print("Updated Transition Matrix:", Transition)
print("Updated Emission Matrix:", Emission)
```

## Requirements
- Python 3.x
- NumPy

## Applications
- Speech recognition
- Natural language processing
- Gene prediction
- Time series analysis

## Author
Davis Joseph (LinkedIn)
