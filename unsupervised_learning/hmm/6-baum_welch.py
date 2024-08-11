#!/usr/bin/env python3
import numpy as np

def forward(Observation, Emission, Transition, Initial):
    """
    Performs the forward algorithm for a hidden Markov model.

    Parameters:
    - Observation (numpy.ndarray): Array of shape (T,) that contains the index of the observation.
                                   T is the number of observations.
    - Emission (numpy.ndarray): Array of shape (N, M) containing the emission probability of a specific
                                observation given a hidden state.
    - Transition (numpy.ndarray): 2D array of shape (N, N) containing the transition probabilities.
    - Initial (numpy.ndarray): Array of shape (N, 1) containing the probability of starting in a particular hidden state.

    Returns:
    - F (numpy.ndarray): Array of shape (N, T) containing the forward path probabilities.
    """
    T = Observation.shape[0]  # Number of observations
    N = Emission.shape[0]      # Number of states

    # Initialize the forward probability matrix
    F = np.zeros((N, T))

    # Initial step: Compute forward probabilities for the first observation
    F[:, 0] = Initial.T * Emission[:, Observation[0]]

    # Recursion step: Compute forward probabilities for each subsequent observation
    for t in range(1, T):
        for j in range(N):
            F[j, t] = np.sum(F[:, t-1] * Transition[:, j] * Emission[j, Observation[t]])

    return F

def backward(Observation, Emission, Transition):
    """
    Performs the backward algorithm for a hidden Markov model.

    Parameters:
    - Observation (numpy.ndarray): Array of shape (T,) that contains the index of the observation.
                                   T is the number of observations.
    - Emission (numpy.ndarray): Array of shape (N, M) containing the emission probability of a specific
                                observation given a hidden state.
    - Transition (numpy.ndarray): 2D array of shape (N, N) containing the transition probabilities.

    Returns:
    - B (numpy.ndarray): Array of shape (N, T) containing the backward path probabilities.
    """
    T = Observation.shape[0]  # Number of observations
    N = Emission.shape[0]      # Number of states

    # Initialize the backward probability matrix
    B = np.zeros((N, T))

    # Initialization step: Set backward probabilities for the last observation to 1
    B[:, T-1] = 1

    # Recursion step: Compute backward probabilities for each previous observation
    for t in range(T-2, -1, -1):
        for i in range(N):
            B[i, t] = np.sum(Transition[i, :] * Emission[:, Observation[t+1]] * B[:, t+1])

    return B

def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Performs the Baum-Welch algorithm to estimate the parameters of a hidden Markov model.

    Parameters:
    - Observations (numpy.ndarray): Array of shape (T,) that contains the index of the observation.
                                    T is the number of observations.
    - Transition (numpy.ndarray): Array of shape (M, M) containing the initialized transition probabilities.
                                  M is the number of hidden states.
    - Emission (numpy.ndarray): Array of shape (M, N) containing the initialized emission probabilities.
                                N is the number of output states.
    - Initial (numpy.ndarray): Array of shape (M, 1) containing the initialized starting probabilities.
    - iterations (int): Number of iterations for expectation-maximization.

    Returns:
    - Transition (numpy.ndarray): The converged transition probabilities.
    - Emission (numpy.ndarray): The converged emission probabilities.
    """
    N, M = Emission.shape  # N = Number of hidden states, M = Number of possible observations
    T = Observations.shape[0]  # Number of observations

    # Loop for the specified number of iterations
    for n in range(iterations):
        # E-step: Calculate forward and backward probabilities
        alpha = forward(Observations, Emission, Transition, Initial)
        beta = backward(Observations, Emission, Transition)

        # Initialize variables for xi (probability of transitioning from state i to state j) and gamma (probability of being in state i)
        xi = np.zeros((N, N, T-1))
        gamma = np.zeros((N, T))

        # Compute xi and gamma for each time step
        for t in range(T-1):
            # Compute the denominator for normalization
            denominator = np.sum(alpha[:, t] * beta[:, t])
            for i in range(N):
                gamma[i, t] = (alpha[i, t] * beta[i, t]) / denominator
                for j in range(N):
                    xi[i, j, t] = (alpha[i, t] * Transition[i, j] *
                                   Emission[j, Observations[t+1]] *
                                   beta[j, t+1]) / denominator

        # Compute gamma for the last observation (T-1)
        gamma[:, T-1] = (alpha[:, T-1] * beta[:, T-1]) / np.sum(alpha[:, T-1] * beta[:, T-1])

        # M-step: Update Transition and Emission matrices
        # Update Transition matrix
        Transition = np.sum(xi, axis=2) / np.sum(gamma[:, :-1], axis=1).reshape(-1, 1)

        # Update Emission matrix
        for k in range(M):
            Emission[:, k] = np.sum(gamma[:, Observations == k], axis=1)
        Emission /= np.sum(gamma, axis=1).reshape(-1, 1)

    return Transition, Emission

