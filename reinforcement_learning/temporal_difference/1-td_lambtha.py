#!/usr/bin/env python3
"""
TD(λ) algorithm
"""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """
    Performs the TD(λ) algorithm for estimating the value function.

    Parameters:
        env: Environment instance.
        V (numpy.ndarray): Array of shape (s,) containing the value estimate.
        policy (function): A function that takes in a state and returns the next action to take.
        lambtha (float): Eligibility trace factor.
        episodes (int): Total number of episodes to train over.
        max_steps (int): Maximum number of steps per episode.
        alpha (float): Learning rate.
        gamma (float): Discount rate.

    Returns:
        numpy.ndarray: Updated value estimates V.
    """
    for episode in range(episodes):
        # Reset the environment and initialize eligibility traces
        state = env.reset()[0]
        eligibility_trace = np.zeros_like(V)

        for step in range(max_steps):
            # Choose an action based on the policy
            action = policy(state)

            # Take action, observe new state, reward, and termination signal
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Calculate the TD error (delta)
            td_error = reward + gamma * V[next_state] * (not terminated) - V[state]

            # Update eligibility trace for the current state
            eligibility_trace[state] += 1

            # Update the value function and eligibility trace for all states
            V += alpha * td_error * eligibility_trace
            eligibility_trace *= gamma * lambtha  # Decay the eligibility traces

            # Move to the next state
            state = next_state

            if terminated or truncated:
                break

    return V
