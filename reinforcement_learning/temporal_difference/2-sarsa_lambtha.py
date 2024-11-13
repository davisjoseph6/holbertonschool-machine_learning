#!/usr/bin/env python3
"""
SARSA(λ) algorithm
"""
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs the SARSA(λ) algorithm for estimating the Q-table.

    Parameters:
        env: Environment instance.
        Q: numpy.ndarray of shape (s, a) containing the Q table.
        lambtha: Eligibility trace factor.
        episodes: Total number of episodes to train over.
        max_steps: Maximum number of steps per episode.
        alpha: Learning rate.
        gamma: Discount rate.
        epsilon: Initial epsilon value for epsilon-greedy policy.
        min_epsilon: Minimum value of epsilon for exploration.
        epsilon_decay: Rate at which epsilon decays after each episode.

    Returns:
        Updated Q table.
    """
    for episode in range(episodes):
        # Reset environment and initialize eligibility traces
        state = env.reset()[0]
        eligibility_traces = np.zeros_like(Q)

        # Choose action using epsilon-greedy policy
        action = np.random.choice(env.action_space.n) if np.random.rand() < epsilon else np.argmax(Q[state])

        for step in range(max_steps):
            # Take action, observe reward and next state
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Choose next action using epsilon-greedy policy
            next_action = np.random.choice(env.action_space.n) if np.random.rand() < epsilon else np.argmax(Q[next_state])

            # Calculate the TD error
            delta = reward + gamma * Q[next_state, next_action] * (not terminated) - Q[state, action]

            # Update eligibility trace for the current state-action pair
            eligibility_traces[state, action] += 1

            # Update Q values and decay eligibility traces
            Q += alpha * delta * eligibility_traces
            eligibility_traces *= gamma * lambtha

            # Move to the next state and action
            state, action = next_state, next_action

            if terminated or truncated:
                break

        # Decay epsilon after each episode
        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))

    return Q

