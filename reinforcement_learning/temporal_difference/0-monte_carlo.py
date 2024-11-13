#!/usr/bin/env python3
"""
    Monte Carlo algorithm
"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
        Function that performs the Monte Carlo algorithm

    :param env: openAI env instance
    :param V: ndarray, shape(s,) value estimate
    :param policy: function that takes in state and return next action
    :param episodes: total number of episodes to train over
    :param max_steps: max number of steps per episode
    :param alpha: learning rate
    :param gamma: discount rate

    :return: V, updated value estimate
    """
    for ep in range(episodes):
        # Start a new episode and get the initial state
        state = env.reset()[0]  # Unpack only the observation

        episode_data = []

        for step in range(max_steps):
            # Determine action based on policy
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            # Store state and reward
            episode_data.append((state, reward))

            if terminated or truncated:
                break

            state = next_state

        # Calculate returns and update the value function
        G = 0
        visited_states = set()
        for s, r in reversed(episode_data):
            G = gamma * G + r
            if s not in visited_states:
                visited_states.add(s)
                V[s] += alpha * (G - V[s])

    # Setting print precision for display consistency
    np.set_printoptions(precision=4, suppress=True)

    return V.round(4)

