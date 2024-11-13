#!/usr/bin/env python3
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.01, gamma=0.99):
    """
    Performs the Monte Carlo algorithm for value estimation.
    """
    returns_sum = {state: 0 for state in range(len(V))}
    returns_count = {state: 0 for state in range(len(V))}
    for episode in range(episodes):
        state = env.reset()[0]
        episode_log = []

        # Generate an episode
        for step in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_log.append((state, reward))
            state = next_state
            if terminated or truncated:
                break

        # Calculate returns and update value estimates
        G = 0
        visited_states = set()
        for state, reward in reversed(episode_log):
            G = reward + gamma * G
            if state not in visited_states:
                visited_states.add(state)
                returns_sum[state] += G
                returns_count[state] += 1
                # Update the value function based on average return
                V[state] = returns_sum[state] / returns_count[state]

    return V
