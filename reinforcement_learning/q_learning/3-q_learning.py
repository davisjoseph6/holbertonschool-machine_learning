#!/usr/bin/env python3
"""
Module to perform Q-learning training on the FrozenLake environment.
"""

import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1,
          min_epsilon=0.1, epsilon_decay=0.05):
    """
    Trains the agent using Q-learning on the provided FrozenLake environment.
    """
    total_rewards = []

    for episode in range(episodes):
        state = env.reset()[0]
        episode_reward = 0

        for step in range(max_steps):
            # Select action using epsilon-greedy policy
            action = epsilon_greedy(Q, state, epsilon)

            # Perform action and observe the new state and reward
            next_state, reward, done, _, _ = env.step(action)

            # Modify reward if the agent falls into a hole
            if done and reward == 0:
                reward = -1

            # Update Q-value using the Q-learning formula
            best_next_action = np.argmax(Q[next_state])
            Q[state, action] = Q[state, action] + alpha * (reward + gamma * Q[next_state, best_next_action] - Q[state, action])

            # Update the current state and total reward
            state = next_state
            episode_reward += reward

            if done:
                break

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))
        total_rewards.append(episode_reward)

    return Q, total_rewards
