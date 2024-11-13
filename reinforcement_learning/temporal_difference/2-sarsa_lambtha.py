#!/usr/bin/env python3
'''SARSA(λ)'''
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    '''Uses epsilon-greedy to determine the next action.
    
    Args:
        Q (numpy.ndarray): The Q-table.
        state (int): The current state.
        epsilon (float): The epsilon value for exploration.

    Returns:
        int: The index of the next action.
    '''
    if np.random.rand() > epsilon:
        return np.argmax(Q[state, :])  # Exploitation
    else:
        return np.random.randint(0, Q.shape[1])  # Exploration


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    '''Performs SARSA(λ) algorithm.

    Args:
        env: The environment instance.
        Q (numpy.ndarray): The Q-table of shape (s, a).
        lambtha (float): The eligibility trace decay factor.
        episodes (int): Number of episodes to train over.
        max_steps (int): Maximum number of steps per episode.
        alpha (float): Learning rate.
        gamma (float): Discount factor.
        epsilon (float): Initial epsilon for exploration.
        min_epsilon (float): Minimum epsilon value.
        epsilon_decay (float): Epsilon decay rate per episode.

    Returns:
        numpy.ndarray: Updated Q-table.
    '''
    init_epsilon = epsilon
    Et = np.zeros_like(Q)
    
    for episode in range(episodes):
        state = env.reset()[0]  # Initialize state
        action = epsilon_greedy(Q, state, epsilon)

        for step in range(max_steps):
            Et *= lambtha * gamma
            Et[state, action] += 1.0

            # Take action and observe next state, reward, and termination
            new_state, reward, terminated, truncated, _ = env.step(action)
            new_action = epsilon_greedy(Q, new_state, epsilon)
            
            # Update reward if in terminal states
            if env.unwrapped.desc.reshape(env.observation_space.n)[new_state] == b'H':
                reward = -1
            elif env.unwrapped.desc.reshape(env.observation_space.n)[new_state] == b'G':
                reward = 1

            # TD error calculation
            delta = reward + gamma * Q[new_state, new_action] - Q[state, action]

            # Update Q-values with eligibility traces
            Q += alpha * delta * Et

            if terminated or truncated:
                break

            state, action = new_state, new_action

        # Epsilon decay
        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))

    return Q

