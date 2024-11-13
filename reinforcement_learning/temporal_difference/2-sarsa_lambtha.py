#!/usr/bin/env python3
'''SARSA(λ)'''
import gymnasium as gym
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    '''uses epsilon-greedy to determine the next action:
    Args:
        Q is a numpy.ndarray containing the q-table
        state is the current state
        epsilon is the epsilon to use for the calculation
    Returns: the next action index
    '''
    if np.random.rand() > epsilon:
        action = np.argmax(Q[state, :])
    else:
        action = np.random.randint(0, int(Q.shape[1]))
    return action


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    '''performs SARSA(λ)
    Args:
        env is the openAI environment instance
        Q is a numpy.ndarray of shape (s,a) containing the Q table
        lambtha is the eligibility trace factor
        episodes is the total number of episodes to train over
        max_steps is the maximum number of steps per episode
        alpha is the learning rate
        gamma is the discount rate
        epsilon is the initial threshold for epsilon greedy
        min_epsilon is the minimum value that epsilon should decay to
        epsilon_decay is the decay rate for updating epsilon between episodes
    Returns: Q, the updated Q table
    '''
    init_epsilon = epsilon
    Et = np.zeros_like(Q)
    for i in range(episodes):
        state = env.reset()[0]  # Updated for gymnasium compatibility
        action = epsilon_greedy(Q, state, epsilon=epsilon)
        for j in range(max_steps):
            Et *= lambtha * gamma
            Et[state, action] += 1.0
            new_state, reward, terminated, truncated, _ = env.step(action)
            new_action = epsilon_greedy(Q, new_state, epsilon=epsilon)
            # Update reward based on terminal states
            if env.unwrapped.desc.reshape(env.observation_space.n)[new_state] == b'H':
                reward = -1
            elif env.unwrapped.desc.reshape(env.observation_space.n)[new_state] == b'G':
                reward = 1
            # Compute the TD error
            delta = reward + gamma * Q[new_state, new_action] - Q[state, action]
            # Update Q-table with eligibility traces
            Q += alpha * delta * Et
            if terminated or truncated:
                break
            state, action = new_state, new_action
        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))
    return Q

