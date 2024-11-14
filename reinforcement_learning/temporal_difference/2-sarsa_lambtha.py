#!/usr/bin/env python3
"""
    SARSA(λ) algorithm
"""
import numpy as np


def epsilon_greedy(state, Q, epsilon):
    """ uses epsilon-greedy to determine the next action"""
    p = np.random.uniform(0, 1)
    if p > epsilon:
        action = np.argmax(Q[state, :])
    else:
        action = np.random.randint(0, int(Q.shape[1]))
    return action


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1,
                  epsilon_decay=0.05):
    """
        function that performs SARSA(λ)

    :param env: openAI env instance
    :param Q: ndarray, shape(s,a) containing the Q table
    :param lambtha: eligibility trace factor
    :param episodes: total number of episodes to train over
    :param max_steps: max number of steps per episode
    :param alpha: learning rate
    :param gamma: discount rate
    :param epsilon: initial threshold for epsilon greedy
    :param min_epsilon: minimum value that epsilon should decay to
    :param epsilon_decay: decay rate for updating epsilon between episodes

    :return: Q, updated Q table
    """
    epsilon_init = epsilon

    for ep in range(episodes):
        # start new episode
        state = env.reset()
        action = epsilon_greedy(state, Q, epsilon)
        eligibility = np.zeros_like(Q)

        for step in range(max_steps):
            # determine action based on policy
            next_state, reward, done, _ = env.step(action)
            next_action = epsilon_greedy(next_state, Q, epsilon)

            # TD error
            delta = (reward + (gamma * Q[next_state, next_action])
                     - Q[state, action])

            # update eligibilities
            eligibility[state, action] += 1
            eligibility *= lambtha * gamma

            # Update value function
            Q += alpha * delta * eligibility

            if done:
                break

            state = next_state
            action = next_action

        # update epsilon
        epsilon = min_epsilon + (epsilon_init - min_epsilon) *\
            np.exp(-epsilon_decay * ep)

    return Q
