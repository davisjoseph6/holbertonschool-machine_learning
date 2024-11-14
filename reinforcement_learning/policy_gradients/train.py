#!/usr/bin/env python3
"""
This module contains the train function to implement a full training loop using policy gradients.
"""

import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient

def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """
    Trains a policy-gradient-based agent on the given environment.

    Args:
        env: The environment to train on.
        nb_episodes (int): Number of episodes used for training.
        alpha (float): The learning rate.
        gamma (float): The discount factor.

    Returns:
        list of float: Scores from each episode (sum of rewards in each episode).
    """
    scores = []
    weight = np.zeros((env.observation_space.shape[0], env.action_space.n))

    for episode in range(nb_episodes):
        state, _ = env.reset()
        done = False
        episode_rewards = []
        transitions = []

        while not done:
            action, grads = policy_gradient(state, weight)
            next_state, reward, done, _, _ = env.step(action)

            episode_rewards.append(reward)
            transitions.append((grads, reward))

            state = next_state

        # Calculate the score for this episode
        score = sum(episode_rewards)
        scores.append(score)

        # Policy gradient update
        for t, (grads, reward) in enumerate(transitions):
            discount = sum([gamma ** i * r for i, r in enumerate(episode_rewards[t:])])
            weight += alpha * discount * grads

        # Print episode information
        print(f"Episode: {episode} Score: {score}")

    return scores

