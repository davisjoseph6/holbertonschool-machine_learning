#!/usr/bin/env python3
"""
Module to let the trained agent play an episode on FrozenLake.
"""

import numpy as np


def play(env, Q, max_steps=100):
    """
    Plays an episode using the trained Q-table.
    """
    state = env.reset()[0]
    rendered_outputs = []
    total_rewards = 0

    for _ in range(max_steps):
        # Render and capture the current state of the environment
        rendered_outputs.append(env.render())

        # Choose the best action (exploit Q-table)
        action = np.argmax(Q[state])

        # Perform the action
        next_state, reward, done, _, _ = env.step(action)

        # Update total rewards
        total_rewards += reward

        # Transition to the next state
        state = next_state

        # End the episode if done
        if done:
            break

    # Ensure the final state is also rendered after the episode concludes
    rendered_outputs.append(env.render())

    return total_rewards, rendered_outputs
