#!/usr/bin/env python3
"""
train.py
Trains a Deep Q-Network (DQN) agent on Atari's Breakout using keras-rl2 and Gymnasium.
The trained policy is saved as 'policy.h5'.
"""

import gymnasium as gym
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Permute
from tensorflow.keras.optimizers.legacy import Adam  # Using legacy Adam optimizer for keras-rl2 compatibility
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy


class CompatibilityWrapper(gym.Wrapper):
    """
    A wrapper to adjust the environment for compatibility with keras-rl2.
    Ensures the environment returns the expected outputs and handles episode
    termination flags as required by keras-rl2.
    """

    def step(self, action):
        """
        Takes a step in the environment with the given action.

        :param action: The action to take in the environment.
        :return: Tuple of (observation, reward, done, info)
                 - observation: the resulting observation
                 - reward: the reward received from this step
                 - done: whether the episode has terminated
                 - info: additional information about the step
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return observation, reward, done, info

    def reset(self, **kwargs):
        """
        Resets the environment and returns the initial observation.

        :param kwargs: additional arguments for resetting the environment
        :return: initial observation
        """
        observation, info = self.env.reset(**kwargs)
        return observation

def create_atari_environment(env_name):
    """
    Initializes and configures an Atari environment for training.

    :param env_name: str, the name of the Atari environment to initialize
    :return: gym.Env object configured for Atari gameplay
    """
    env = gym.make(env_name, render_mode='rgb_array')
    env = gym.wrappers.AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=1, noop_max=30)
    env = CompatibilityWrapper(env)
    return env


def build_model(window_length, shape, actions):
    """
    Constructs a Convolutional Neural Network (CNN) model for DQN learning.

    :param window_length: int, number of frames stacked together as input to represent motion
    :param shape: tuple, the shape of individual frames (height, width, channels)
    :param actions: int, the number of possible actions in the environment
    :return: Sequential keras model ready for DQN training
    """
    model = Sequential()
    # Reorder input dimensions to fit keras-rl2 requirements
    model.add(Permute((2, 3, 1), input_shape=(window_length,) + shape))
    # First convolutional layer to capture spatial patterns
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu'))
    # Second convolutional layer for deeper pattern recognition
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    # Third convolutional layer to refine spatial features
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    # Flatten the feature maps into a single vector
    model.add(Flatten())
    # Fully connected layer to process combined features
    model.add(Dense(512, activation='relu'))
    # Output layer with linear activation to predict Q-values for each action
    model.add(Dense(actions, activation='linear'))
    return model

# DQN Agent Setup

if __name__ == "__main__":
    # Initialize the environment
    env = create_atari_environment('ALE/Breakout-v5')
    nb_actions = env.action_space.n  # Number of possible actions in Breakout

    # Build and configure the model
    window_length = 4  # Number of consecutive frames to form an observation
    model = build_model(window_length, env.observation_space.shape, nb_actions)

    # Define the DQN agent with a policy and memory buffer
    memory = SequentialMemory(limit=1000000, window_length=window_length)  # Replay buffer to store past experiences
    policy = EpsGreedyQPolicy()  # Epsilon-greedy policy for exploration-exploitation balance
    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        policy=policy,
        memory=memory,
        nb_steps_warmup=50000,  # Steps before training begins to populate memory
        gamma=0.99,  # Discount factor for future rewards
        target_model_update=10000,  # Interval for updating target network weights
        train_interval=4,  # Frequency of training updates
        delta_clip=1.0  # Clip error term for stability
    )
    dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])

    # Train the DQN agent on the environment
    dqn.fit(env, nb_steps=1000000, visualize=False, verbose=2)

    # Save the trained model weights
    dqn.save_weights('policy.h5', overwrite=True)

    # Close the environment to free resources
    env.close()

