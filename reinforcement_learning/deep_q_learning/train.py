#!/usr/bin/env python3
"""
train.py
Trains a DQN agent using keras-rl2 and Gymnasium on Atari's Breakout environment.
Saves the trained policy network as policy.h5.
"""

import gymnasium as gym
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Permute
from tensorflow.keras.optimizers.legacy import Adam  # Use legacy Adam optimizer for compatibility
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy

#####################################
#            Setup ENV              #
#####################################

class CompatibilityWrapper(gym.Wrapper):
    """
    Compatibility wrapper for gymnasium env to ensure compatibility with keras-rl2.
    """

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return observation, reward, done, info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return observation

def create_atari_environment(env_name):
    """
    Create and configure an Atari env for reinforcement learning.

    :param env_name: name of the Atari env
    :return: gym.Env: configured Atari env
    """
    env = gym.make(env_name, render_mode='rgb_array')
    env = gym.wrappers.AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=1, noop_max=30)
    env = CompatibilityWrapper(env)
    return env

#####################################
#            CNN model              #
#####################################

def build_model(window_length, shape, actions):
    """
    Build a CNN model for reinforcement learning.

    :param window_length: int, number of frames to stack as input
    :param shape: tuple, shape of the input img (height, width, channels)
    :param actions: int, number possible actions in env
    :return: keras.models.Sequential: compiled keras model
    """
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=(window_length,) + shape))
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

#####################################
#              AGENT                #
#####################################

if __name__ == "__main__":
    # 1. CREATE ENV
    env = create_atari_environment('ALE/Breakout-v5')
    nb_actions = env.action_space.n

    # 2. BUILD MODEL
    window_length = 4
    model = build_model(window_length, env.observation_space.shape, nb_actions)

    # 3. DEFINE AGENT
    memory = SequentialMemory(limit=1000000, window_length=window_length)
    policy = EpsGreedyQPolicy()
    dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
                   nb_steps_warmup=50000, gamma=0.99, target_model_update=10000, train_interval=4, delta_clip=1.0)
    dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])

    # 4. TRAIN MODEL
    dqn.fit(env, nb_steps=1000000, visualize=False, verbose=2)

    # 5. SAVE MODEL
    dqn.save_weights('policy.h5', overwrite=True)

    # 6. CLOSE ENVIRONMENT
    env.close()

