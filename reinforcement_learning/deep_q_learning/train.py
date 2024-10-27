#!/usr/bin/env python3
"""
train.py

This script trains a DQN agent on Atari's Breakout using keras-rl2, Keras, and gymnasium.
The final policy network is saved as policy.h5.
"""

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras_rl.memory import SequentialMemory
from keras_rl.policy import EpsGreedyQPolicy
from keras_rl.agents import DQNAgent

# Initialize the environment with gymnasium wrappers
env = gym.make('ALE/Breakout-v5')
env = AtariPreprocessing(env)
env = FrameStack(env, num_stack=4)

nb_actions = env.action_space.n

# Build a simple neural network model
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(nb_actions, activation='linear'))

# Configure and compile the DQN agent
memory = SequentialMemory(limit=1000000, window_length=1)
policy = EpsGreedyQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1000,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])

# Train the agent
dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)

# Save the final policy model
dqn.model.save('policy.h5')
