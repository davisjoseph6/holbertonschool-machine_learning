#!/usr/bin/env python3

"""
play.py: Loads the policy network from policy.h5 and plays Atari's Breakout using GreedyQPolicy.
Displays the gameplay in the environment.
"""

import gymnasium as gym
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy
from gymnasium.wrappers import AtariPreprocessing, FrameStack

# Create the environment with wrappers for compatibility
env = FrameStack(AtariPreprocessing(gym.make("ALE/Breakout-v5", render_mode="human")), num_stack=4)
nb_actions = env.action_space.n

# Build the model (structure must match the training model)
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(64, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(nb_actions, activation="linear"))

# Configure the agent
memory = SequentialMemory(limit=1000000, window_length=1)
policy = GreedyQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, policy=policy)

# Load the weights from the saved policy network
dqn.load_weights("policy.h5")

# Play a game
dqn.test(env, nb_episodes=5, visualize=True)
env.close()

