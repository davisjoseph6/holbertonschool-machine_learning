#!/usr/bin/env python3
"""
play.py

This script loads the policy network saved in policy.h5 and uses it to display
a game of Atari's Breakout using a GreedyQPolicy.
"""

import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack
from keras.models import load_model
from keras_rl.policy import GreedyQPolicy
from keras_rl.agents import DQNAgent
from keras_rl.memory import SequentialMemory

# Initialize the environment with gymnasium wrappers
env = gym.make('ALE/Breakout-v5')
env = AtariPreprocessing(env)
env = FrameStack(env, num_stack=4)

nb_actions = env.action_space.n

# Load the trained model
model = load_model('policy.h5')

# Configure the agent with a greedy policy
memory = SequentialMemory(limit=1000000, window_length=1)
policy = GreedyQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, policy=policy)
dqn.compile()

# Play the game
dqn.test(env, nb_episodes=5, visualize=True)
