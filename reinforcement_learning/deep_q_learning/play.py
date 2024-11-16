#!/usr/bin/env python3
"""
    Visualize a Deep Q-Learning (DQN) agent playing Atari's Breakout
    using a trained policy. The game environment is displayed using Pygame,
    and the agent's performance is assessed over several episodes.
"""

from __future__ import division

import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Permute
import time
import pygame
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy
from rl.util import *
from rl.core import Processor
from rl.callbacks import Callback


class CompatibilityWrapper(gym.Wrapper):
    """
    Wrapper for ensuring compatibility with older versions of Gymnasium.
    Modifies the step and reset methods to provide consistent outputs.

    Attributes:
        env (gym.Env): The wrapped Gym environment.
    """

    def step(self, action):
        """
        Executes a given action in the environment.

        Args:
            action (int): The action to be performed.

        Returns:
            tuple: Contains the following:
                - observation (object): The next observation from the environment.
                - reward (float): The reward received after performing the action.
                - done (bool): True if the episode has ended, otherwise False.
                - info (dict): Additional environment-specific information.
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return observation, reward, done, info

    def reset(self, **kwargs):
        """
        Resets the environment to its initial state.

        Args:
            **kwargs: Additional arguments for the environment's reset method.

        Returns:
            observation (object): The initial observation of the environment.
        """
        observation, info = self.env.reset(**kwargs)
        return observation


def create_atari_environment(env_name):
    """
    Sets up and preprocesses an Atari environment for reinforcement learning.

    Args:
        env_name (str): The name of the Atari environment.

    Returns:
        gym.Env: The configured environment.
    """
    # Instantiate the specified Atari environment with RGB rendering mode.
    env = gym.make(env_name, render_mode='rgb_array')

    # Apply preprocessing:
    #   Resize frames to 84x84 pixels, convert observations to grayscale,
    #   implement frame skipping, add random no-op actions at the start of each episode.
    env = AtariPreprocessing(
        env,
        screen_size=84,
        grayscale_obs=True,
        frame_skip=1,
        noop_max=30
    )

    # Ensure compatibility with older versions of Gymnasium using the wrapper.
    env = CompatibilityWrapper(env)
    return env


def build_model(window_length, shape, actions):
    """
    Constructs a Convolutional Neural Network (CNN) model for processing
    stacked frames in the Atari environment.

    Args:
        window_length (int): The number of frames to stack as input.
        shape (tuple): The shape of a single input frame (height, width, channels).
        actions (int): The total number of possible actions in the environment.

    Returns:
        keras.models.Sequential: The constructed CNN model.
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


class AtariProcessor(Processor):
    """
    Custom processor for preprocessing observations, rewards, and state batches
    in the Atari environment before passing them to the DQN agent.
    """

    def process_observation(self, observation):
        """
        Converts observations into a consistent format (NumPy array).

        Args:
            observation (object): The raw observation from the environment.

        Returns:
            np.ndarray: Processed observation.
        """
        if isinstance(observation, tuple):
            observation = observation[0]
        img = np.array(observation, dtype='uint8')
        return img

    def process_state_batch(self, batch):
        """
        Normalizes pixel values in a batch of states to the range [0, 1].

        Args:
            batch (np.ndarray): Batch of states.

        Returns:
            np.ndarray: Normalized batch of states.
        """
        return batch.astype('float32') / 255.0

    def process_reward(self, reward):
        """
        Clips rewards to be within the range [-1, 1].

        Args:
            reward (float): The raw reward from the environment.

        Returns:
            float: Clipped reward.
        """
        return np.clip(reward, -1., 1.)


# PYGAME DISPLAY CALLBACK

class PygameCallback(Callback):
    """
    Callback for visualizing the agent's gameplay using Pygame.
    """

    def __init__(self, env, delay=0.02):
        """
        Initializes the PygameCallback.

        Args:
            env (gym.Env): The environment being visualized.
            delay (float): Time (in seconds) to pause between rendering frames.
        """
        self.env = env
        self.delay = delay
        pygame.init()
        self.screen = pygame.display.set_mode((420, 320))
        pygame.display.set_caption("Atari Breakout - DQN Agent")

    def on_action_end(self, action, logs={}):
        """
        Executes after an agent action and renders the frame using Pygame.

        Args:
            action (int): The action performed by the agent.
            logs (dict): Training-related logs (optional).
        """
        # Render the current environment frame
        frame = self.env.render()
        surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        surf = pygame.transform.scale(surf, (420, 320))
        self.screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Handle Pygame events (e.g., closing the window)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.env.close()
                pygame.quit()

        time.sleep(self.delay)

    def on_episode_end(self, episode, logs={}):
        """
        Executes after an episode ends, providing a short pause.

        Args:
            episode (int): The episode number that just finished.
            logs (dict): Training-related logs (optional).
        """
        pygame.time.wait(1000)


# MAIN SCRIPT

if __name__ == "__main__":
    # Create the Atari Breakout environment
    env = create_atari_environment('ALE/Breakout-v5')
    nb_actions = env.action_space.n

    # Build the CNN model for the agent
    window_length = 4
    input_shape = (84, 84)
    model = build_model(window_length, input_shape, nb_actions)

    # Load the pre-trained model weights
    model.load_weights('policy.h5')

    # Configure the DQN agent
    memory = SequentialMemory(limit=1000000, window_length=window_length)
    processor = AtariProcessor()
    policy = GreedyQPolicy()
    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        policy=policy,
        memory=memory,
        processor=processor,
        nb_steps_warmup=50000,
        gamma=0.99,
        target_model_update=10000,
        train_interval=4,
        delta_clip=1.0
    )
    dqn.compile(optimizer='adam', metrics=['mae'])

    # Test the agent's performance and visualize gameplay
    pygame_callback = PygameCallback(env, delay=0.02)
    scores = dqn.test(env, nb_episodes=5, visualize=False, callbacks=[pygame_callback])

    # Print the average score
    print('Average score over 5 test episodes:', np.mean(scores.history['episode_reward']))

    # Close the environment and Pygame
    env.close()
    pygame.quit()

