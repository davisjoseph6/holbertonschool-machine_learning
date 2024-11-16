#!/usr/bin/env python3
"""
    Display a game played by the agent trained on Atari's Breakout
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


#####################################
#            Setup ENV              #
#####################################
class CompatibilityWrapper(gym.Wrapper):
    """
        Compatibility wrapper for gym env to ensure
        compatibility with older versions of gym
    """

    def step(self, action):
        """
            take a step in the env using the given action

        :param action: action to be taken in env

        :return: tuple containing
            - observation: obs from env after action taken
            - reward: reward obtain after taking the action
            - done: bool indicating whether episode has ended
            - info: additional information from the env
        """
        observation, reward, terminated, truncated, info = (
            self.env.step(action))
        done = terminated or truncated
        return observation, reward, done, info

    def reset(self, **kwargs):
        """
            reset env and return the initial obs

        :param kwargs: additional args

        :return: observation: initial obs of the env
        """
        observation, info = self.env.reset(**kwargs)
        return observation


def create_atari_environment(env_name):
    """
        Create and configure an Atari env for reinforcement learning

    :param env_name: name of the Atari env

    :return: gym.Env: configured Atari env
    """
    # Create specified Atari env with RDB rendering mode
    env = gym.make(env_name, render_mode='rgb_array')
    # Apply preprocessing to the env
    # - Resize the screen to 84x84
    # - Convert observations to grayscale
    # - Apply frame skipping
    # - Apply a random number of no-ops at the start of each episode
    env = AtariPreprocessing(env,
                             screen_size=84,
                             grayscale_obs=True,
                             frame_skip=1,
                             noop_max=30)
    # Wrap the environment to ensure compatibility with older versions of gym
    env = CompatibilityWrapper(env)
    return env


#####################################
#            CNN model              #
#####################################

def build_model(window_length, shape, actions):
    """
        Build a CNN model for reinforcement learning

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

class AtariProcessor(Processor):
    """
        Custom processor class for Atari env
        to handle obs and rewards before passing to DQN agent
    """

    def process_observation(self, observation):
        """
            Process the obs by convert in numpy array

        :param observation: (object) obs from env

        :return: processed obs
        """
        if isinstance(observation, tuple):
            observation = observation[0]

        img = np.array(observation)
        img = img.astype('uint8')
        return img

    def process_state_batch(self, batch):
        """
            Process a batch of states by normalizing the pixel values

        :param batch: ndarray, batch of states

        :return: ndarray, processed batch of states
        """
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        """
            Clip the reward to be within the range [-1,1]

        :param reward: float, reward from the env

        :return: clipped reward
        """
        return np.clip(reward, -1., 1.)


class PygameCallback(Callback):
    """
        Callback class to display the game played by the agent
        using Pygame
    """

    def __init__(self, env, delay=0.02):
        """
            Initializes the PygameCallback instance.
            Initializes Pygame and sets up the display window.

        :param env: gym env instance
        :param delay: in second, between rendering frames
        """
        self.env = env
        self.delay = delay
        # Initialize Pygame and set up the display window
        pygame.init()
        self.screen = pygame.display.set_mode((420, 320))
        pygame.display.set_caption("Atari Breakout - DQN Agent")

    def on_action_end(self, action, logs={}):
        """
            Callback function triggered after an action is taken by the agent.

        :param action: action taken by the agent
        :param logs: dict, logs from training process

        :return:Renders the current frame,
                updates the display,
                and handles pygame events.
        """
        # Render the current frame from the environment
        frame = self.env.render()
        # Convert the frame to a Pygame surface
        surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        # Scale the surface to fit the display window
        surf = pygame.transform.scale(surf, (420, 320))
        # Blit (draw) the surface onto the screen
        self.screen.blit(surf, (0, 0))
        # Update the display
        pygame.display.flip()

        # Check for Pygame events (like closing the window)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.env.close()
                pygame.quit()

        # Introduce a delay to control frame rate
        time.sleep(self.delay)

    def on_episode_end(self, episode, logs={}):
        """
            Callback function triggered after an episode ends.

        :param episode: episode number that just ended
        :param logs: dict containing logs from training process

        :return: waits for 1 second between episodes
        """
        pygame.time.wait(1000)


if __name__ == "__main__":
    # 1. CREATE ENV
    env = create_atari_environment('ALE/Breakout-v5')
    nb_actions = env.action_space.n

    # 2. BUILD MODEL
    window_length = 4
    input_shape = (84, 84)
    model = build_model(window_length, input_shape, nb_actions)

    # 3. LOAD TRAINED WEIGHTS
    model.load_weights('policy.h5')

    # 4. CONFIGURE AGENT
    memory = SequentialMemory(limit=1000000,
                              window_length=window_length)
    processor = AtariProcessor()
    policy = GreedyQPolicy()

    dqn = DQNAgent(model=model,
                   nb_actions=nb_actions,
                   policy=policy,
                   memory=memory,
                   processor=processor,
                   nb_steps_warmup=50000,
                   gamma=.99,
                   target_model_update=10000,
                   train_interval=4,
                   delta_clip=1.)
    dqn.compile(optimizer='adam',
                metrics=['mae'])

    # 5. TEST AGENT
    pygame_callback = PygameCallback(env, delay=0.02)
    scores = dqn.test(env, nb_episodes=5,
                      visualize=False,
                      callbacks=[pygame_callback])

    # 6. DISPLAY RESULT
    print('Average score over 5 test episodes:',
          np.mean(scores.history['episode_reward']))

    # 7. CLOSE ENV AND PYGAME
    env.close()
    pygame.quit()
