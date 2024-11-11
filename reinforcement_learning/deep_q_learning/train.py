#!/usr/bin/env python3
"""
    Train agent that can play Atari's Breakout
"""
from __future__ import division
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
from keras.optimizers.legacy import Adam
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Permute
import matplotlib.pyplot as plt
import pickle

from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.util import *
from rl.core import Processor

#####################################
#         Setup PARAMETER           #
#####################################

# Configuration parameters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (
        epsilon_max - epsilon_min
)  # Rate at which to reduce chance of random action being taken
batch_size = 32  # Size of batch taken from replay buffer
max_steps_per_episode = 10000
max_episodes = 10  # Limit training episodes, will run until solved
                    # if smaller than 1


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


if __name__ == "__main__":
    # 1. CREATE ENV

    env = create_atari_environment('ALE/Breakout-v5')
    # Reset env
    observation = env.reset()

    # Visualise one frame
    plt.imshow(observation, cmap='gray')
    plt.title("Initial Observation")
    plt.axis('off')
    plt.show()

    # get number of possible actions
    nb_actions = env.action_space.n

    # 2. BUILD MODEL

    window_length = 4
    model = build_model(window_length, observation.shape, nb_actions)

    # 3. DEFINE AGENT

    # Define sequential memory to store agent's experience
    memory = SequentialMemory(limit=1000000,
                              window_length=window_length)

    # Define processor to preprocess obs, states and reward
    processor = AtariProcessor()

    # Define an epsilon-greedy policy with linear annealing
    # for exploration-exploitation trade-off
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(),
                                  attr='eps',
                                  value_max=1.,
                                  value_min=.1,
                                  value_test=.05,
                                  nb_steps=1000000)

    # Define a DQN agent with specified components and parameters
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

    # Compile the DQN agent with Adam optimizer and mae as metrics
    dqn.compile(Adam(learning_rate=0.00025),
                metrics=['mae'])

    # 4. TRAIN MODEL

    history = dqn.fit(env, nb_steps=1000000, visualize=False, verbose=2)

    # 5. SAVE MODEL & HISTORY

    dqn.save_weights('policy.h5', overwrite=True)

    with open('training_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)

    # Visualise performance
    plt.figure(figsize=(14, 7))
    plt.subplot(2, 1, 1)
    plt.plot(history.history['episode_reward'])
    plt.title('Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    plt.tight_layout()
    plt.savefig('training_performance.png')
    plt.show()

    # 6. TEST

    test_env = gym.make('ALE/Breakout-v5', render_mode='human')
    test_env = AtariPreprocessing(test_env,
                                  screen_size=84,
                                  grayscale_obs=True,
                                  frame_skip=4,
                                  noop_max=30)

    scores = dqn.test(test_env,
                      nb_episodes=10,
                      visualize=True)
    print('Average score over 10 test episodes:',
          np.mean(scores.history['episode_reward']))

    # 7. Close env
    env.close()
