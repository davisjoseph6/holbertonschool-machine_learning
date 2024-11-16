# Q-Learning for FrozenLake Environment

This project demonstrates the implementation of Q-learning for training an agent to navigate the **FrozenLake** environment from **Gymnasium**. The agent learns to navigate the environment by exploring and exploiting actions based on the Q-values stored in a Q-table.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Modules](#modules)
  - [0-load_env.py](#0-load-envpy)
  - [1-q_init.py](#1-q_initpy)
  - [2-epsilon_greedy.py](#2-epsilon_greedypy)
  - [3-q_learning.py](#3-q_learningpy)
  - [4-play.py](#4-playpy)
- [How to Run](#how-to-run)
- [Requirements](#requirements)
- [Author](#author)

---

## Project Overview

This project implements Q-learning to train an agent to solve the **FrozenLake-v1** environment. The agent explores the environment with an epsilon-greedy policy, adjusts its Q-values based on rewards, and uses the Q-table to select the optimal actions during gameplay.

---

## Modules

### `0-load_env.py`
This module is responsible for loading the **FrozenLake** environment. It provides a function to load the environment with customizable parameters like the lake map, slipperiness, and rendering mode.

- **Function**: `load_frozen_lake(desc=None, map_name=None, is_slippery=False)`

### `1-q_init.py`
This module initializes the Q-table, a matrix where the agent's action values are stored. The Q-table is initialized as a numpy array of zeros.

- **Function**: `q_init(env)`

### `2-epsilon_greedy.py`
This module implements the **epsilon-greedy** policy. It provides a function to choose an action based on the current state and epsilon value, balancing exploration and exploitation.

- **Function**: `epsilon_greedy(Q, state, epsilon)`

### `3-q_learning.py`
This module contains the core Q-learning algorithm. It trains the agent using the Q-learning formula, updating the Q-table over multiple episodes and adjusting epsilon for exploration decay.

- **Function**: `train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05)`

### `4-play.py`
Once the agent is trained, this module allows the agent to play an episode by selecting actions based on the trained Q-table. The agent's actions are based purely on the exploitation of the Q-table (greedy policy).

- **Function**: `play(env, Q, max_steps=100)`

---

## How to Run

1. Clone the repository.
2. Install the required dependencies by running:
   ```bash
   pip install gymnasium numpy
   ```
3. Start by loading the FrozenLake environment:
   ```python
   from load_env import load_frozen_lake
   env = load_frozen_lake()
   ```
4. Initialize the Q-table using:
   ```python
   from q_init import q_init
   Q = q_init(env)
   ```
5. Train the agent using Q-learning:
   ```python
   from q_learning import train
   Q, total_rewards = train(env, Q)
   ```
6. Let the trained agent play and evaluate:
   ```python
   from play import play
   total_rewards, rendered_outputs = play(env, Q)
   ```

## Requirements
- Python 3.x
- gymnasium library for environment simulation
- numpy library for numerical computations

You can install the dependencies by running the following command:
```bash
pip install gymnasium numpy
```

## Author
Davis Joseph
