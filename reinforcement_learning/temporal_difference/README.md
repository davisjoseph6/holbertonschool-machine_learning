# Temporal Difference Learning

This project covers various algorithms within the field of Reinforcement Learning, focusing on Temporal Difference (TD) learning methods, including Monte Carlo, TD(λ), and SARSA(λ). These methods are implemented and tested on environments like `FrozenLake` to help understand value function estimation and action-value methods.

## Project Structure

The project files are organized as follows:

- `0-monte_carlo.py`: Implements the **Monte Carlo** algorithm for estimating state values.
- `1-td_lambtha.py`: Implements the **TD(λ)** algorithm using eligibility traces for value function estimation.
- `2-sarsa_lambtha.py`: Implements the **SARSA(λ)** algorithm with eligibility traces, which estimates a Q-table using an epsilon-greedy policy.

Each file includes a main file (`0-main.py`, `1-main.py`, and `2-main.py`) to test the functionality of each algorithm.

## Files

- **0-monte_carlo.py**  
  - **Description**: Implements the Monte Carlo method to estimate the value function by averaging returns received after visiting each state. Uses a specified policy to determine actions within the environment.
  - **Function**: `monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99)`
  
- **1-td_lambtha.py**  
  - **Description**: Implements the TD(λ) algorithm, which uses eligibility traces to update value estimates incrementally. This allows recent states to have more influence, making it a balance between Monte Carlo and TD methods.
  - **Function**: `td_lambtha(env, V, policy, lambtha=0.9, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99)`
  
- **2-sarsa_lambtha.py**  
  - **Description**: Implements the SARSA(λ) algorithm, which combines SARSA with eligibility traces to update a Q-table. It uses an epsilon-greedy policy for action selection and an exponentially decaying epsilon to balance exploration and exploitation.
  - **Function**: `sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05)`

## Requirements

Install the necessary libraries using:

```bash
pip install -r requirements.txt
```

The primary dependencies are:

- `numpy`: Used for mathematical operations and array manipulations.
- `gymnasium`: Provides various reinforcement learning environments, such as `FrozenLake`.

## Usage
To run each algorithm, execute the respective main file. For example:

```bash
./0-main.py
```
The output will display the updated value estimates or Q-table after training, reflecting the policy learned by each algorithm.

### Running the Project
Each main file initializes an environment and tests the algorithm’s functionality. Example usage:

```bash
chmod +x 0-main.py  # Make the script executable
./0-main.py         # Run the Monte Carlo algorithm
```

## Algorithms Overview

1. **Monte Carlo**: Estimates state values based on averaging complete episode returns.
2. **TD(λ)**: Combines TD learning with eligibility traces to balance between Monte Carlo and standard TD methods.
3. **SARSA(λ)**: A Q-learning-based approach with eligibility traces, using epsilon-greedy for exploration.

## License
This project is intended for educational purposes under the Holberton School Machine Learning curriculum.
