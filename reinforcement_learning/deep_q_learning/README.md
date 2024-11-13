# Deep Q-Learning on Atari's Breakout

**Note**: _an_ ***advanced project setup guide*** _(virtual enviromentments/Google colab, etc.) is given below_

This project implements a Deep Q-Learning (DQN) agent trained on Atari's Breakout environment using `keras-rl2` and `Gymnasium`. The project consists of two main scripts:

1. **`train.py`**: Trains a DQN agent and saves the trained model.
2. **`play.py`**: Loads the trained model and demonstrates the agent's gameplay in a simulated environment.

## Project Structure

. ├── train.py # Script to train the DQN agent 
. ├── play.py # Script to visualize the trained agent playing Breakout 
  ├── policy.h5 # Saved model weights after training 
  ├── training_history.pkl # Pickle file containing training history (optional) 
  ├── training_performance.png # Visualization of training performance (optional) 
  ├── requirements.txt # List of required packages 
  └── README.md # Project documentation

## Requirements

Install the required Python packages using:
```bash
pip install -r requirements.txt
```

## Key Dependencies
- gymnasium==0.29.1
- tensorflow==2.15
- keras-rl2
- pygame (for visualizing gameplay)

## Usage
1. Training the DQN Agent
To train the agent from scratch, run:

```bash
python3 train.py
```

This script will:

- Create a preprocessed Atari Breakout environment.
- Build and compile a Convolutional Neural Network (CNN) for Deep Q-Learning.
- Train the agent using the DQN algorithm and save the trained model weights in policy.h5.

2. Visualizing Gameplay
To see the trained agent play Breakout, run:

```bash
python3 play.py
```
This script will:

- Load the trained model weights from policy.h5.
- Display the agent playing the game using pygame.

## Code Overview
`train.py`
- Environment Setup: The Atari environment is created using gymnasium with preprocessing for frame resizing, grayscale conversion, and frame skipping.
- Model Architecture: A Convolutional Neural Network (CNN) is built to handle the visual input from the Atari environment.
- Agent Configuration: A DQN agent is created with an epsilon-greedy policy for exploration.
- Training: The agent is trained using one million steps and the model weights are saved in policy.h5.

`play.py`
- Model Loading: Loads the saved weights from policy.h5.
- Gameplay Visualization: Uses a custom callback (PygameCallback) to render each frame and visualize the agent's actions in real-time.

## Results
After training, the DQN agent should be able to achieve a reasonable score in Atari's Breakout. training_performance.png and training_history.pkl (optional) can be used to analyze training performance and evaluate model improvements.

# References
Keras-RL2 Documentation
Gymnasium Documentation
Deep Q-Learning Algorithm

This project demonstrates how to train and deploy a DQN-based reinforcement learning agent in a classic Atari environment. Feel free to tweak hyperparameters, model architecture, and policies to experiment with improved performance.

# Advanced Project Setup Guide

## Requirements
This setup requires Python version 3.10.12. If you have a newer version of Python installed, you can create a virtual environment to ensure compatibility.

## Steps

### 1. Create a Virtual Environment
It is recommended to use a virtual environment if you have a newer version of Python installed. Use the following commands to set up a virtual environment with Python 3.10.12:

```bash
sudo apt install python3.10-venv
python3.10 -m venv myvenv
source myvenv/bin/activate
```

### 2. Install CMake
CMake is needed for some dependencies. If it's not already installed, run:

```bash
sudo apt install cmake -y
```

### 3. Install Project Dependencies
Install all required packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Modifying the `rl/callbacks.py` File
In some cases, you may need to modify the `callbacks.py` file in the `rl` package to ensure compatibility with your TensorFlow setup.

If you are not using a virtual environment: Locate the `callbacks.py` file in the `rl` package using the `find` command:

```bash
find /usr/local/lib/ -name "callbacks.py" 2>/dev/null
```

If you are using a virtual environment: The `callbacks.py` file can be found here:

```bash
myvenv/lib/python3.10/site-packages/rl/callbacks.py
```

If you are using Google Colab: Locate `callbacks.py` with the following command:

```bash
!find / -name "callbacks.py" 2>/dev/null
```

## Editing `callbacks.py`
Locate the following line in `callbacks.py`:

```python
from tensorflow.keras import __version__ as KERAS_VERSION
```

Replace it with the following code to use an alternative import method:

```python
try:
    from tensorflow import __version__ as KERAS_VERSION
except ImportError:
    KERAS_VERSION = '2.x'  # Replace '2.x' with your TensorFlow version if needed
```

## Running the Training Script
Once setup is complete, run the training script with:

```bash
./train.py
```
