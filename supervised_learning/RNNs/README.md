# Recurrent Neural Networks (RNN) Implementation

This project demonstrates how to implement and use simple RNN, GRU, and other recurrent network architectures from scratch. The project focuses on the building blocks of recurrent neural networks (RNNs), including basic RNN cells, GRU cells, and the process of forward propagation through these cells. It includes various modules for RNN implementations and their applications.

---

## Directory Overview

### Files and Scripts

1. **`0-rnn_cell.py`**  
   - **RNNCell class**:  
     Implements the basic building block of a simple RNN. It defines the weights for the input and hidden state, and the forward propagation method.
     - **`forward(h_prev, x_t)`**: Performs forward propagation using a single time step. It computes the next hidden state (`h_next`) and the output (`y`) using the softmax activation function.

2. **`1-rnn.py`**  
   - **`rnn(rnn_cell, X, h_0)`**:  
     Implements forward propagation for a sequence of inputs through an RNN.
     - **Inputs**:
       - `rnn_cell`: An instance of the RNNCell class.
       - `X`: Input data sequence.
       - `h_0`: Initial hidden state.
     - **Outputs**:
       - Hidden states and output values for each time step in the sequence.

3. **`2-gru_cell.py`**  
   - **GRUCell class**:  
     Implements a single unit of a Gated Recurrent Unit (GRU) network. The GRU cell uses update and reset gates to control the flow of information.
     - **`forward(h_prev, x_t)`**: Computes the next hidden state (`h_next`) and the output (`y`) using the GRU mechanism.
     - **Activation Functions**:
       - `sigmoid()`: Applies the sigmoid activation function for the gates.
       - `tanh()`: Applies the tanh activation function for the intermediate state.

4. **`3-lstm_cell.py`**  
   - **LSTMCell class** (not displayed in the snippet):  
     Implements the LSTM (Long Short-Term Memory) unit, which is another type of recurrent unit that uses gates to manage long-term dependencies. It includes:
     - Forget, input, and output gates to control the flow of information.
     - Memory cell for maintaining long-term information.

5. **`4-deep_rnn.py`**  
   - Implements a deeper RNN architecture, stacking multiple RNN cells to increase the model's capacity for learning complex sequences.

6. **`5-bi_forward.py`**  
   - **Bi-directional RNN**: Implements a forward-pass for bi-directional RNN, where the data is processed both from the beginning and the end of the sequence.

7. **`6-bi_backward.py`**  
   - Implements the backward-pass for a bi-directional RNN, where the data is processed from the reverse order of the sequence.

8. **`7-bi_output.py`**  
   - Combines the forward and backward passes in a bi-directional RNN to output predictions.

9. **`8-bi_rnn.py`**  
   - **Bi-directional RNN**: Implements a full bi-directional RNN architecture using the forward and backward passes, combining the outputs from both directions.

10. **`README.md`**  
   - Provides an overview of the project, including how to use and understand the various modules.

---

## How to Use

1. **Initialize an RNN Cell**:
   - Create an instance of the `RNNCell` or `GRUCell` class:
     ```python
     rnn_cell = RNNCell(i, h, o)
     gru_cell = GRUCell(i, h, o)
     ```
     Where:
     - `i`: Number of input features.
     - `h`: Number of hidden units.
     - `o`: Number of output units.

2. **Forward Propagation**:
   - For a single time step using `RNNCell`:
     ```python
     h_next, y = rnn_cell.forward(h_prev, x_t)
     ```
   - For processing a sequence using `rnn`:
     ```python
     H, Y = rnn(rnn_cell, X, h_0)
     ```
     Where:
     - `X`: Input data sequence (3D array with shape `(time_steps, batch_size, input_size)`).
     - `h_0`: Initial hidden state (2D array with shape `(batch_size, hidden_size)`).
     - `H`: Hidden states at each time step (3D array).
     - `Y`: Outputs at each time step (2D array).

3. **GRU Cell Forward Propagation**:
   - For the GRUCell, forward propagation is similar:
     ```python
     h_next, y = gru_cell.forward(h_prev, x_t)
     ```

---

## Key Concepts

### RNNCell
- **Basic Recurrent Unit**: A simple RNN cell processes one time step at a time and updates the hidden state using a non-linear activation function (`tanh`).
- **Softmax Output**: After computing the hidden state, the cell outputs the prediction using a softmax function.

### GRUCell
- **Gated Recurrent Unit**: An advanced RNN unit with gates (update and reset) to better control the flow of information. It helps to mitigate the vanishing gradient problem in standard RNNs.

### Bi-Directional RNN
- **Forward and Backward Processing**: A bi-directional RNN processes data in both directions (forward and backward) to capture information from both past and future time steps.

---

## Requirements

- Python 3.x
- NumPy

---

## Next Steps

- You can expand this implementation to include:
  - LSTM cells for long-term sequence modeling.
  - Multi-layer RNNs for deeper architectures.
  - Fully connected layers for sequence classification tasks.

---

## Author

- **Davis Joseph**  
  [LinkedIn Profile](https://www.linkedin.com/in/davisjoseph767/)

