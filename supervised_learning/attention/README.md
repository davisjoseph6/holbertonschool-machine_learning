# Attention Mechanism for Machine Translation

This project implements the key components of the attention mechanism for machine translation using TensorFlow. The main components are:

- **RNN Encoder**: Encodes the input sequence using a GRU-based RNN.
- **Self-Attention**: Computes attention weights to focus on important parts of the input sequence.
- **RNN Decoder**: Decodes the encoded sequence using the attention mechanism and outputs the predicted sequence.

## Directory Overview

### Files and Scripts

1. **`0-rnn_encoder.py`**
   - **Class**: `RNNEncoder`
   - **Description**: Implements an RNN encoder using a GRU cell to process input sequences.
     - The encoder is responsible for processing the input sequence and generating hidden states that are passed to the decoder.

2. **`1-self_attention.py`**
   - **Class**: `SelfAttention`
   - **Description**: Implements the self-attention mechanism as used in machine translation.
     - The self-attention mechanism computes the attention scores and generates a context vector based on the hidden states and previous decoder state.
     - This layer is essential for focusing on relevant parts of the input sequence while decoding.

3. **`2-rnn_decoder.py`**
   - **Class**: `RNNDecoder`
   - **Description**: Implements an RNN-based decoder with the addition of self-attention.
     - The decoder generates the output sequence by attending to the encoded input sequence, utilizing both previous hidden states and context vectors.

4. **`4-positional_encoding.py`**
   - **Description**: Calculates the positional encoding used in transformer models to inject information about the position of tokens in the sequence.
     - The positional encoding is crucial for transformer-based models to maintain information about the relative positions of tokens, as transformers do not inherently capture sequential information like RNNs.

---

## Requirements

- Python 3.x
- TensorFlow 2.x

---

## How to Use

### 1. **RNN Encoder**:
   - The encoder takes the input sequence, embeds it using a `tf.keras.layers.Embedding` layer, and processes it with a GRU layer.
   - Example usage:
     ```python
     from 0-rnn_encoder import RNNEncoder
     encoder = RNNEncoder(vocab=10000, embedding=256, units=512, batch=32)
     encoder.initialize_hidden_state()
     encoder_output, hidden_state = encoder(input_sequence, initial_state)
     ```

### 2. **Self-Attention**:
   - The self-attention mechanism computes the attention scores between the previous decoder state and the hidden states from the encoder.
   - Example usage:
     ```python
     from 1-self_attention import SelfAttention
     attention_layer = SelfAttention(units=512)
     context, weights = attention_layer(previous_decoder_state, encoder_hidden_states)
     ```

### 3. **RNN Decoder**:
   - The decoder takes in the embedded input, computes attention over the encoder's hidden states, and generates the next predicted token.
   - Example usage:
     ```python
     from 2-rnn_decoder import RNNDecoder
     decoder = RNNDecoder(vocab=10000, embedding=256, units=512, batch=32)
     decoder_output, next_state = decoder(input_sequence, previous_decoder_state, encoder_hidden_states)
     ```

---

## Attention Mechanism in Detail

The core idea behind attention is to allow the decoder to focus on relevant parts of the input sequence while generating the output. The attention mechanism computes a weighted sum of encoder hidden states, where the weights are determined by how much focus the decoder should place on each input token. This allows the model to capture dependencies between input and output sequences even when they are far apart.

### Key Components:

1. **RNN Encoder**: 
   - Processes the input sequence and outputs hidden states that summarize the sequence.

2. **Self-Attention**: 
   - Calculates attention weights based on the previous state of the decoder and the hidden states from the encoder. It generates a context vector as a weighted sum of encoder states.

3. **RNN Decoder**: 
   - Takes in the context vector (from self-attention) and previous decoder states to predict the next word in the sequence.

---

## Positional Encoding

Positional encodings are used to give the model information about the position of the tokens in the sequence. In models like transformers, this information is crucial because they process tokens simultaneously and do not have the inherent sequential nature of RNNs or LSTMs.

Example of usage (continued from the previous script):

```python
from 4-positional_encoding import positional_encoding
# pos_encoding = positional_encoding(sequence_length=100, embedding_dim=512)
```

## References

- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)

## Author

- **Davis Joseph**  
  [LinkedIn Profile](https://www.linkedin.com/in/davisjoseph767/)

