# Transformer Applications for Machine Translation

This project demonstrates how to implement a machine translation model using a Transformer architecture. Specifically, it utilizes the TED HRLR (Portuguese to English) translation dataset to train a model that can translate sentences from Portuguese to English.

---

## Directory Overview

### Files and Scripts

1. **`0-dataset.py` to `4-dataset.py`**
   - These scripts contain different stages of the data preparation pipeline for the TED HRLR dataset, from loading and tokenizing the data to preparing it for model training.

2. **`4-create_masks.py`**
   - Contains the code for creating the masks that will be used in the Transformer model. Masks help the model focus on specific tokens during training.

3. **`5-main.py`**
   - The main entry point for training the Transformer model. It initializes the dataset, model, and training loop.

4. **`5-transformer.py`**
   - Contains the Transformer architecture implementation, including the encoder, decoder, and attention mechanisms.

5. **`README.md`**
   - This file, which provides an overview of the project.

6. **`load_dataset.py`**
   - A script for loading and preprocessing the dataset for training.

---

## Dataset: TED HRLR Translation Dataset

The TED HRLR dataset contains pairs of sentences in Portuguese and English. The dataset is used for training machine translation models. In this project, the focus is on training a model to translate from Portuguese to English.

- **Dataset Name**: TED HRLR
- **Language Pairs**: Portuguese → English
- **Source**: [TensorFlow Datasets - TED HRLR](https://www.tensorflow.org/datasets/catalog/ted_hrlr_translate#ted_hrlr_translatept_to_en)

---

## Key Classes and Functions

### `Dataset` Class

The `Dataset` class is used to load, prepare, and encode the TED HRLR translation dataset for machine translation. It performs the following tasks:
1. Loads the training and validation datasets from TensorFlow Datasets.
2. Tokenizes the dataset using pre-trained tokenizers for Portuguese and English.
3. Creates a data pipeline for efficient training using TensorFlow's `tf.data` API.

#### Key Methods:
- **`__init__(batch_size, max_len)`**: Initializes the `Dataset` object, loads the data, and sets up the data pipeline.
- **`tokenize_dataset(data)`**: Tokenizes the dataset using pre-trained BERT tokenizers for both Portuguese and English.
- **`encode(pt, en)`**: Encodes a Portuguese-English sentence pair into tokenized sentences.
- **`tf_encode(pt, en)`**: TensorFlow wrapper for the `encode` method, used in the data pipeline.

---

## Model Architecture: Transformer

The Transformer architecture is designed to handle sequences of data, such as sentences for machine translation. It uses attention mechanisms to process sequences in parallel, unlike traditional RNNs, which process sequences step-by-step.

In this project, the Transformer model is implemented in `5-transformer.py`. The model consists of:
- **Encoder**: Processes the input sequence (Portuguese sentence).
- **Decoder**: Generates the output sequence (English translation).
- **Attention Mechanism**: Enables the model to focus on different parts of the input sequence when generating the output.

---

## Training the Model

The model is trained using the `5-main.py` script. It does the following:
1. Loads the preprocessed data.
2. Initializes the Transformer model.
3. Trains the model using the training dataset.
4. Evaluates the model using the validation dataset.

---

## Requirements

- Python 3.x
- TensorFlow 2.x
- Hugging Face Transformers
- TensorFlow Datasets

---

## How to Run

1. **Prepare the Dataset**:
   - Run `3-dataset.py` to prepare and encode the TED HRLR translation dataset.
     ```bash
     python3 3-dataset.py
     ```

2. **Create the Model**:
   - Run `5-transformer.py` to implement the Transformer architecture.
     ```bash
     python3 5-transformer.py
     ```

3. **Train the Model**:
   - Run `5-main.py` to train the model.
     ```bash
     python3 5-main.py
     ```

4. **Evaluate and Test**:
   - After training, you can evaluate the model's performance and test it on new sentence pairs.

---

## Applications

- **Machine Translation**: This project is primarily focused on translating sentences from Portuguese to English.
- **Natural Language Processing (NLP)**: The Transformer model can be adapted to other NLP tasks, such as text summarization, question answering, and more.

---

## References

- [Attention Is All You Need (Transformer Paper)](https://arxiv.org/abs/1706.03762)
- [TED HRLR Translation Dataset](https://www.tensorflow.org/datasets/catalog/ted_hrlr_translate#ted_hrlr_translatept_to_en)

---

## Author

- **Davis Joseph**  
  [LinkedIn Profile](https://www.linkedin.com/in/davisjoseph767/)

