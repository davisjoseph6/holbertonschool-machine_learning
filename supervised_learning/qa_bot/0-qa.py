#!/usr/bin/env python3
"""
Question Answering with pretrained BERT
"""

import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """
    Finds a snippet of text within a reference document to answer a question.
    """

    print("Initializing BERT Tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(
            "bert-large-uncased-whole-word-masking-finetuned-squad")

    print("Loading BERT model from TensorFlow Hub...")
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    # Tokenize the inputs using the BERT tokenizer
    print("Tokenizing the question and reference document...")
    max_len = 512  # BERT max token length
    inputs = tokenizer(question, reference, return_tensors="tf")

    # Prepare the input tensors for the TensorFlow Hub model
    input_tensors = [
            inputs["input_ids"],      # Token IDs
            inputs["attention_mask"], # mask for padding tokens
            inputs["token_type_ids"]  # Token type IDs to distinguish question from context
            ]

    print("Running inference on the model...")
    # Pass the input tensors to the BERT QA model and get start and end logits
    output = model(input_tensors)

    # Access the start and end logits
    start_logits = output[0]
    end_logits = output[1]

    # Get the input sequence length
    sequence_length = inputs["input_ids"].shape[1]
    print(f"Input sequence length: {sequence_length}")

    # Find the best start and end indices within the input sequence
    print("Determining the best start and end indices for the answer...")
    start_index = tf.math.argmax(start_logits[0, 1:sequence_length - 1]) + 1
    end_index = tf.math.argmax(end_logits[0, 1:sequence_length - 1]) + 1
    print(f"Start index: {start_index}, End index: {end_index}")

    # Get the answer tokens using the best indices
    print("Extracting the answer tokens...")
    answer_tokens = inputs["input_ids"][0][start_index: end_index + 1]

    # Decode the answer tokens to get the final answer
    print("Decoding the answer tokens...")
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True,
                              clean_up_tokenization_spaces=True)

    # If no answer is found (i.e., empty or whitespace answer), return None
    if not answer.strip():
        print("No valid answer found.")
        return None

    print(f"Answer: {answer}")
    return answer
