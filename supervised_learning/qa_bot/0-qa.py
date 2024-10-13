#!/usr/bin/env python3

import tensorflow_hub as hub
from transformers import BertTokenizer
import numpy as np


def question_answer(question, reference):
    """
    Finds a snippet of text within a reference document to answer a question.
    """
    print("Loading model and tokenizer...")

    # Load pre-trained BERT model and tokenizer
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

    print("Tokenizing inputs...")

    # Tokenize the inputs (question and reference)
    max_len = 512  # BERT max token length
    inputs = tokenizer(question, reference, return_tensors='np', truncation=True, padding=True, max_length=max_len)

    # Getting input tensors for BERT QA model
    input_ids = inputs['input_ids'].astype(np.int32)
    input_mask = inputs['attention_mask'].astype(np.int32)
    segment_ids = np.zeros_like(input_ids, dtype=np.int32)  # All question tokens are segment 0, reference tokens are segment 1

    print("Running model inference...")

    # Get start and end logits from the BERT QA model
    outputs = model([input_ids, input_mask, segment_ids])
    start_logits, end_logits = outputs[0], outputs[1]

    print("Extracting the answer...")

    # Find the start and end index of the answer
    start_index = np.argmax(start_logits)
    end_index = np.argmax(end_logits)

    # Convert tokens back to words
    if start_index <= end_index:
        answer_tokens = input_ids[0, start_index:end_index + 1]
        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
        return answer
    else:
        return None
