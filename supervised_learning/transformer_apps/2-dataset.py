#!/usr/bin/env python3
"""
Module for preparing and encoding TED HRLR translation dataset
from Portuguese to English using pre-trained tokenizers and TensorFlow.
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import transformers


class Dataset:
    """
    A class to load, prepare, and encode the TED HRLR translation
    dataset for machine translation from Portuguese to English.
    """

    def __init__(self):
        """
        Initializes the Dataset object and loads the training and validation
        datasets for Portuguese to English translation.
        """
        # Load the TED HRLR Portuguese to English translation dataset
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation', as_supervised=True)

        # Initialize tokenizers for both languages by training on the dataset
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
                self.data_train)

        # Tokenize the dataset splits using the tf_encode method
        self.data_train = self.data_train.map(
                self.tf_encode, num_parallel_calls=tf.data.AUTOTUNE)
        self.data_valid = self.data_valid.map(
                self.tf_encode, num_parallel_calls=tf.data.AUTOTUNE)

    def tokenize_dataset(self, data):
        """
        Tokenizes the dataset using pre-trained tokenizers and adapts them to
        the specific datatset.
        """
        # Extract and decode sentences from the dataset to prepare them for
        # tokenizer training
        pt_sentences = []
        en_sentences = []
        for pt, en in data.as_numpy_iterator():
            pt_sentences.append(pt.decode('utf-8'))
            en_sentences.append(en.decode('utf-8'))

        # Load pre-trained BERT tokenizers for Portuguese and English
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
                'neuralmind/bert-base-portuguese-cased', use_fast=True,
                clean_up_tokenization_spaces=True)
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
                'bert-base-uncased', use_fast=True,
                clean_up_tokenization_spaces=True)

        # Train both tokenizers on the dataset sentence iterators
        tokenizer_pt = tokenizer_pt.train_new_from_iterator(pt_sentences,
                                                            vocab_size=2**13)
        tokenizer_en = tokenizer_en.train_new_from_iterator(en_sentences,
                                                            vocab_size=2**13)

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Encodes a translation pair into tokenized sentences.
        """
        # Decode TensorFlow Tensors to strings
        pt_sentence = pt.numpy().decode('utf-8')
        en_sentence = en.numpy().decode('utf-8')

        # Get the vocab_size for both tokenizers
        vocab_size_pt = self.tokenizer_pt.vocab_size
        vocab_size_en = self.tokenizer_en.vocab_size

        # Tokenize the sentences without adding special tokens
        pt_tokens = self.tokenizer_pt.encode(pt_sentence,
                                             add_special_tokens=False)
        en_tokens = self.tokenizer_en.encode(en_sentence,
                                             add_special_tokens=False)

        # Add start (vocab_size) and end (vocab_size + 1) tokens
        # to the tokenized sentences
        pt_tokens = [vocab_size_pt] + pt_tokens + [vocab_size_pt + 1]
        en_tokens = [vocab_size_en] + en_tokens + [vocab_size_en + 1]

        # Return the tokens as numpy arrays
        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        A TensorFlow wrapper for the encode method.
        """
        # Use tf.py_function to apply the encode method in TensorFlow
        pt_tokens, en_tokens = tf.py_function(func=self.encode,
                                              inp=[pt, en],
                                              Tout=[tf.int64, tf.int64])

        # Set the shapes of the tokenized tensors
        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])

        return pt_tokens, en_tokens
