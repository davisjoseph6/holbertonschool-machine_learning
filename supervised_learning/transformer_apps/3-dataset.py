#!/usr/bin/env python3
"""
Module for preparing, encoding, and setting up a data pipeline for TED HRLR
translation dataset from Portuguese to English using pre-trained tokenizers
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import transformers


class Dataset:
    """
    A class to load, prepare, and encode the TED HRLR translation
    dataset for machine translation from Portuguese to English.
    """

    def __init__(self, batch_size, max_len):
        """
        Initializes the Dataset object, loads the training and validation
        datasets, and sets up the data pipeline.
        """
        # Load the Portuguese to English translation dataset
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation', as_supervised=True)

        # Initialize tokenizers for both languages by training on the dataset
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
                self.data_train)

        # Set up the training data pipeline
        self.data_train = self.data_train.map(
                self.tf_encode, num_parallel_calls=tf.data.AUTOTUNE)
        self.data_train = self.data_train.filter(
                lambda pt,
                en: tf.logical_and(
                    tf.size(pt) <= max_len, tf.size(en) <= max_len)
                )
        self.data_train = self.data_train.cache()
        self.data_train = self.data_train.shuffle(buffer_size=20000)
        self.data_train = self.data_train.padded_batch(
                batch_size, padded_shapes=([None], [None]))
        self.data_train = self.data_train.prefetch(
                buffer_size=tf.data.AUTOTUNE)

        # Set up the validation data pipeline
        self.data_valid = self.data_valid.map(
                self.tf_encode, num_parallel_calls=tf.data.AUTOTUNE)
        self.data_valid = self.data_valid.filter(
                lambda pt,
                en: tf.logical_and(
                    tf.size(pt) <= max_len, tf.size(en) <= max_len))
        self.data_valid = self.data_valid.padded_batch(
                batch_size, padded_shapes=([None], [None])
                )

    def tokenize_dataset(self, data):
        """
        Tokenizes the dataset using pre-trained tokenizers and adapts them to
        the datatset.
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

        # Train the tokenizers on the dataset sentence iterators
        tokenizer_pt = tokenizer_pt.train_new_from_iterator(pt_sentences,
                                                            vocab_size=2**13)
        tokenizer_en = tokenizer_en.train_new_from_iterator(en_sentences,
                                                            vocab_size=2**13)

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Encodes a translation pair into tokenized sentences.
        """
        # Decode the Tensors to strings
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
        pt_tokens = [self.tokenizer_pt.vocab_size] + pt_tokens + \
                    [self.tokenizer_pt.vocab_size + 1]
        en_tokens = [self.tokenizer_en.vocab_size] + en_tokens + \
                    [self.tokenizer_en.vocab_size + 1]

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

        # Set the shape of the tokenized tensors
        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])

        return pt_tokens, en_tokens
