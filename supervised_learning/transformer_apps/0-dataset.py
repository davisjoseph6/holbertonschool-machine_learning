#!/usr/bin/env python3
"""
Module for preparing and encoding TED HRLR translation dataset
from Portuguese to English using pre-trained tokenizers.
"""

import tensorflow_datasets as tfds
import transformers


class Dataset:
    """
    A class to load and prepare the TED HRLR translation dataset for machine
    translation from Portuguese to English.

    Attributes:
        data_train: A tf.data.Dataset object containing the training set
                    (Portuguese, English) pairs.
        data_valid: A tf.data.Dataset object containing the validation set
                    (Portuguese, English) pairs.
        tokenizer_pt: A pre-trained tokenizer for Portuguese text, adapted
                      to the training dataset.
        tokenizer_en: A pre-trained tokenizer for English text, adapted to
                      the training dataset.
    """

    def __init__(self):
        """
        Initializes the Dataset object and loads the training and validation
        datasets for Portuguese to English translation. It also initializes
        pre-trained tokenizers for Portuguese and English and trains them
        on the TED HRLR dataset.
        """
        # Load the TED HRLR Portuguese to English translation dataset
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation', as_supervised=True)

        # Initialize tokenizers for both languages by training on the dataset
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for the dataset using pre-trained models
        and adapts them to the specific dataset.

        Args:
            data (tf.data.Dataset): A dataset containing sentence pairs in
                                    Portuguese (pt) and English (en) formatted
                                    as tuples (pt, en).

        Returns:
            tuple:
                - tokenizer_pt (transformers.PreTrainedTokenizerFast): The tokenizer
                  trained on the Portuguese dataset.
                - tokenizer_en (transformers.PreTrainedTokenizerFast): The tokenizer
                  trained on the English dataset.
        """
        # Extract and decode sentences from the dataset to prepare them for tokenizer training
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

        # Train both tokenizers on the extracted sentences, setting a vocabulary size of 2^13
        tokenizer_pt = tokenizer_pt.train_new_from_iterator(pt_sentences, vocab_size=2**13)
        tokenizer_en = tokenizer_en.train_new_from_iterator(en_sentences, vocab_size=2**13)

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Encodes Portuguese and English sentences into tokens with additional
        start and end of sentence tokens.

        Args:
            pt (tf.Tensor): Tensor containing the Portuguese sentence.
            en (tf.Tensor): Tensor containing the corresponding English sentence.

        Returns:
            tuple:
                - pt_tokens (list[int]): The tokenized Portuguese sentence with start and end tokens.
                - en_tokens (list[int]): The tokenized English sentence with start and end tokens.
        """
        # Decode the tf.Tensor into strings
        pt_sentence = pt.numpy().decode('utf-8')
        en_sentence = en.numpy().decode('utf-8')

        # Get the vocabulary size of both tokenizers
        vocab_size_pt = self.tokenizer_pt.vocab_size
        vocab_size_en = self.tokenizer_en.vocab_size

        # Tokenize both sentences without adding special tokens
        pt_tokens = self.tokenizer_pt.encode(pt_sentence, add_special_tokens=False)
        en_tokens = self.tokenizer_en.encode(en_sentence, add_special_tokens=False)

        # Add start (vocab_size) and end (vocab_size + 1) tokens to the tokenized sentences
        pt_tokens = [vocab_size_pt] + pt_tokens + [vocab_size_pt + 1]
        en_tokens = [vocab_size_en] + en_tokens + [vocab_size_en + 1]

        return pt_tokens, en_tokens

