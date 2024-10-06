#!/usr/bin/env python3

import tensorflow_datasets as tfds
from transformers import BertTokenizerFast


class Dataset:
    """
    Dataset class for loading and preparing the TED HRLR Portuguese to English
    dataset.
    """

    def __init__(self):
        """
        Class constructor for initializing the dataset and tokenizers.
        """
        # Load the training and validation dataset splits
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en', split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en', split='validation', as_supervised=True)

        # Tokenizers for Portuguese and English text
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for the dataset using pre-trained BERT models.
        """
        # Load pre-trained BERT tokenizers for both languages
        tokenizer_pt = BertTokenizerFast.from_pretrained('neuralmind/bert-base-portuguese-cased')
        tokenizer_en = BertTokenizerFast.from_pretrained('bert-base-uncased')

        return tokenizer_pt, tokenizer_en
