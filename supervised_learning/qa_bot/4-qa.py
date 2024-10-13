#!/usr/bin/env python3
"""
Question answering system using semantic search and BERT.
"""

import os
from sentence_transformers import SentenceTransformer
import numpy as np

qa_module = __import__('0-qa')
extract_answer = qa_module.question_answer


def cosine_similarity(vec1, vec2):
    """
    Compute the cosine similarity between two vectors.
    """
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


def load_corpus(corpus_path):
    """
    Loads the text files from the corpus_path directory into a list of strings.
    """
    corpus = []
    for filename in os.listdir(corpus_path):
        file_path = os.path.join(corpus_path, filename)
        if file_path.endswith('.md'):  # Only load markdown files
            with open(file_path, 'r', encoding='utf-8') as f:
                corpus.append(f.read())

    return corpus


def semantic_search(corpus, sentence):
    """
    Perform semantic search on a corpus of documents.
    """
    # Load a pre-trained Sentence-BERT model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate embeddings for the corpus documents
    doc_embeddings = model.encode(corpus)

    # Generate an embedding for the input sentence
    query_embedding = model.encode([sentence])[0]

    # Compute cosine similarities between the query and each document
    similarities = [cosine_similarity(query_embedding, doc_embedding)
                    for doc_embedding in doc_embeddings]

    # Find the index of the document with the highest similarity score
    best_doc_index = np.argmax(similarities)

    # Return the most similar document
    return corpus[best_doc_index]

def question_answer(corpus_path):
    """
    Interactively answers questions based on multiple reference texts.
    """
    # Load the corpus documents
    corpus = load_corpus(corpus_path)

    # Exit keywords
    exit_keywords = ['exit', 'quit', 'goodbye', 'bye']

    while True:
        # Get user input
        user_input = input("Q: ").strip()

        # Check if the user wants to exit
        if user_input.lower() in exit_keywords:
            print("A: Goodbye")
            break

        # Perform semantic search to find the most relevant document
        relevant_doc = semantic_search(corpus, user_input)

        # Use the question_answer function to extract the answer from the relevant document
        answer = extract_answer(user_input, relevant_doc)

        # If no valid answer is found, return a default response
        if answer:
            print(f"A: {answer}")
        else:
            print("A: Sorry, I do not understand your question.")
