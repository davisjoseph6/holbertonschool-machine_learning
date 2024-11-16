# Word Embeddings in Natural Language Processing

This project demonstrates various methods for generating word embeddings from text data. These methods include Bag-of-Words (BoW), TF-IDF, Word2Vec, and FastText. The embeddings generated can be used for various NLP tasks such as text classification, sentiment analysis, or as inputs to more advanced models like RNNs or transformers.

## Directory Overview

### Files and Scripts

1. **`0-bag_of_words.py`**
   - **Function**: `bag_of_words(sentences, vocab=None)`
   - **Description**: Converts a list of sentences into a Bag-of-Words embedding matrix.
     - Tokenizes and cleans sentences by removing non-alphabetic characters.
     - Creates a vocabulary (if not provided) and generates a binary embedding matrix, where each row represents a sentence and each column corresponds to the count of a word in that sentence.

2. **`1-tf_idf.py`**
   - **Function**: `tf_idf(sentences, vocab=None)`
   - **Description**: Creates a TF-IDF (Term Frequency-Inverse Document Frequency) embedding matrix.
     - Tokenizes and preprocesses sentences (removes possessive suffix `'s` and converts to lowercase).
     - Uses `TfidfVectorizer` from `sklearn` to compute TF-IDF values for the provided vocabulary.

3. **`2-word2vec.py`**
   - **Function**: `word2vec_model(sentences, ...)`
   - **Description**: Trains a Word2Vec model using Gensim.
     - Uses the Word2Vec model from Gensim to train embeddings on a set of sentences. You can configure the model's parameters such as vector size, window, and training algorithm (CBOW or skip-gram).
     - The trained model can be used for word similarity, analogy tasks, and more.

4. **`3-gensim_to_keras.py`**
   - **Function**: `gensim_to_keras(model)`
   - **Description**: Converts a trained Gensim Word2Vec model to a Keras Embedding layer.
     - Converts the word vectors from a Gensim Word2Vec model into a Keras `Embedding` layer, which can be used directly in Keras models for deep learning tasks.

5. **`4-fasttext.py`**
   - **Function**: `fasttext_model(sentences, ...)`
   - **Description**: Trains a FastText model using Gensim.
     - Similar to Word2Vec but capable of representing out-of-vocabulary words by breaking them down into character-level n-grams. This can be particularly useful for languages with rich morphology or when dealing with rare words.

---

## Requirements

- Python 3.x
- Gensim
- TensorFlow (for `gensim_to_keras.py`)
- scikit-learn
- NumPy
- Matplotlib (optional, for visualization)

---

## How to Use

### 1. **Bag-of-Words Embedding**:
   - Create a Bag-of-Words representation for a list of sentences.
   - Example:
     ```python
     from 0-bag_of_words import bag_of_words
     sentences = ["This is a sample sentence.", "This is another one."]
     embeddings, vocab = bag_of_words(sentences)
     print(embeddings)
     ```

### 2. **TF-IDF Embedding**:
   - Generate a TF-IDF matrix for a list of sentences.
   - Example:
     ```python
     from 1-tf_idf import tf_idf
     sentences = ["This is a sample sentence.", "This is another one."]
     tfidf_matrix, features = tf_idf(sentences)
     print(tfidf_matrix)
     ```

### 3. **Word2Vec Model**:
   - Train a Word2Vec model on your sentences and use it to obtain word embeddings.
   - Example:
     ```python
     from 2-word2vec import word2vec_model
     sentences = [["this", "is", "a", "sentence"], ["another", "sentence"]]
     model = word2vec_model(sentences)
     word_vector = model.wv['sentence']
     print(word_vector)
     ```

### 4. **Convert Gensim Model to Keras**:
   - After training a Word2Vec model with Gensim, convert it to a Keras Embedding layer.
   - Example:
     ```python
     from 3-gensim_to_keras import gensim_to_keras
     embedding_layer = gensim_to_keras(model)
     ```

### 5. **Train FastText Model**:
   - Train a FastText model using Gensim for handling subword information.
   - Example:
     ```python
     from 4-fasttext import fasttext_model
     sentences = [["this", "is", "a", "sentence"], ["another", "sentence"]]
     fasttext_model = fasttext_model(sentences)
     ```

---

## References

- [Gensim Word2Vec Documentation](https://radimrehurek.com/gensim/models/word2vec.html)
- [FastText](https://fasttext.cc/)
- [TF-IDF in NLP](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)

---

## Author

- **Davis Joseph**  
  [LinkedIn Profile](https://www.linkedin.com/in/davisjoseph767/)

