# NLP Metrics: BLEU Score Calculation

This project implements several variations of the BLEU (Bilingual Evaluation Understudy) score, a popular metric for evaluating the quality of machine-generated translations by comparing them to reference translations. It includes:

- Unigram BLEU score
- N-gram BLEU score
- Cumulative n-gram BLEU score

The BLEU score is widely used in machine translation and NLP tasks to assess how well a model's output matches reference translations.

## Directory Overview

### Files and Scripts

1. **`0-uni_bleu.py`**
   - **Function**: `uni_bleu(references, sentence)`
   - **Description**: Calculates the unigram BLEU score for a given sentence compared to reference translations.
     - Compares individual words (unigrams) between the candidate sentence and the reference sentences.
     - Includes a brevity penalty based on the length of the sentence.

2. **`1-ngram_bleu.py`**
   - **Function**: `ngram_bleu(references, sentence, n)`
   - **Description**: Calculates the n-gram BLEU score for a given sentence.
     - Allows for any `n` (e.g., 2 for bigrams, 3 for trigrams) to be specified.
     - Computes precision by counting the overlap of n-grams between the candidate sentence and references.
     - Includes a brevity penalty based on sentence length.

3. **`2-cumulative_bleu.py`**
   - **Function**: `cumulative_bleu(references, sentence, n)`
   - **Description**: Calculates the cumulative n-gram BLEU score for a given sentence.
     - Computes precision for n-grams of varying sizes (from 1 to n) and combines them to give a cumulative BLEU score.
     - Incorporates a brevity penalty similar to the n-gram BLEU calculation.

---

## Requirements

- Python 3.x
- NumPy

---

## How to Use

### 1. **Unigram BLEU Score**:
   - Example:
     ```python
     from 0-uni_bleu import uni_bleu
     references = [["the", "cat", "sat", "on", "the", "mat"]]
     sentence = ["the", "cat", "sat", "on", "mat"]
     score = uni_bleu(references, sentence)
     print(score)
     ```

### 2. **N-gram BLEU Score**:
   - Example for bigram (n=2):
     ```python
     from 1-ngram_bleu import ngram_bleu
     references = [["the", "cat", "sat", "on", "the", "mat"]]
     sentence = ["the", "cat", "sat", "on", "mat"]
     score = ngram_bleu(references, sentence, 2)
     print(score)
     ```

### 3. **Cumulative n-gram BLEU Score**:
   - Example for cumulative BLEU score up to trigrams (n=3):
     ```python
     from 2-cumulative_bleu import cumulative_bleu
     references = [["the", "cat", "sat", "on", "the", "mat"]]
     sentence = ["the", "cat", "sat", "on", "mat"]
     score = cumulative_bleu(references, sentence, 3)
     print(score)
     ```

---

## BLEU Score Formula

The BLEU score combines two main components:

1. **Precision**: Measures how many n-grams in the candidate sentence match the reference n-grams.
2. **Brevity Penalty**: Penalizes translations that are shorter than the reference translations.

The score is computed as:

\[
BLEU = BP \times \text{precision}
\]

Where \( BP \) (Brevity Penalty) is defined as:

\[
BP = \begin{cases}
1 & \text{if sentence length is greater than or equal to reference length} \\
\exp(1 - \frac{\text{closest reference length}}{\text{sentence length}}) & \text{otherwise}
\end{cases}
\]

---

## References

- [BLEU Score Wikipedia](https://en.wikipedia.org/wiki/BLEU)
- [NLP Metric Evaluation Paper](https://aclanthology.org/P03-1022.pdf)

---

## Author

- **Davis Joseph**  
  [LinkedIn Profile](https://www.linkedin.com/in/davisjoseph767/)

