# QA Bot with Pretrained BERT

This project implements an interactive Question Answering (QA) system using a pretrained BERT model. The system uses BERT's capabilities to extract answers from a provided reference document based on user queries. The project includes a loop that allows continuous interaction with the bot and exits when specific keywords are typed.

## Project Structure

### 0-qa.py
- Contains the function `question_answer(question, reference)` which uses the pretrained BERT model to answer a given question based on the reference text.
- The function uses TensorFlow and TensorFlow Hub to load the model and BERT tokenizer.

### 1-loop.py
- Implements an interactive loop that continuously asks the user for a question and prints a placeholder response.
- The loop exits when the user types an exit keyword like 'exit', 'quit', 'goodbye', or 'bye'.

### 2-qa.py
- Extends the functionality of the QA bot by integrating the `question_answer` function from `0-qa.py`.
- The loop allows users to input a question and receive answers from a reference document until an exit keyword is entered.

### 3-semantic_search.py
- Implements semantic search functionality (not detailed here, likely used for enhancing question matching).

### ZendeskArticles
- Contains reference documents (likely FAQ or support articles) that the QA bot can use to answer user questions.

### c15a067b44a328c7d5a03c79070b7865f444d1e3
- Could be a file containing additional resources or data for the QA bot (likely a serialized model or dataset).

## Key Features

- **Interactive QA loop**: Allows users to ask questions in natural language and get relevant answers from the reference text.
- **Pretrained BERT model**: Uses the BERT model fine-tuned on the SQuAD dataset to extract answers from a document.
- **Exit functionality**: The loop can be exited using keywords like 'exit', 'quit', 'goodbye', and 'bye'.

## Requirements

- Python 3.x
- TensorFlow 2.x
- TensorFlow Hub
- Transformers library

## Usage

1. **Initialize the Bot**: Run the `2-qa.py` script to start the interactive QA loop. The bot will prompt you to enter a question.
   
2. **Ask Questions**: Type your questions, and the bot will use the reference document to find the best answer using the pretrained BERT model.

3. **Exit the Bot**: Type any of the following exit keywords to end the loop: `exit`, `quit`, `goodbye`, or `bye`.

```bash
$ python3 2-qa.py
Q: What is the capital of France?
A: Paris
Q: exit
A: Goodbye
```

## Example Flow
```bash
$ python3 2-qa.py
Q: Who is the president of the United States?
A: Joe Biden
Q: What is the capital of France?
A: Paris
Q: goodbye
A: Goodbye
```

## References
- BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding by Jacob Devlin et al.
- SQuAD: 100,000+ Questions for Machine Comprehension of Text

## Author
Davis Joseph (LinkedIn)
