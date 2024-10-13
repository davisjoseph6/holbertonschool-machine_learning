#!/usr/bin/env python3
"""
Question Answering loop using pretrained BERT that exits on specific keywords.
"""

# Import the 'question_answer' function from '0-qa.py'
qa_module = __import__('0-qa')
question_answer = qa_module.question_answer  # Extract the function


def answer_loop(reference):
    """
    Interactive loop that answers questions based on a reference text.
    """
    exit_keywords = ['exit', 'quit', 'goodbye', 'bye']

    while True:
        # Get input from the user
        user_input = input("Q: ").strip()

        # Check if the input is an exit keyword
        if user_input.lower() in exit_keywords:
            print("A: Goodbye")
            break

        # Call the question_answer funciton to get the answer
        answer = question_answer(user_input, reference)

        # If no valid answer is found, return a default response
        if answer:
            print(f"A: {answer}")
        else:
            print("A: Sorry, I do not understand your question.")
