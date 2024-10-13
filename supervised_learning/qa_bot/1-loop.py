#!/usr/bin/env python3
"""
Interactive QA loop that exits when the user types an exit keyword.
"""


def qa_loop():
    """
    Starts an interactive question-answering lop.
    """
    exit_keywords = ['exit', 'quit', 'goodbye', 'bye']

    while True:
        # Get input from the user
        user_input = input("Q: ").strip()

        # Check if the input is an exit keyword
        if user_input.lower() in exit_keywords:
            print("A: Goodbye")
            break

        # Placeholder response for the example
        # You can replace this part with actual logic to generate responses
        print("A:")


if __name__ == "__main__":
    qa_loop()
