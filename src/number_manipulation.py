import random


def add_random_number(user_input: int) -> int:
    random_addition = random.randint(1, 100)  # Add a random number between 1 and 100
    result = user_input + random_addition
    return result
