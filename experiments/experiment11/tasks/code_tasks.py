"""code_tasks.py — свежие мини-задачи «реализовать функцию» для
эксперимента 11.

Даёт механическую разбивку на syntactic/structural/logical (см.
seams.code_seam): visible_tests используются для первичного FAIL,
holdout_tests -- ТОЛЬКО действием SEEK_EVIDENCE (доп. механическое
evidence без новой генерации, п.2 задания).

НЕ переиспользует arch1/src/seams/exec_seam.py -- проверка пишется
заново в seams.py (тот же паттерн, что experiment8/run_tests.py,
задолго до Mycelium: compile+exec+assert, не архитектурная эвристика).
"""

from __future__ import annotations

CODE_TASKS = {
    "CODE_01": {
        "description": "Implement a function `is_even(n)` that returns True if n is even, False otherwise.",
        "function_name": "is_even",
        "signature_hint": "def is_even(n):",
        "visible_tests": ["is_even(4) == True", "is_even(7) == False"],
        "holdout_tests": ["is_even(0) == True", "is_even(-3) == False"],
    },
    "CODE_02": {
        "description": "Implement a function `sum_list(xs)` that returns the sum of all numbers in the list xs.",
        "function_name": "sum_list",
        "signature_hint": "def sum_list(xs):",
        "visible_tests": ["sum_list([1, 2, 3]) == 6", "sum_list([]) == 0"],
        "holdout_tests": ["sum_list([10, -2, 5]) == 13", "sum_list([7]) == 7"],
    },
    "CODE_03": {
        "description": "Implement a function `reverse_string(s)` that returns the string s reversed.",
        "function_name": "reverse_string",
        "signature_hint": "def reverse_string(s):",
        "visible_tests": ["reverse_string('abc') == 'cba'", "reverse_string('') == ''"],
        "holdout_tests": ["reverse_string('a') == 'a'", "reverse_string('hello') == 'olleh'"],
    },
    "CODE_04": {
        "description": "Implement a function `max_of_three(a, b, c)` that returns the largest of the three numbers a, b, c.",
        "function_name": "max_of_three",
        "signature_hint": "def max_of_three(a, b, c):",
        "visible_tests": ["max_of_three(1, 5, 3) == 5", "max_of_three(-1, -5, -3) == -1"],
        "holdout_tests": ["max_of_three(2, 2, 2) == 2", "max_of_three(10, 9, 10) == 10"],
    },
    "CODE_05": {
        "description": "Implement a function `count_vowels(s)` that returns the number of vowels (a, e, i, o, u, case-insensitive) in the string s.",
        "function_name": "count_vowels",
        "signature_hint": "def count_vowels(s):",
        "visible_tests": ["count_vowels('hello') == 2", "count_vowels('xyz') == 0"],
        "holdout_tests": ["count_vowels('AEIOU') == 5", "count_vowels('') == 0"],
    },
    "CODE_06": {
        "description": "Implement a function `is_palindrome(s)` that returns True if the string s reads the same forwards and backwards, False otherwise.",
        "function_name": "is_palindrome",
        "signature_hint": "def is_palindrome(s):",
        "visible_tests": ["is_palindrome('level') == True", "is_palindrome('hello') == False"],
        "holdout_tests": ["is_palindrome('') == True", "is_palindrome('a') == True"],
    },
    "CODE_07": {
        "description": "Implement a function `factorial(n)` that returns n! (the factorial of n), where factorial(0) == 1.",
        "function_name": "factorial",
        "signature_hint": "def factorial(n):",
        "visible_tests": ["factorial(5) == 120", "factorial(0) == 1"],
        "holdout_tests": ["factorial(1) == 1", "factorial(6) == 720"],
    },
    "CODE_08": {
        "description": "Implement a function `gcd(a, b)` that returns the greatest common divisor of positive integers a and b.",
        "function_name": "gcd",
        "signature_hint": "def gcd(a, b):",
        "visible_tests": ["gcd(12, 8) == 4", "gcd(17, 5) == 1"],
        "holdout_tests": ["gcd(100, 75) == 25", "gcd(7, 7) == 7"],
    },
    "CODE_09": {
        "description": "Implement a function `remove_duplicates(xs)` that returns a list with duplicate values removed, preserving the first occurrence order.",
        "function_name": "remove_duplicates",
        "signature_hint": "def remove_duplicates(xs):",
        "visible_tests": ["remove_duplicates([1, 2, 2, 3, 1]) == [1, 2, 3]", "remove_duplicates([]) == []"],
        "holdout_tests": ["remove_duplicates([5, 5, 5]) == [5]", "remove_duplicates([1, 2, 3]) == [1, 2, 3]"],
    },
    "CODE_10": {
        "description": "Implement a function `fibonacci_nth(n)` that returns the n-th Fibonacci number (0-indexed: fibonacci_nth(0) == 0, fibonacci_nth(1) == 1).",
        "function_name": "fibonacci_nth",
        "signature_hint": "def fibonacci_nth(n):",
        "visible_tests": ["fibonacci_nth(0) == 0", "fibonacci_nth(1) == 1"],
        "holdout_tests": ["fibonacci_nth(5) == 5", "fibonacci_nth(8) == 21"],
    },
}

CODE_TASK_IDS = tuple(CODE_TASKS)


def generation_prompt(task: dict) -> str:
    return (
        f"{task['description']}\n\n"
        f"Reply with ONLY the Python function definition, starting with "
        f"`{task['signature_hint']}`. No explanation, no example usage, no other text."
    )


def retry_prompt(task: dict, prior_artifact: str, evidence: str) -> str:
    return (
        f"{task['description']}\n\n"
        f"A previous attempt produced this code:\n{prior_artifact}\n\n"
        f"This was mechanically verified INCORRECT: {evidence}\n\n"
        f"Provide a corrected, complete implementation, starting with `{task['signature_hint']}`. "
        "Reply with ONLY the function definition, no explanation."
    )


def retry_prompt_v2(task: dict, prior_artifact: str, evidence: str) -> str:
    """prompt_profile B (плечо CHANGE_PROMPT) -- зафиксирована ДО прогона."""
    return (
        f"Fix this Python function so it correctly does the following:\n{task['description']}\n\n"
        f"Broken version:\n{prior_artifact}\n\n"
        f"Problem found: {evidence}\n\n"
        f"Think about which input breaks it, then write a fixed version starting with "
        f"`{task['signature_hint']}`. Output only the corrected code."
    )
