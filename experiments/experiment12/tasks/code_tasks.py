"""code_tasks.py — 25 свежих задач «реализовать функцию» для
эксперимента 12, с decomposition в виде именованных requirements
(поведенческих категорий), а не плоских visible/holdout тестов.

Каждый requirement -- {"name", "description", "check"}. `description`
описывает ПОВЕДЕНЧЕСКУЮ КАТЕГОРИЮ ("handles zero correctly"), а не
конкретную пару вход/выход -- так K2 может честно показать «это
подтверждено» без раскрытия ожидаемого значения. Любой проваленный
requirement на первичной генерации -> коллизия. `check` -- python-булево
выражение, вычисляется тем же compile+exec+assert конвейером, что
`experiment11/seams.py.code_seam` (переиспользуется, не переписывается).
"""

from __future__ import annotations

CODE_TASKS = {
    "MSCODE_01": {
        "description": "Implement a function `is_even(n)` that returns True if n is even, False otherwise.",
        "function_name": "is_even",
        "signature_hint": "def is_even(n):",
        "requirements": [
            {"name": "typical_even", "description": "correctly identifies a typical positive even number", "check": "is_even(4) == True"},
            {"name": "typical_odd", "description": "correctly identifies a typical positive odd number", "check": "is_even(7) == False"},
            {"name": "zero_case", "description": "handles zero correctly", "check": "is_even(0) == True"},
            {"name": "negative_case", "description": "handles negative numbers correctly", "check": "is_even(-4) == True and is_even(-3) == False"},
        ],
    },
    "MSCODE_02": {
        "description": "Implement a function `sum_list(xs)` that returns the sum of all numbers in the list xs.",
        "function_name": "sum_list",
        "signature_hint": "def sum_list(xs):",
        "requirements": [
            {"name": "typical", "description": "sums a typical list of positive integers", "check": "sum_list([1, 2, 3]) == 6"},
            {"name": "empty", "description": "handles an empty list", "check": "sum_list([]) == 0"},
            {"name": "negatives", "description": "handles negative numbers in the list", "check": "sum_list([10, -2, 5]) == 13"},
            {"name": "single", "description": "handles a single-element list", "check": "sum_list([7]) == 7"},
        ],
    },
    "MSCODE_03": {
        "description": "Implement a function `reverse_string(s)` that returns the string s reversed.",
        "function_name": "reverse_string",
        "signature_hint": "def reverse_string(s):",
        "requirements": [
            {"name": "typical", "description": "reverses a typical multi-character string", "check": "reverse_string('abc') == 'cba'"},
            {"name": "empty", "description": "handles an empty string", "check": "reverse_string('') == ''"},
            {"name": "single_char", "description": "handles a single-character string", "check": "reverse_string('a') == 'a'"},
            {"name": "palindrome_input", "description": "returns the same string when input is already a palindrome", "check": "reverse_string('level') == 'level'"},
        ],
    },
    "MSCODE_04": {
        "description": "Implement a function `max_of_three(a, b, c)` that returns the largest of the three numbers a, b, c.",
        "function_name": "max_of_three",
        "signature_hint": "def max_of_three(a, b, c):",
        "requirements": [
            {"name": "typical", "description": "finds the max among distinct positive numbers", "check": "max_of_three(1, 5, 3) == 5"},
            {"name": "all_negative", "description": "handles all-negative inputs correctly", "check": "max_of_three(-1, -5, -3) == -1"},
            {"name": "all_tied", "description": "handles all three values being equal", "check": "max_of_three(2, 2, 2) == 2"},
            {"name": "two_tied_for_max", "description": "handles two values tied for the maximum", "check": "max_of_three(10, 9, 10) == 10"},
        ],
    },
    "MSCODE_05": {
        "description": "Implement a function `count_vowels(s)` that returns the number of vowels (a, e, i, o, u, case-insensitive) in the string s.",
        "function_name": "count_vowels",
        "signature_hint": "def count_vowels(s):",
        "requirements": [
            {"name": "typical", "description": "counts vowels in a typical lowercase word", "check": "count_vowels('hello') == 2"},
            {"name": "none", "description": "returns zero when there are no vowels", "check": "count_vowels('xyz') == 0"},
            {"name": "uppercase", "description": "counts uppercase vowels case-insensitively", "check": "count_vowels('AEIOU') == 5"},
            {"name": "empty", "description": "handles an empty string", "check": "count_vowels('') == 0"},
        ],
    },
    "MSCODE_06": {
        "description": "Implement a function `is_palindrome(s)` that returns True if the string s reads the same forwards and backwards, False otherwise.",
        "function_name": "is_palindrome",
        "signature_hint": "def is_palindrome(s):",
        "requirements": [
            {"name": "typical_true", "description": "correctly identifies a typical palindrome", "check": "is_palindrome('level') == True"},
            {"name": "typical_false", "description": "correctly identifies a typical non-palindrome", "check": "is_palindrome('hello') == False"},
            {"name": "empty", "description": "handles an empty string (considered a palindrome)", "check": "is_palindrome('') == True"},
            {"name": "single_char", "description": "handles a single-character string", "check": "is_palindrome('a') == True"},
        ],
    },
    "MSCODE_07": {
        "description": "Implement a function `factorial(n)` that returns n! (the factorial of n), where factorial(0) == 1.",
        "function_name": "factorial",
        "signature_hint": "def factorial(n):",
        "requirements": [
            {"name": "typical", "description": "computes a typical factorial", "check": "factorial(5) == 120"},
            {"name": "zero", "description": "handles the base case n=0 correctly", "check": "factorial(0) == 1"},
            {"name": "one", "description": "handles n=1 correctly", "check": "factorial(1) == 1"},
            {"name": "larger", "description": "computes a somewhat larger factorial correctly", "check": "factorial(6) == 720"},
        ],
    },
    "MSCODE_08": {
        "description": "Implement a function `gcd(a, b)` that returns the greatest common divisor of positive integers a and b.",
        "function_name": "gcd",
        "signature_hint": "def gcd(a, b):",
        "requirements": [
            {"name": "typical", "description": "finds the gcd of two typical numbers with a common factor", "check": "gcd(12, 8) == 4"},
            {"name": "coprime", "description": "handles coprime numbers (gcd = 1)", "check": "gcd(17, 5) == 1"},
            {"name": "shared_large_factor", "description": "handles numbers sharing a large common factor", "check": "gcd(100, 75) == 25"},
            {"name": "equal_inputs", "description": "handles equal inputs (gcd = the value itself)", "check": "gcd(7, 7) == 7"},
        ],
    },
    "MSCODE_09": {
        "description": "Implement a function `remove_duplicates(xs)` that returns a list with duplicate values removed, preserving the first occurrence order.",
        "function_name": "remove_duplicates",
        "signature_hint": "def remove_duplicates(xs):",
        "requirements": [
            {"name": "typical", "description": "removes duplicates while preserving first-occurrence order", "check": "remove_duplicates([1, 2, 2, 3, 1]) == [1, 2, 3]"},
            {"name": "empty", "description": "handles an empty list", "check": "remove_duplicates([]) == []"},
            {"name": "all_same", "description": "handles a list where every element is identical", "check": "remove_duplicates([5, 5, 5]) == [5]"},
            {"name": "no_duplicates", "description": "leaves a list with no duplicates unchanged", "check": "remove_duplicates([1, 2, 3]) == [1, 2, 3]"},
        ],
    },
    "MSCODE_10": {
        "description": "Implement a function `fibonacci_nth(n)` that returns the n-th Fibonacci number (0-indexed: fibonacci_nth(0) == 0, fibonacci_nth(1) == 1).",
        "function_name": "fibonacci_nth",
        "signature_hint": "def fibonacci_nth(n):",
        "requirements": [
            {"name": "base0", "description": "handles the base case n=0 correctly", "check": "fibonacci_nth(0) == 0"},
            {"name": "base1", "description": "handles the base case n=1 correctly", "check": "fibonacci_nth(1) == 1"},
            {"name": "typical", "description": "computes a typical Fibonacci number", "check": "fibonacci_nth(5) == 5"},
            {"name": "larger", "description": "computes a somewhat larger Fibonacci number correctly", "check": "fibonacci_nth(8) == 21"},
        ],
    },
    "MSCODE_11": {
        "description": "Implement a function `is_prime(n)` that returns True if n is a prime number, False otherwise.",
        "function_name": "is_prime",
        "signature_hint": "def is_prime(n):",
        "requirements": [
            {"name": "typical_prime", "description": "correctly identifies a typical prime number", "check": "is_prime(7) == True"},
            {"name": "typical_nonprime", "description": "correctly identifies a typical non-prime number", "check": "is_prime(8) == False"},
            {"name": "one", "description": "handles n=1 correctly (not prime)", "check": "is_prime(1) == False"},
            {"name": "two", "description": "handles n=2 correctly (smallest prime)", "check": "is_prime(2) == True"},
            {"name": "larger_prime", "description": "correctly identifies a larger prime number", "check": "is_prime(97) == True"},
        ],
    },
    "MSCODE_12": {
        "description": "Implement a function `count_words(s)` that returns the number of whitespace-separated words in the string s.",
        "function_name": "count_words",
        "signature_hint": "def count_words(s):",
        "requirements": [
            {"name": "typical", "description": "counts words in a typical sentence", "check": "count_words('the quick brown fox') == 4"},
            {"name": "empty", "description": "handles an empty string", "check": "count_words('') == 0"},
            {"name": "single_word", "description": "handles a single-word string", "check": "count_words('hello') == 1"},
            {"name": "extra_spaces", "description": "handles multiple consecutive spaces between words", "check": "count_words('a   b  c') == 3"},
        ],
    },
    "MSCODE_13": {
        "description": "Implement a function `flatten_list(nested)` that flattens a list of lists by exactly one level, returning a single flat list.",
        "function_name": "flatten_list",
        "signature_hint": "def flatten_list(nested):",
        "requirements": [
            {"name": "typical", "description": "flattens a typical list of non-empty sublists", "check": "flatten_list([[1, 2], [3, 4]]) == [1, 2, 3, 4]"},
            {"name": "empty_sublists", "description": "handles empty sublists mixed with non-empty ones", "check": "flatten_list([[], [1], []]) == [1]"},
            {"name": "empty", "description": "handles an empty outer list", "check": "flatten_list([]) == []"},
            {"name": "single_sublist", "description": "handles a single sublist", "check": "flatten_list([[1, 2, 3]]) == [1, 2, 3]"},
        ],
    },
    "MSCODE_14": {
        "description": "Implement a function `is_anagram(a, b)` that returns True if strings a and b are anagrams of each other, False otherwise.",
        "function_name": "is_anagram",
        "signature_hint": "def is_anagram(a, b):",
        "requirements": [
            {"name": "typical_true", "description": "correctly identifies a typical anagram pair", "check": "is_anagram('listen', 'silent') == True"},
            {"name": "typical_false", "description": "correctly identifies a typical non-anagram pair", "check": "is_anagram('hello', 'world') == False"},
            {"name": "different_lengths", "description": "handles strings of different lengths correctly (not anagrams)", "check": "is_anagram('abc', 'ab') == False"},
            {"name": "identical_strings", "description": "handles two identical strings (trivially anagrams)", "check": "is_anagram('abc', 'abc') == True"},
        ],
    },
    "MSCODE_15": {
        "description": "Implement a function `sum_digits(n)` that returns the sum of the decimal digits of non-negative integer n.",
        "function_name": "sum_digits",
        "signature_hint": "def sum_digits(n):",
        "requirements": [
            {"name": "typical", "description": "sums the digits of a typical multi-digit number", "check": "sum_digits(1234) == 10"},
            {"name": "single_digit", "description": "handles a single-digit number", "check": "sum_digits(7) == 7"},
            {"name": "zero", "description": "handles zero correctly", "check": "sum_digits(0) == 0"},
            {"name": "larger", "description": "sums the digits of a larger number correctly", "check": "sum_digits(98765) == 35"},
        ],
    },
    "MSCODE_16": {
        "description": "Implement a function `min_max_diff(xs)` that returns the difference between the maximum and minimum values in the list xs.",
        "function_name": "min_max_diff",
        "signature_hint": "def min_max_diff(xs):",
        "requirements": [
            {"name": "typical", "description": "computes the range of a typical list of positive integers", "check": "min_max_diff([3, 1, 4, 1, 5]) == 4"},
            {"name": "all_same", "description": "handles a list where every element is identical (range 0)", "check": "min_max_diff([2, 2, 2]) == 0"},
            {"name": "negatives", "description": "handles a list spanning negative and positive values", "check": "min_max_diff([-5, 0, 5]) == 10"},
            {"name": "two_elements", "description": "handles a two-element list", "check": "min_max_diff([10, 3]) == 7"},
        ],
    },
    "MSCODE_17": {
        "description": "Implement a function `capitalize_words(s)` that capitalizes the first letter of every whitespace-separated word in s.",
        "function_name": "capitalize_words",
        "signature_hint": "def capitalize_words(s):",
        "requirements": [
            {"name": "typical", "description": "capitalizes each word in a typical lowercase sentence", "check": "capitalize_words('hello world') == 'Hello World'"},
            {"name": "empty", "description": "handles an empty string", "check": "capitalize_words('') == ''"},
            {"name": "single_word", "description": "handles a single-word string", "check": "capitalize_words('python') == 'Python'"},
            {"name": "already_capitalized", "description": "leaves an already-capitalized sentence unchanged", "check": "capitalize_words('Hello World') == 'Hello World'"},
        ],
    },
    "MSCODE_18": {
        "description": "Implement a function `count_occurrences(xs, target)` that returns how many times target appears in list xs.",
        "function_name": "count_occurrences",
        "signature_hint": "def count_occurrences(xs, target):",
        "requirements": [
            {"name": "typical", "description": "counts a typical repeated value", "check": "count_occurrences([1, 2, 2, 3, 2], 2) == 3"},
            {"name": "none", "description": "returns zero when the target is not present", "check": "count_occurrences([1, 2, 3], 5) == 0"},
            {"name": "empty", "description": "handles an empty list", "check": "count_occurrences([], 1) == 0"},
            {"name": "all_match", "description": "handles a list where every element matches the target", "check": "count_occurrences([4, 4, 4], 4) == 3"},
        ],
    },
    "MSCODE_19": {
        "description": "Implement a function `is_sorted(xs)` that returns True if the list xs is sorted in non-decreasing order, False otherwise.",
        "function_name": "is_sorted",
        "signature_hint": "def is_sorted(xs):",
        "requirements": [
            {"name": "typical_true", "description": "correctly identifies a typical sorted list", "check": "is_sorted([1, 2, 3, 4]) == True"},
            {"name": "typical_false", "description": "correctly identifies a typical unsorted list", "check": "is_sorted([3, 1, 2]) == False"},
            {"name": "empty", "description": "handles an empty list (trivially sorted)", "check": "is_sorted([]) == True"},
            {"name": "single_element", "description": "handles a single-element list", "check": "is_sorted([5]) == True"},
            {"name": "duplicates", "description": "treats duplicate consecutive values as sorted", "check": "is_sorted([1, 2, 2, 3]) == True"},
        ],
    },
    "MSCODE_20": {
        "description": "Implement a function `average(xs)` that returns the arithmetic mean of the numbers in list xs.",
        "function_name": "average",
        "signature_hint": "def average(xs):",
        "requirements": [
            {"name": "typical", "description": "computes the average of a typical list", "check": "average([2, 4, 6]) == 4"},
            {"name": "single", "description": "handles a single-element list", "check": "average([10]) == 10"},
            {"name": "negatives", "description": "handles a list that averages to zero", "check": "average([-2, 2]) == 0"},
            {"name": "decimal_result", "description": "returns a non-integer average correctly", "check": "average([1, 2]) == 1.5"},
        ],
    },
    "MSCODE_21": {
        "description": "Implement a function `is_leap_year(year)` that returns True if year is a leap year in the Gregorian calendar, False otherwise.",
        "function_name": "is_leap_year",
        "signature_hint": "def is_leap_year(year):",
        "requirements": [
            {"name": "typical_leap", "description": "correctly identifies a typical leap year", "check": "is_leap_year(2020) == True"},
            {"name": "typical_nonleap", "description": "correctly identifies a typical non-leap year", "check": "is_leap_year(2019) == False"},
            {"name": "century_nonleap", "description": "handles a century year divisible by 100 but not 400 (not a leap year)", "check": "is_leap_year(1900) == False"},
            {"name": "century_leap", "description": "handles a century year divisible by 400 (a leap year)", "check": "is_leap_year(2000) == True"},
        ],
    },
    "MSCODE_22": {
        "description": "Implement a function `celsius_to_fahrenheit(c)` that converts a Celsius temperature to Fahrenheit.",
        "function_name": "celsius_to_fahrenheit",
        "signature_hint": "def celsius_to_fahrenheit(c):",
        "requirements": [
            {"name": "freezing", "description": "correctly converts the freezing point of water", "check": "celsius_to_fahrenheit(0) == 32"},
            {"name": "boiling", "description": "correctly converts the boiling point of water", "check": "celsius_to_fahrenheit(100) == 212"},
            {"name": "negative", "description": "correctly handles a negative Celsius value", "check": "celsius_to_fahrenheit(-40) == -40"},
            {"name": "decimal", "description": "correctly handles a non-round Celsius value", "check": "abs(celsius_to_fahrenheit(37) - 98.6) < 0.01"},
        ],
    },
    "MSCODE_23": {
        "description": "Implement a function `binary_search(xs, target)` that returns the index of target in the SORTED list xs, or -1 if not present.",
        "function_name": "binary_search",
        "signature_hint": "def binary_search(xs, target):",
        "requirements": [
            {"name": "found_middle", "description": "finds a target located in the middle of the list", "check": "binary_search([1, 3, 5, 7, 9], 5) == 2"},
            {"name": "not_found", "description": "returns -1 when the target is absent", "check": "binary_search([1, 3, 5, 7, 9], 4) == -1"},
            {"name": "found_first", "description": "finds a target at the first position", "check": "binary_search([1, 3, 5, 7, 9], 1) == 0"},
            {"name": "found_last", "description": "finds a target at the last position", "check": "binary_search([1, 3, 5, 7, 9], 9) == 4"},
            {"name": "empty", "description": "handles an empty list", "check": "binary_search([], 5) == -1"},
        ],
    },
    "MSCODE_24": {
        "description": "Implement a function `rotate_list(xs, k)` that returns xs rotated left by k positions.",
        "function_name": "rotate_list",
        "signature_hint": "def rotate_list(xs, k):",
        "requirements": [
            {"name": "typical", "description": "rotates a typical list left by a partial amount", "check": "rotate_list([1, 2, 3, 4, 5], 2) == [3, 4, 5, 1, 2]"},
            {"name": "zero_rotation", "description": "leaves the list unchanged when k=0", "check": "rotate_list([1, 2, 3], 0) == [1, 2, 3]"},
            {"name": "full_rotation", "description": "leaves the list unchanged when k equals the list length", "check": "rotate_list([1, 2, 3], 3) == [1, 2, 3]"},
            {"name": "empty", "description": "handles an empty list", "check": "rotate_list([], 2) == []"},
        ],
    },
    "MSCODE_25": {
        "description": "Implement a function `is_perfect_square(n)` that returns True if non-negative integer n is a perfect square, False otherwise.",
        "function_name": "is_perfect_square",
        "signature_hint": "def is_perfect_square(n):",
        "requirements": [
            {"name": "typical_true", "description": "correctly identifies a typical perfect square", "check": "is_perfect_square(16) == True"},
            {"name": "typical_false", "description": "correctly identifies a typical non-square", "check": "is_perfect_square(15) == False"},
            {"name": "zero", "description": "handles zero correctly (a perfect square)", "check": "is_perfect_square(0) == True"},
            {"name": "one", "description": "handles one correctly (a perfect square)", "check": "is_perfect_square(1) == True"},
            {"name": "larger", "description": "correctly identifies a larger perfect square", "check": "is_perfect_square(10000) == True"},
        ],
    },
}

CODE_TASK_IDS = tuple(CODE_TASKS)


def generation_prompt(task: dict) -> str:
    return (
        f"{task['description']}\n\n"
        f"Reply with ONLY the Python function definition, starting with "
        f"`{task['signature_hint']}`. No explanation, no example usage, no other text."
    )
