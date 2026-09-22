"""
Fixed task set for Experiment 2 (artifact pipeline vs monolithic best-of-N).
Code domain only. Reuses CODE_01..03 verbatim from experiment 1 (same task
descriptions, same ground truth) and adds three new tasks of the same kind
(fixed, objectively checkable, not tailored to the pipeline -- ordinary
small coding exercises).

Each task carries a REFERENCE_IMPLEMENTATION (hand-written correct code,
used only for: mutation testing of A2's tests, and as a sanity baseline)
and REFERENCE_TESTS (a separate, hand-written assert-based test suite --
the external ground-truth verifier, independent of anything an LLM
produces). Test functions must be named test_*, take no arguments, and use
plain `assert` (no pytest dependency).
"""

TASKS = [
    {
        "task_id": "CODE_01",
        "question": (
            "Write a Python function named is_palindrome(s) that returns True if the "
            "string s is a palindrome when case is ignored, and False otherwise. Only "
            "alphabetic case should be ignored; do not strip spaces or punctuation."
        ),
        "function_name": "is_palindrome",
        "arg_names": ["s"],
        "reference_implementation": (
            "def is_palindrome(s):\n"
            "    t = s.lower()\n"
            "    return t == t[::-1]\n"
        ),
        "reference_tests": (
            "def test_simple_true():\n"
            "    assert is_palindrome('Racecar') is True\n\n"
            "def test_simple_false():\n"
            "    assert is_palindrome('hello') is False\n\n"
            "def test_single_char():\n"
            "    assert is_palindrome('A') is True\n\n"
            "def test_case_mixed_false():\n"
            "    assert is_palindrome('Ab') is False\n\n"
            "def test_empty_string():\n"
            "    assert is_palindrome('') is True\n\n"
            "def test_spaces_not_stripped():\n"
            "    assert is_palindrome('a b a') is True\n"
            "    assert is_palindrome('ab cd') is False\n"
        ),
    },
    {
        "task_id": "CODE_02",
        "question": (
            "Write a Python function named sum_even(nums) that takes a list of integers "
            "nums and returns the sum of all even numbers in the list. If there are no "
            "even numbers, return 0."
        ),
        "function_name": "sum_even",
        "arg_names": ["nums"],
        "reference_implementation": (
            "def sum_even(nums):\n"
            "    return sum(n for n in nums if n % 2 == 0)\n"
        ),
        "reference_tests": (
            "def test_mixed():\n"
            "    assert sum_even([1, 2, 3, 4, 5, 6]) == 12\n\n"
            "def test_empty():\n"
            "    assert sum_even([]) == 0\n\n"
            "def test_all_odd():\n"
            "    assert sum_even([1, 3, 5]) == 0\n\n"
            "def test_negative():\n"
            "    assert sum_even([-2, -3, 4]) == 2\n\n"
            "def test_zero_included():\n"
            "    assert sum_even([0, 1, 2]) == 2\n"
        ),
    },
    {
        "task_id": "CODE_03",
        "question": (
            "Write a Python function named fibonacci(n) that returns the n-th Fibonacci "
            "number using 0-indexing, where fibonacci(0) == 0 and fibonacci(1) == 1."
        ),
        "function_name": "fibonacci",
        "arg_names": ["n"],
        "reference_implementation": (
            "def fibonacci(n):\n"
            "    a, b = 0, 1\n"
            "    for _ in range(n):\n"
            "        a, b = b, a + b\n"
            "    return a\n"
        ),
        "reference_tests": (
            "def test_base0():\n"
            "    assert fibonacci(0) == 0\n\n"
            "def test_base1():\n"
            "    assert fibonacci(1) == 1\n\n"
            "def test_five():\n"
            "    assert fibonacci(5) == 5\n\n"
            "def test_ten():\n"
            "    assert fibonacci(10) == 55\n\n"
            "def test_two():\n"
            "    assert fibonacci(2) == 1\n"
        ),
    },
    {
        "task_id": "CODE_04",
        "question": (
            "Write a Python function named count_vowels(s) that returns the number of "
            "vowel characters (a, e, i, o, u, both uppercase and lowercase) in the "
            "string s. Count 'y' as a consonant, never as a vowel."
        ),
        "function_name": "count_vowels",
        "arg_names": ["s"],
        "reference_implementation": (
            "def count_vowels(s):\n"
            "    return sum(1 for ch in s if ch.lower() in 'aeiou')\n"
        ),
        "reference_tests": (
            "def test_simple():\n"
            "    assert count_vowels('hello world') == 3\n\n"
            "def test_uppercase():\n"
            "    assert count_vowels('HELLO') == 2\n\n"
            "def test_no_vowels():\n"
            "    assert count_vowels('xyz') == 0\n\n"
            "def test_empty():\n"
            "    assert count_vowels('') == 0\n\n"
            "def test_y_is_consonant():\n"
            "    assert count_vowels('sky') == 0\n\n"
            "def test_all_vowels():\n"
            "    assert count_vowels('AEIOUaeiou') == 10\n"
        ),
    },
    {
        "task_id": "CODE_05",
        "question": (
            "Write a Python function named is_prime(n) that returns True if the integer "
            "n is a prime number, and False otherwise. Numbers less than 2 are not prime."
        ),
        "function_name": "is_prime",
        "arg_names": ["n"],
        "reference_implementation": (
            "def is_prime(n):\n"
            "    if n < 2:\n"
            "        return False\n"
            "    if n < 4:\n"
            "        return True\n"
            "    if n % 2 == 0:\n"
            "        return False\n"
            "    i = 3\n"
            "    while i * i <= n:\n"
            "        if n % i == 0:\n"
            "            return False\n"
            "        i += 2\n"
            "    return True\n"
        ),
        "reference_tests": (
            "def test_two_is_prime():\n"
            "    assert is_prime(2) is True\n\n"
            "def test_one_not_prime():\n"
            "    assert is_prime(1) is False\n\n"
            "def test_zero_not_prime():\n"
            "    assert is_prime(0) is False\n\n"
            "def test_negative_not_prime():\n"
            "    assert is_prime(-7) is False\n\n"
            "def test_nine_not_prime():\n"
            "    assert is_prime(9) is False\n\n"
            "def test_four_not_prime():\n"
            "    assert is_prime(4) is False\n\n"
            "def test_hundred_not_prime():\n"
            "    assert is_prime(100) is False\n\n"
            "def test_seventeen_is_prime():\n"
            "    assert is_prime(17) is True\n\n"
            "def test_large_composite():\n"
            "    assert is_prime(97 * 89) is False\n"
        ),
    },
    {
        "task_id": "CODE_06",
        "question": (
            "Write a Python function named two_sum(nums, target) that takes a list of "
            "integers nums and an integer target, and returns a tuple of the two "
            "0-based indices (i, j) with i < j such that nums[i] + nums[j] == target. "
            "Assume exactly one valid pair exists. Return the pair with the smallest i "
            "(and smallest j for that i) if multiple pairs satisfy the sum."
        ),
        "function_name": "two_sum",
        "arg_names": ["nums", "target"],
        "reference_implementation": (
            "def two_sum(nums, target):\n"
            "    seen = {}\n"
            "    for j, v in enumerate(nums):\n"
            "        need = target - v\n"
            "        if need in seen:\n"
            "            return (seen[need], j)\n"
            "        if v not in seen:\n"
            "            seen[v] = j\n"
            "    return None\n"
        ),
        "reference_tests": (
            "def test_simple():\n"
            "    assert two_sum([2, 7, 11, 15], 9) == (0, 1)\n\n"
            "def test_later_pair():\n"
            "    assert two_sum([3, 2, 4], 6) == (1, 2)\n\n"
            "def test_negative_numbers():\n"
            "    assert two_sum([-3, 4, 3, 90], 0) == (0, 2)\n\n"
            "def test_duplicates():\n"
            "    assert two_sum([3, 3], 6) == (0, 1)\n\n"
            "def test_smallest_i_j():\n"
            "    assert two_sum([1, 5, 3, 5], 8) == (1, 2)\n"
        ),
    },
]

TASKS_BY_ID = {t["task_id"]: t for t in TASKS}
