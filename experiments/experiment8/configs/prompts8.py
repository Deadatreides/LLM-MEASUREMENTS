"""
Section 1/9: P (flat) and D (decomposed) prompt templates. Same
underlying task in both modes -- only the requested output shape differs.
D mode uses a FIXED, parseable section-header format (### A<n>_NAME) so
parsing never has to guess artifact boundaries from prose.
"""

MATH_P_TEMPLATE = (
    "{question}\n\n"
    "Solve this problem. Show your work, then clearly state your final numeric answer."
)

MATH_D_TEMPLATE = (
    "{question}\n\n"
    "Answer using EXACTLY this structure, with each section under its own header line, "
    "nothing before the first header:\n\n"
    "### A1_METHOD\n"
    "State which method/formula applies to this problem and briefly why.\n\n"
    "### A2_VALUES\n"
    "State the specific numeric values from the problem that this method needs.\n\n"
    "### A3_COMPUTATION\n"
    "Show the arithmetic using the method from A1 and the values from A2.\n\n"
    "### A4_FINAL_ANSWER\n"
    "State the final numeric answer, nothing else in this section."
)

CODE_P_TEMPLATE = (
    "{question}\n\n"
    "Write a complete, correct Python function that solves this. Output only the code "
    "in a single ```python code block, no explanation."
)

CODE_D_TEMPLATE = (
    "{question}\n\n"
    "Answer using EXACTLY this structure, with each section under its own header line, "
    "nothing before the first header:\n\n"
    "### A1_CONTRACT\n"
    "A single JSON object with fields: function_name, arguments (list of "
    "{{\"name\":str,\"type\":str}}), return_type, behavior (short string).\n\n"
    "### A2_EDGE_CASES\n"
    "A short list of sentences, each stating what the function should do for one "
    "specific edge case (e.g. \"An empty string is considered a palindrome.\"), not just "
    "a bare list of case names.\n\n"
    "### A3_IMPLEMENTATION\n"
    "The complete Python implementation in a single ```python code block."
)

MATH_MAX_TOKENS_P = 400
MATH_MAX_TOKENS_D = 700
CODE_MAX_TOKENS_P = 500
CODE_MAX_TOKENS_D = 900
