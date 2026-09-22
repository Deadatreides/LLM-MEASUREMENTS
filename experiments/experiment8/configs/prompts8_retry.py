"""
Section 14: D_RETRY prompt -- regenerates ONLY the MRS-affected artifacts
(the origin claim's artifact and everything downstream of it), showing
the untouched upstream artifacts as GIVEN, FIXED context. Never asks the
model to regenerate an artifact outside the MRS (section 14's explicit
rule: don't touch downstream-invalidating claims without also fixing
what depends on them, but likewise don't touch artifacts that were NOT
found to be part of the affected subgraph).
"""

MATH_RETRY_TEMPLATE = (
    "{question}\n\n"
    "The following has already been correctly determined and must not be changed:\n"
    "{fixed_context}\n\n"
    "Continue the answer using EXACTLY this structure for the remaining parts, each "
    "section under its own header line:\n\n{remaining_sections}"
)

MATH_SECTION_PROMPTS = {
    "A2_VALUES": "### A2_VALUES\nState the specific numeric values from the problem that this method needs.\n\n",
    "A3_COMPUTATION": "### A3_COMPUTATION\nShow the arithmetic using the method and values above.\n\n",
    "A4_FINAL_ANSWER": "### A4_FINAL_ANSWER\nState the final numeric answer, nothing else in this section.",
}
MATH_CONTEXT_LABELS = {"A1_METHOD": "METHOD", "A2_VALUES": "VALUES", "A3_COMPUTATION": "COMPUTATION", "A4_FINAL_ANSWER": "FINAL ANSWER"}

CODE_RETRY_TEMPLATE = (
    "{question}\n\n"
    "The following has already been correctly determined and must not be changed:\n"
    "{fixed_context}\n\n"
    "Provide ONLY the remaining section:\n\n"
    "### A3_IMPLEMENTATION\n"
    "The complete Python implementation in a single ```python code block."
)
CODE_CONTEXT_LABELS = {"A1_CONTRACT": "CONTRACT", "A2_EDGE_CASES": "EDGE CASES"}

MATH_MAX_TOKENS_RETRY = 500
CODE_MAX_TOKENS_RETRY = 500
