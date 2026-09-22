"""
Three substantially different phrasings of the same A1/CONTRACT generation
task (spec section 12). Same JSON schema and same underlying task in all
three -- only the framing/emphasis of the instruction differs.
"""

_SCHEMA = (
    '  "function_name": string,\n'
    '  "arguments": list of {{"name": string, "type": string}},\n'
    '  "return_type": string,\n'
    '  "behavior": short string describing what the function computes,\n'
    '  "edge_cases": list of strings,\n'
    '  "constraints": list of strings,\n'
    '  "input_format": string,\n'
    '  "output_format": string'
)

P1_NORMAL = (
    "Write a specification (contract) for the following task, not an implementation.\n\n"
    "Task: {question}\n\n"
    "Output ONLY a single JSON object with exactly these fields:\n" + _SCHEMA + "\n\n"
    "Do NOT write any implementation code or tests. Output only the JSON object."
)

P2_STRICT_FORMAL = (
    "You are producing a FORMAL specification document for the function described below. "
    "Precision matters: every field must be exact and unambiguous. Do not write code.\n\n"
    "Task: {question}\n\n"
    "Output ONLY a single JSON object with exactly these fields (be precise and complete, "
    "do not leave any field vague):\n" + _SCHEMA + "\n\n"
    "Output only the JSON object, no commentary, no code, no tests."
)

P3_EDGE_CASE_FOCUSED = (
    "Write a specification (contract) for the following task, not an implementation. Pay "
    "special attention to edge cases and boundary conditions -- think carefully about "
    "empty inputs, smallest/largest values, and any subtlety in how the task is worded, "
    "and list every edge case you can identify.\n\n"
    "Task: {question}\n\n"
    "Output ONLY a single JSON object with exactly these fields:\n" + _SCHEMA + "\n\n"
    "Do NOT write any implementation code or tests. Output only the JSON object."
)

PROMPTS = {"P1": P1_NORMAL, "P2": P2_STRICT_FORMAL, "P3": P3_EDGE_CASE_FOCUSED}
