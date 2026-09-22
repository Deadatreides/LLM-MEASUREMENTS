"""
Role prompts for Experiment 2. Each role is instructed to do exactly one
job and not to do the others' jobs (spec section 17). Kept short on
purpose -- the point is separation of responsibility, not elaborate
instructions.
"""

BASELINE_TEMPLATE = (
    "{question}\n"
    "Provide the complete function definition in a single Python code block."
)

A1_CONTRACT_TEMPLATE = (
    "You are writing a strict function CONTRACT, not an implementation.\n\n"
    "Task: {question}\n\n"
    "Output ONLY a single JSON object with exactly these fields:\n"
    '  "function_name": string,\n'
    '  "arguments": list of {{"name": string, "type": string}},\n'
    '  "return_type": string,\n'
    '  "behavior": short string describing what the function computes,\n'
    '  "edge_cases": list of strings,\n'
    '  "constraints": list of strings,\n'
    '  "input_format": string,\n'
    '  "output_format": string\n\n'
    "Do NOT write any implementation code. Do NOT write any tests. Do NOT "
    "write prose outside the JSON object. Output only the JSON object."
)

A2_TESTS_TEMPLATE = (
    "You are writing TESTS, not an implementation.\n\n"
    "Task: {question}\n\n"
    "Contract (JSON):\n{contract_json}\n\n"
    "Write Python tests for a function named {function_name}. Requirements:\n"
    "- Use plain `assert` statements inside test functions named test_*, each taking no arguments.\n"
    "- Do NOT define or implement {function_name} yourself -- assume it already exists and is available.\n"
    "- Do NOT use pytest, unittest, or import the function under test.\n"
    "- Cover the edge cases and constraints listed in the contract.\n"
    "Output only a single Python code block containing the test functions, nothing else."
)

A3_IMPLEMENTATION_TEMPLATE = (
    "You are writing an IMPLEMENTATION, not tests.\n\n"
    "Task: {question}\n\n"
    "Contract (JSON):\n{contract_json}\n\n"
    "Implement the function exactly as specified in the contract above. Output "
    "only a single Python code block containing the function definition (and any "
    "imports/helpers it needs), nothing else. Do not write any test code."
)

A4_PATCH_TEMPLATE = (
    "Your previous implementation of {function_name} failed mechanical checks. "
    "Fix ONLY what is necessary to resolve the failures below -- do not rewrite "
    "unrelated parts of the function.\n\n"
    "Task: {question}\n\n"
    "Contract (JSON):\n{contract_json}\n\n"
    "Previous implementation:\n```python\n{previous_code}\n```\n\n"
    "Mechanical failure report:\n{failure_report}\n\n"
    "Output only a single corrected Python code block containing the full "
    "function definition, nothing else."
)
