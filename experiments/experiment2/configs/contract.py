"""
Mechanical checks and extraction for the A1 CONTRACT artifact (spec
sections 2-3). No LLM is used to validate A1 -- everything here is
structural/string-based. Fields that genuinely cannot be checked
mechanically are left UNVERIFIED rather than guessed.
"""
import re

from ast_checks import extract_json_object

REQUIRED_FIELDS = [
    "function_name",
    "arguments",
    "return_type",
    "behavior",
    "edge_cases",
    "constraints",
    "input_format",
    "output_format",
]

_LEAK_PATTERNS = re.compile(r"\bdef\s+\w+\s*\(|\breturn\s+\w|\bassert\s+")


def _flatten_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _flatten_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _flatten_strings(v)


def check_contract(raw_text, expected_function_name):
    """Returns a dict: parse status, required-field presence, leak checks,
    name match, plus the extracted characteristics from spec section 2."""
    result = {
        "json_parse_status": None,  # PASS / FAIL
        "missing_fields": [],
        "has_all_required_fields": None,
        "contains_implementation_leak": None,
        "contains_test_leak": None,
        "contract_matches_expected_name": None,
        "contract_function_name": None,
        "contract_argument_count": None,
        "contract_argument_names": None,
        "contract_return_description": None,
        "contract_constraints_count": None,
        "seam_a1_self_status": None,  # MECHANICAL_PASS / MECHANICAL_FAIL / UNVERIFIED
        "error_class": None,
        "error_signature": None,
    }

    obj, raw_json = extract_json_object(raw_text)
    if obj is None or not isinstance(obj, dict):
        result["json_parse_status"] = "FAIL"
        result["seam_a1_self_status"] = "MECHANICAL_FAIL"
        result["error_class"] = "CONTRACT_PARSE_ERROR"
        result["error_signature"] = "no_parseable_json_object"
        return result

    result["json_parse_status"] = "PASS"
    missing = [f for f in REQUIRED_FIELDS if f not in obj or obj[f] in (None, "", [])]
    result["missing_fields"] = missing
    result["has_all_required_fields"] = len(missing) == 0

    all_text = " ".join(_flatten_strings(obj)).lower()
    result["contains_implementation_leak"] = bool(_LEAK_PATTERNS.search(all_text)) and (
        "def " in all_text
    )
    result["contains_test_leak"] = "assert " in all_text or "def test_" in all_text

    fn_name = obj.get("function_name")
    if isinstance(fn_name, str):
        result["contract_function_name"] = fn_name.strip()
        result["contract_matches_expected_name"] = fn_name.strip() == expected_function_name
    else:
        result["contract_matches_expected_name"] = False

    args = obj.get("arguments")
    if isinstance(args, list):
        result["contract_argument_count"] = len(args)
        names = []
        for a in args:
            if isinstance(a, dict) and "name" in a:
                names.append(a["name"])
            elif isinstance(a, str):
                names.append(a)
        result["contract_argument_names"] = names
    else:
        result["contract_argument_count"] = None  # UNVERIFIED shape

    rd = obj.get("return_type") or obj.get("behavior")
    result["contract_return_description"] = rd if isinstance(rd, str) else None

    constraints = obj.get("constraints")
    result["contract_constraints_count"] = len(constraints) if isinstance(constraints, list) else None

    # overall mechanical seam verdict for A1 itself
    if not result["has_all_required_fields"]:
        result["seam_a1_self_status"] = "MECHANICAL_FAIL"
        result["error_class"] = "CONTRACT_MISSING_FIELDS"
        result["error_signature"] = f"missing:{','.join(sorted(missing))}"
    elif result["contains_implementation_leak"]:
        result["seam_a1_self_status"] = "MECHANICAL_FAIL"
        result["error_class"] = "CONTRACT_IMPLEMENTATION_LEAK"
        result["error_signature"] = "def_in_contract_fields"
    elif result["contains_test_leak"]:
        result["seam_a1_self_status"] = "MECHANICAL_FAIL"
        result["error_class"] = "CONTRACT_TEST_LEAK"
        result["error_signature"] = "assert_in_contract_fields"
    elif not result["contract_matches_expected_name"]:
        result["seam_a1_self_status"] = "MECHANICAL_FAIL"
        result["error_class"] = "CONTRACT_NAME_MISMATCH"
        result["error_signature"] = f"got:{result['contract_function_name']}"
    else:
        result["seam_a1_self_status"] = "MECHANICAL_PASS"

    return result
