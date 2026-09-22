"""
Per-task CLAIM checklists for Experiment 3. Each claim is checked
mechanically (regex over specific JSON contract fields), never by an LLM
judge. Classification per claim, applied to a single A1 (never comparing
two A1s directly -- see configs/claims.py for that):

  CORRECT   -- text matches a pattern that agrees with the known reference fact
  INCORRECT -- text matches a pattern that contradicts the reference fact
  OMITTED   -- neither pattern found (claim not addressed at all)
  AMBIGUOUS -- BOTH patterns found (contradictory signal within the same A1),
               or the field needed to check it is unparseable

Claim types map onto spec section 8's categories:
  SIGNATURE  (name, arg count)      -- claim types 2
  TYPE       (arg/return types)     -- claim type 3
  SEMANTICS  (core behavior)        -- claim type 4 (the interesting one)
  EDGE_CASE  (boundary behavior)    -- claim type 5 (the interesting one)

"structural" claims are checked against the parsed contract dict directly
(function_name, arguments length). "text" claims are checked against a
concatenation of specific string fields via regex.
"""
import re


def _re(pattern):
    return re.compile(pattern, re.IGNORECASE)


_Y_NEG_RE = re.compile(r"not|never|exclud\w*|isn'?t|consonant")
_Y_POS_RE = re.compile(r"\b(is|as|counted|considered|included)\b[^.]{0,15}vowel")


def _classify_fib_value(n_val, expected_val):
    """Returns a classifier for 'fibonacci(n_val) == expected_val'.

    A plain proximity regex got this wrong: A1s that merely LIST "n = 0"
    and "n = 1" as edge-case labels (without ever stating what the
    function returns for them) were misread as asserting a specific wrong
    value, because some OTHER digit (e.g. from an unrelated "n >= 0"
    constraint) happened to fall within the match window. Fixed by
    requiring an explicit equals/is/returns connector directly between the
    input reference and the asserted output digit -- a bare list of edge
    case labels with no connector now correctly falls through to OMITTED.
    """
    trigger = re.compile(
        rf"(?:fibonacci\({n_val}\)|\bn\s*(?:=|==)\s*{n_val}\b)\s*(?:=|==|is|returns?)\s*(\d+)",
        re.IGNORECASE,
    )

    def _fn(text):
        m = trigger.search(text)
        if not m:
            return "OMITTED", None
        got = m.group(1)
        if got == str(expected_val):
            return "CORRECT", m.group(0)
        return "INCORRECT", m.group(0)

    return _fn


_TWO_BOUNDARY_PRE_RE = re.compile(r"(less than|below|under|smaller than|<\s*)\s*$")
_TWO_POS_RE = re.compile(r"^[^.]{0,30}(is prime|prime number|smallest prime)", re.IGNORECASE)
_TWO_NEG_RE = re.compile(r"^[^.]{0,30}(not prime|isn'?t prime|non-?prime)", re.IGNORECASE)


def _classify_two_is_prime(text):
    """Custom classifier for is_prime's '2 is prime' edge case.

    A plain proximity regex got this wrong too: phrasing like "numbers
    less than 2 are not prime" contains "...2...not prime" and was
    misread as the model claiming 2 itself is not prime, when it is
    actually correctly describing the boundary BELOW 2. Fixed by skipping
    any "2" that is immediately preceded by a "less than / below / <"
    style boundary phrase before checking what follows it.
    """
    for m in re.finditer(r"\b2\b", text):
        pre = text[max(0, m.start() - 20): m.start()]
        if _TWO_BOUNDARY_PRE_RE.search(pre):
            continue
        after = text[m.end(): m.end() + 35]
        if _TWO_POS_RE.match(after):
            return "CORRECT", text[m.start():m.end() + 30]
        if _TWO_NEG_RE.match(after):
            return "INCORRECT", text[m.start():m.end() + 30]
    return "OMITTED", None


def _classify_y_not_vowel(text):
    """Custom classifier for CODE_04/C6_EDGE_Y ('y' is never a vowel).

    A plain correct_re/incorrect_re pair got this wrong: phrasings like
    "excluding 'y' as a vowel" are a CORRECT statement (y is excluded from
    being a vowel) but contain the substring "...y...as a vowel", which a
    naive incorrect_re (matching "y ... is/as/included ... vowel") flags as
    INCORRECT. Found via manual inspection of real generations while
    building Experiment 4's hypothesis space (see REPORT3.md errata) --
    fixed here so both experiments use the corrected classification.

    Fix: for each mention of "y", look at a window around it and check for
    a negation word BEFORE deciding a positive-inclusion phrase means "y IS
    a vowel". Negation wins if both appear in the same window.
    """
    for m in re.finditer(r"\by\b", text):
        window = text[max(0, m.start() - 40): m.end() + 40]
        has_neg = _Y_NEG_RE.search(window)
        has_pos = _Y_POS_RE.search(window)
        if has_neg:
            return "CORRECT", window
        if has_pos:
            return "INCORRECT", window
    return "OMITTED", None


# Each entry: claim_id, claim_text, claim_type, kind ("structural_name" /
# "structural_argc" / "text"), text_fields (for kind="text"), correct_re,
# incorrect_re (may be None -- OMITTED is then the only negative outcome).

CLAIMS_BY_TASK = {
    "CODE_01": [  # is_palindrome(s): case-insensitive; spaces/punctuation NOT stripped
        dict(claim_id="C1_NAME", claim_text="function name is is_palindrome", claim_type="SIGNATURE", kind="structural_name"),
        dict(claim_id="C2_ARGC", claim_text="exactly 1 argument", claim_type="SIGNATURE", kind="structural_argc", expected_argc=1),
        dict(claim_id="C3_ARG_TYPE", claim_text="argument type is string", claim_type="TYPE", kind="text_args",
             correct_re=_re(r"\bstr(ing)?\b"), incorrect_re=_re(r"\b(int|float|bool|list|dict)\b")),
        dict(claim_id="C4_RETURN_TYPE", claim_text="return type is bool", claim_type="TYPE", kind="text_return",
             correct_re=_re(r"\bbool(ean)?\b"), incorrect_re=_re(r"\b(str|int|float|list|dict|none)\b")),
        dict(claim_id="C5_SEMANTICS_CASE", claim_text="comparison is case-insensitive", claim_type="SEMANTICS", kind="text_all",
             correct_re=_re(r"case[- ]?insensit|ignor\w* case|regardless of case|lower ?case"),
             incorrect_re=_re(r"case[- ]?sensitive|exact(ly)? case|preserv\w* case")),
        dict(claim_id="C6_EDGE_EMPTY", claim_text="empty string is treated as a palindrome (True)", claim_type="EDGE_CASE", kind="text_all",
             correct_re=_re(r"empty\s*(string)?[^.]{0,40}(true|palindrome|considered)"),
             incorrect_re=_re(r"empty\s*(string)?[^.]{0,40}(false|not a palindrome|error|exception)")),
        dict(claim_id="C7_EDGE_SPACES", claim_text="spaces/punctuation are NOT stripped before comparison", claim_type="EDGE_CASE", kind="text_all",
             correct_re=_re(r"(space|punctuation)\w*[^.]{0,50}(not\s+\w*\s*(remov|strip|ignor)|preserv|kept|keep|include)"),
             incorrect_re=_re(r"(space|punctuation)\w*[^.]{0,50}(remov|strip|ignor(e|ed|ing)|exclud)")),
    ],
    "CODE_02": [  # sum_even(nums): sum of even numbers; 0 if none/empty
        dict(claim_id="C1_NAME", claim_text="function name is sum_even", claim_type="SIGNATURE", kind="structural_name"),
        dict(claim_id="C2_ARGC", claim_text="exactly 1 argument", claim_type="SIGNATURE", kind="structural_argc", expected_argc=1),
        dict(claim_id="C3_ARG_TYPE", claim_text="argument type is list of int", claim_type="TYPE", kind="text_args",
             correct_re=_re(r"list|array|sequence"), incorrect_re=_re(r"\b(str|dict|set)\b")),
        dict(claim_id="C4_RETURN_TYPE", claim_text="return type is int", claim_type="TYPE", kind="text_return",
             correct_re=_re(r"\bint(eger)?\b"), incorrect_re=_re(r"\b(str|bool|list|float)\b")),
        dict(claim_id="C5_SEMANTICS", claim_text="sums only even numbers (not odd)", claim_type="SEMANTICS", kind="text_all",
             correct_re=_re(r"even"), incorrect_re=_re(r"\bodd numbers?\b[^.]{0,20}sum|\bsum\w*[^.]{0,20}\bodd\b")),
        dict(claim_id="C6_EDGE_EMPTY", claim_text="empty list returns 0", claim_type="EDGE_CASE", kind="text_all",
             correct_re=_re(r"empty[^.]{0,40}(0|zero)"), incorrect_re=_re(r"empty[^.]{0,40}(error|exception|none|null)")),
        dict(claim_id="C7_EDGE_NO_EVEN", claim_text="list with no even numbers returns 0", claim_type="EDGE_CASE", kind="text_all",
             correct_re=_re(r"(no even|all odd|none.{0,15}even)[^.]{0,30}(0|zero)"), incorrect_re=None),
    ],
    "CODE_03": [  # fibonacci(n): 0-indexed, fib(0)=0, fib(1)=1
        dict(claim_id="C1_NAME", claim_text="function name is fibonacci", claim_type="SIGNATURE", kind="structural_name"),
        dict(claim_id="C2_ARGC", claim_text="exactly 1 argument", claim_type="SIGNATURE", kind="structural_argc", expected_argc=1),
        dict(claim_id="C3_ARG_TYPE", claim_text="argument type is int", claim_type="TYPE", kind="text_args",
             correct_re=_re(r"\bint(eger)?\b"), incorrect_re=_re(r"\b(str|float|list)\b")),
        dict(claim_id="C4_RETURN_TYPE", claim_text="return type is int", claim_type="TYPE", kind="text_return",
             correct_re=_re(r"\bint(eger)?\b"), incorrect_re=_re(r"\b(str|bool|list|float)\b")),
        dict(claim_id="C5_EDGE_ZERO", claim_text="fibonacci(0) == 0", claim_type="EDGE_CASE", kind="text_all_custom",
             custom_fn=_classify_fib_value(0, 0)),
        dict(claim_id="C6_EDGE_ONE", claim_text="fibonacci(1) == 1", claim_type="EDGE_CASE", kind="text_all_custom",
             custom_fn=_classify_fib_value(1, 1)),
        dict(claim_id="C7_SEMANTICS_RECURRENCE", claim_text="each term is the sum of the two preceding terms", claim_type="SEMANTICS", kind="text_all",
             correct_re=_re(r"sum[^.]{0,25}(two|preceding|previous)|preceding[^.]{0,15}(two|sum)"), incorrect_re=None),
    ],
    "CODE_04": [  # count_vowels(s): case-insensitive a,e,i,o,u; y NEVER a vowel
        dict(claim_id="C1_NAME", claim_text="function name is count_vowels", claim_type="SIGNATURE", kind="structural_name"),
        dict(claim_id="C2_ARGC", claim_text="exactly 1 argument", claim_type="SIGNATURE", kind="structural_argc", expected_argc=1),
        dict(claim_id="C3_ARG_TYPE", claim_text="argument type is string", claim_type="TYPE", kind="text_args",
             correct_re=_re(r"\bstr(ing)?\b"), incorrect_re=_re(r"\b(int|list|bool)\b")),
        dict(claim_id="C4_RETURN_TYPE", claim_text="return type is int", claim_type="TYPE", kind="text_return",
             correct_re=_re(r"\bint(eger)?\b"), incorrect_re=_re(r"\b(str|bool|list|float)\b")),
        dict(claim_id="C5_SEMANTICS_CASE", claim_text="counting is case-insensitive (both upper and lower)", claim_type="SEMANTICS", kind="text_all",
             correct_re=_re(r"case[- ]?insensit|both\s*(upper|lower)|regardless of case|upper.{0,10}and.{0,10}lower"),
             incorrect_re=_re(r"case[- ]?sensitive|only\s*lower|only\s*upper")),
        dict(claim_id="C6_EDGE_Y", claim_text="'y' is never counted as a vowel", claim_type="EDGE_CASE", kind="text_all_custom",
             custom_fn=_classify_y_not_vowel),
        dict(claim_id="C7_EDGE_EMPTY", claim_text="empty string returns 0", claim_type="EDGE_CASE", kind="text_all",
             correct_re=_re(r"empty[^.]{0,40}(0|zero)"), incorrect_re=_re(r"empty[^.]{0,40}(error|exception|none)")),
    ],
    "CODE_05": [  # is_prime(n): n<2 not prime
        dict(claim_id="C1_NAME", claim_text="function name is is_prime", claim_type="SIGNATURE", kind="structural_name"),
        dict(claim_id="C2_ARGC", claim_text="exactly 1 argument", claim_type="SIGNATURE", kind="structural_argc", expected_argc=1),
        dict(claim_id="C3_ARG_TYPE", claim_text="argument type is int", claim_type="TYPE", kind="text_args",
             correct_re=_re(r"\bint(eger)?\b"), incorrect_re=_re(r"\b(str|float|list)\b")),
        dict(claim_id="C4_RETURN_TYPE", claim_text="return type is bool", claim_type="TYPE", kind="text_return",
             correct_re=_re(r"\bbool(ean)?\b"), incorrect_re=_re(r"\b(str|int|list)\b")),
        dict(claim_id="C5_EDGE_LT2", claim_text="numbers less than 2 are not prime", claim_type="EDGE_CASE", kind="text_all",
             correct_re=_re(r"(less than 2|<\s*2|n\s*<\s*2)[^.]{0,30}(not prime|false)"), incorrect_re=None),
        dict(claim_id="C6_EDGE_TWO", claim_text="2 is prime (the smallest prime)", claim_type="EDGE_CASE", kind="text_all_custom",
             custom_fn=_classify_two_is_prime),
        dict(claim_id="C7_EDGE_NEGATIVE", claim_text="negative numbers are not prime", claim_type="EDGE_CASE", kind="text_all",
             correct_re=_re(r"negative[^.]{0,30}(not prime|false)"), incorrect_re=None),
    ],
    "CODE_06": [  # two_sum(nums, target): returns 0-based INDICES, smallest i then j, exactly one solution assumed
        dict(claim_id="C1_NAME", claim_text="function name is two_sum", claim_type="SIGNATURE", kind="structural_name"),
        dict(claim_id="C2_ARGC", claim_text="exactly 2 arguments", claim_type="SIGNATURE", kind="structural_argc", expected_argc=2),
        dict(claim_id="C3_RETURN_TYPE", claim_text="return type is a tuple/pair of indices", claim_type="TYPE", kind="text_return",
             correct_re=_re(r"tuple|pair"), incorrect_re=_re(r"\bint\b|\bbool\b|\bstr\b")),
        dict(claim_id="C4_SEMANTICS_INDICES", claim_text="returns indices, not the values themselves", claim_type="SEMANTICS", kind="text_all",
             correct_re=_re(r"index|indices|position"), incorrect_re=None),
        dict(claim_id="C5_SEMANTICS_SMALLEST_I", claim_text="returns the pair with the smallest valid i (tie-break rule)", claim_type="SEMANTICS", kind="text_all",
             correct_re=_re(r"smallest|first|lowest|earliest"), incorrect_re=None),
        dict(claim_id="C6_EDGE_ONE_SOLUTION", claim_text="assumes exactly one valid pair exists", claim_type="EDGE_CASE", kind="text_all",
             correct_re=_re(r"exactly one|assum\w*[^.]{0,25}(one|single|unique)"), incorrect_re=None),
        dict(claim_id="C7_EDGE_NOT_SAME", claim_text="i and j must be different indices (i < j)", claim_type="EDGE_CASE", kind="text_all",
             correct_re=_re(r"i\s*<\s*j|different indices|distinct indices|two different"), incorrect_re=None),
    ],
}
