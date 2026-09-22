"""Section 4/23: structured D-mode prompt for the 6-node branching DAG.
Same fixed-header format discipline as experiments 7-8 (never guess
section boundaries from prose)."""

D_TEMPLATE = (
    "{question}\n\n"
    "Answer using EXACTLY this structure, with each section under its own header line, "
    "nothing before the first header:\n\n"
    "### A1_ROOT\n"
    "State that this problem has two parts that must be handled separately, naming both.\n\n"
    "### A2_METHOD_A\n"
    "State the quantity and rate that apply to the first part.\n\n"
    "### A3_METHOD_B\n"
    "State the quantity and rate that apply to the second part.\n\n"
    "### A4_SUBTOTAL_A\n"
    "Show the arithmetic for the first part's subtotal.\n\n"
    "### A5_SUBTOTAL_B\n"
    "Show the arithmetic for the second part's subtotal.\n\n"
    "### A6_FINAL_TOTAL\n"
    "Add both subtotals and state the final numeric total, nothing else in this section."
)

MAX_TOKENS_D = 700
