"""
Builds the hypothesis space for Experiment 4 from Experiment 3's already
mechanically-classified claims (runs/claims_index.jsonl in experiment3),
without any LLM judge.

Design note (grounded in the actual data, checked before writing this):
for structural claims (function name, arg count) and type claims, the
hypothesis space is naturally multi-way and exactly determined by the
literal value a generation asserted -- no normalization ambiguity, no
judgment call needed. For the harder semantic/edge-case claims, a manual
audit of every claim's real evidence text (done while preparing this
experiment, see experiment3/REPORT3.md errata) found that this dataset's
actual observed hypothesis diversity is much narrower than the spec's
illustrative example (H1..H5): in practice at most one coherent CORRECT
reading and at most one coherent INCORRECT reading ever recur per claim,
plus occasional AMBIGUOUS text. Given that, this experiment does NOT
invent additional named hypotheses that were never observed (spec: "не
создавать искусственные claims ради увеличения N") -- CORRECT/INCORRECT
map directly to one named hypothesis each, AMBIGUOUS is its own
AMBIGUOUS_EQUIVALENCE bucket per spec section 7, and OMITTED means no
hypothesis was stated at all (excluded from the hypothesis pool, not
treated as a hypothesis).
"""
import json
import re
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "experiment3", "configs"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "experiment3", "tasks"))

from ast_checks import extract_json_object  # noqa: E402

EXP3_RUNS = r"<PROJECT_ROOT>\trace-probe\experiment3\runs"

TYPE_CANON = [
    (re.compile(r"\bstr(ing)?\b", re.I), "STRING"),
    (re.compile(r"\bbool(ean)?\b", re.I), "BOOL"),
    (re.compile(r"\bint(eger)?\b", re.I), "INT"),
    (re.compile(r"\bfloat\b", re.I), "FLOAT"),
    (re.compile(r"\btuple|pair\b", re.I), "TUPLE"),
    (re.compile(r"\blist|array|sequence\b", re.I), "LIST"),
    (re.compile(r"\bdict\b", re.I), "DICT"),
    (re.compile(r"\bnone|null\b", re.I), "NONE"),
]


def canon_type(text):
    if not text:
        return "OMITTED"
    for pat, label in TYPE_CANON:
        if pat.search(text):
            return label
    return "OTHER_TYPE"


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_a1_runs():
    import glob
    rows = []
    for p in sorted(glob.glob(os.path.join(EXP3_RUNS, "a1_index.*.jsonl"))):
        rows.extend(load_jsonl(p))
    return [r for r in rows if not r["generation_failed"]]


def load_claims():
    return load_jsonl(os.path.join(EXP3_RUNS, "claims_index.jsonl"))


def build_hypothesis_rows():
    """Returns a list of hypothesis rows, one per (run, claim) instance
    that actually stated something (OMITTED rows are dropped from the
    hypothesis pool, but their count is tracked separately upstream via
    claims_index.jsonl, which is never modified)."""
    runs = load_a1_runs()
    runs_by_id = {r["run_id"]: r for r in runs}
    claim_rows = load_claims()

    out = []
    for c in claim_rows:
        run = runs_by_id.get(c["run_id"])
        if run is None:
            continue
        cls = c["classification"]
        claim_id = c["claim_id"]

        if cls == "OMITTED":
            continue  # no hypothesis stated

        if claim_id in ("C1_NAME",):
            hyp_id = f"NAME={c['evidence']}" if c["evidence"] else "NAME=UNKNOWN"
        elif claim_id in ("C2_ARGC",):
            hyp_id = f"ARGC={c['evidence']}" if c["evidence"] else "ARGC=UNKNOWN"
        elif claim_id.endswith("_TYPE") or claim_id in ("C3_RETURN_TYPE", "C4_RETURN_TYPE"):
            hyp_id = f"TYPE={canon_type(c['evidence'])}"
        elif cls == "AMBIGUOUS":
            hyp_id = "AMBIGUOUS_EQUIVALENCE"
        else:  # CORRECT or INCORRECT text claim -> exactly one named hypothesis each (see module docstring)
            hyp_id = f"{cls}_1"

        out.append({
            "task_id": c["task_id"], "claim_id": claim_id, "run_id": c["run_id"],
            "model_id": run["model_id"], "temperature": run["temperature"],
            "prompt_id": run["prompt_id"], "seed": run["seed"],
            "raw_claim_evidence": c["evidence"],
            "hypothesis_id": hyp_id, "status": cls,
        })
    return out


if __name__ == "__main__":
    rows = build_hypothesis_rows()
    out_dir = os.path.join(os.path.dirname(__file__), "..", "runs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "hypotheses_index.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"n_hypothesis_rows={len(rows)} -> {out_path}")
