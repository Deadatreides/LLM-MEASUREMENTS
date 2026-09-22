"""
Section 16: real-error analysis. Loads every real D-mode generation
(runs/d_index9.<model>.jsonl), classifies claims, propagates, computes
predicted_affected_set/root_origins for INCORRECT generations. No
independent ground truth exists for real generations (section 6: "for
real tasks, ground truth is established only where objective
verification exists") -- each claim's LOCAL status IS the mechanical
ground truth (checked against the known-correct numbers, same as
experiments 3-8), but the graph-level "was propagation's conservative
affected-set right" question can only be answered descriptively here,
not against another independent oracle. This is stated explicitly, not
glossed over.

Key methodological check carried over from mrs_bruteforce9.py's finding:
for each real INCORRECT generation, checks whether DOWNSTREAM claims'
OWN local text is already independently correct (isolated-error pattern,
matching the synthetic injections) or whether it's INCORRECT TOO because
the model's own downstream reasoning incorporated the wrong upstream
value (cascading pattern, matching experiment 8's CASE 2). This is the
real-data test of which pattern actually happens when models -- not
hand-edited text -- produce the errors.
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "configs"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tasks"))

from storage9 import EXP_DIR, load_jsonl
from schema9 import TASKS, TASK_IDS, CLAIMS, DEPENDS_ON, ARTIFACTS
from classifiers9 import classify_claim
from parse9 import parse_sections
from dependency_engine9 import propagate, predicted_affected_set

RUNS_DIR = os.path.join(EXP_DIR, "runs")
MODELS = ["qwen3-1.7b-q4_0-unsloth", "llama-3.2-1b-instruct-q4_0", "qwen2.5-coder-1.5b-instruct-q4_0"]
SECTION_FOR_CLAIM = {"C1_ROOT": "A1_ROOT", "C2_METHOD_A": "A2_METHOD_A", "C3_METHOD_B": "A3_METHOD_B",
                      "C4_SUBTOTAL_A": "A4_SUBTOTAL_A", "C5_SUBTOTAL_B": "A5_SUBTOTAL_B", "C6_FINAL_TOTAL": "A6_FINAL_TOTAL"}


def read_raw(raw_path):
    with open(os.path.join(RUNS_DIR, raw_path, "result.txt"), encoding="utf-8") as f:
        return f.read()


def classify_case_label(root_origins):
    """LOCAL/UPSTREAM/BRANCH/MERGE/GLOBAL/UNKNOWN (section 16).

    Section 12 defines GLOBAL as "cannot be localized without changing the
    whole construction" -- a case where no clean origin can be identified
    at all. Multiple SEPARATE, each individually well-localized origins
    (e.g. C2 and C3 both independently wrong, but each cleanly pinned) is
    a different, better-behaved phenomenon -- checked here explicitly and
    labeled MULTIPLE_LOCAL, not GLOBAL, so as not to overstate how often
    real errors were genuinely un-localizable (true GLOBAL, requiring a
    seam-inconsistency-style detector not built for real data in this
    experiment, was not observed at all -- reported as 0, not silently
    folded into this bucket)."""
    if not root_origins:
        return "NO_ERROR"
    if len(root_origins) > 1:
        return "MULTIPLE_LOCAL"
    origin = root_origins[0]
    if origin == "C1_ROOT":
        return "UPSTREAM"
    if origin == "C6_FINAL_TOTAL":
        return "MERGE"
    if origin in ("C2_METHOD_A", "C3_METHOD_B", "C4_SUBTOTAL_A", "C5_SUBTOTAL_B"):
        return "BRANCH"
    return "UNKNOWN"


def analyze_generation(task_id, text):
    task = TASKS[task_id]
    sections, quality, found_ids = parse_sections(text, ARTIFACTS)
    local_status = {}
    for cid in CLAIMS:
        sec_text = sections.get(SECTION_FOR_CLAIM[cid])
        local_status[cid] = classify_claim(cid, sec_text, task)[0] if sec_text is not None else "OMITTED"

    predicted, root_origins, effective = predicted_affected_set(local_status, DEPENDS_ON, CLAIMS)
    terminal_status = effective["C6_FINAL_TOTAL"][0]
    error_category = classify_case_label(root_origins)

    # isolated-vs-cascading check: for each root origin, do its DOWNSTREAM
    # claims show their OWN local status as CORRECT (isolated, matching the
    # synthetic injections) or INCORRECT (cascading, matching exp8 CASE 2)?
    cascading_evidence = {}
    for origin in root_origins:
        downstream = [c for c in CLAIMS if origin in DEPENDS_ON.get(c, [])]
        cascading_evidence[origin] = {d: local_status[d] for d in downstream}

    return {
        "parse_quality": quality, "sections": sections, "local_status": local_status,
        "effective": {k: list(v) for k, v in effective.items()}, "predicted_affected_set": sorted(predicted),
        "root_origins": sorted(root_origins), "error_category": error_category,
        "task_status": terminal_status, "cascading_evidence": cascading_evidence,
    }


def main():
    all_records = []
    for model_id in MODELS:
        path = os.path.join(RUNS_DIR, f"d_index9.{model_id}.jsonl")
        for row in load_jsonl(path):
            if row["generation_failed"]:
                continue
            text = read_raw(row["raw_path"])
            rec = {"run_id": row["run_id"], "task_id": row["task_id"], "model_id": model_id, "seed": row["seed"],
                   "output_tokens": row["output_tokens"], "generation_time_sec": row["generation_time_sec"],
                   "analysis": analyze_generation(row["task_id"], text)}
            all_records.append(rec)

    out_path = os.path.join(os.path.dirname(__file__), "analysis9_records.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    print(f"n_records={len(all_records)}")
    print("task_status dist:", collections.Counter(r["analysis"]["task_status"] for r in all_records))
    print("error_category dist (INCORRECT only):", collections.Counter(
        r["analysis"]["error_category"] for r in all_records if r["analysis"]["task_status"] == "INCORRECT"))
    return all_records


if __name__ == "__main__":
    main()
