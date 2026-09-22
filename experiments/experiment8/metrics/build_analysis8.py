"""
Main orchestration: loads every P/D generation (runs/pd_index.<model>.jsonl),
classifies claims, propagates dependencies, computes seams/MRS, and builds
the per-generation + aggregated dataset used by REPORT8.md. Applies the
FROZEN protocol (schema8.py, classifiers8.py, dependency_engine.py,
seams8.py) unchanged -- this script only orchestrates and aggregates.
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "configs"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tasks"))

from storage8 import EXP_DIR, load_jsonl
from schema8 import (MATH_TASKS, MATH_ARTIFACTS, MATH_CLAIMS, MATH_CLAIM_ARTIFACT, MATH_DEPENDS_ON,
                      CODE_ARTIFACTS, CODE_CLAIM_ARTIFACT_BY_TASK, CODE_DEPENDS_ON, TASK_FAMILY, ALL_TASK_IDS)
from tasks_pipeline_code import TASKS_BY_ID as CODE_TASKS_BY_ID
from classifiers8 import classify_math_claim
from claims_code_wrapper import classify_code_claims
from claims_def_code import CLAIMS_BY_TASK as CODE_CLAIMS_BY_TASK
from mech_check_code import classify_c8_tests_pass
from ast_checks import extract_code, extract_json_object
from parse8 import parse_sections
from dependency_engine import propagate, minimal_repair_set, error_taxonomy_label, normalize_status
from seams8 import seam_dependency_status, seam_consistency_math, seam_consistency_code, seam_execution_status
from p_mode_check import check_p_mode_math, check_p_mode_code

RUNS_DIR = os.path.join(EXP_DIR, "runs")
MODELS = ["qwen3-1.7b-q4_0-unsloth", "llama-3.2-1b-instruct-q4_0", "qwen2.5-coder-1.5b-instruct-q4_0"]

CODE_CLAIM_ORDER = {"CODE_01": ["C1_NAME", "C2_ARGC", "C3_ARG_TYPE", "C4_RETURN_TYPE", "C5_SEMANTICS_CASE", "C6_EDGE_EMPTY", "C7_EDGE_SPACES", "C8_TESTS_PASS"],
                     "CODE_06": ["C1_NAME", "C2_ARGC", "C3_RETURN_TYPE", "C4_SEMANTICS_INDICES", "C5_SEMANTICS_SMALLEST_I", "C6_EDGE_ONE_SOLUTION", "C7_EDGE_NOT_SAME", "C8_TESTS_PASS"]}


def read_raw(run_row):
    path = os.path.join(RUNS_DIR, run_row["raw_path"], "result.txt")
    with open(path, encoding="utf-8") as f:
        return f.read()


def analyze_d_math(task_id, text):
    task = MATH_TASKS[task_id]
    sections, quality, found_ids = parse_sections(text, MATH_ARTIFACTS)
    section_map = {"C1_METHOD": "A1_METHOD", "C2_VALUES": "A2_VALUES", "C3_COMPUTATION": "A3_COMPUTATION", "C4_FINAL_ANSWER": "A4_FINAL_ANSWER"}
    local_status = {}
    local_evidence = {}
    for cid in MATH_CLAIMS:
        sec_text = sections.get(section_map[cid])
        if sec_text is None:
            local_status[cid] = "OMITTED"
            local_evidence[cid] = None
        else:
            local_status[cid], local_evidence[cid] = classify_math_claim(task_id, cid, sec_text)

    effective = propagate(local_status, MATH_DEPENDS_ON)
    mrs, root_origins = minimal_repair_set(effective, MATH_DEPENDS_ON, MATH_CLAIMS)
    labels = {cid: error_taxonomy_label(cid, effective, root_origins) for cid in MATH_CLAIMS}
    seam_cons = seam_consistency_math(task, sections, local_status["C1_METHOD"], local_status["C3_COMPUTATION"], local_evidence["C3_COMPUTATION"])
    seam_deps = [s for s in (seam_dependency_status(cid, local_status[cid], effective, MATH_DEPENDS_ON) for cid in MATH_CLAIMS) if s]
    seam_exec = seam_execution_status(local_status["C4_FINAL_ANSWER"])

    terminal_status = effective["C4_FINAL_ANSWER"][0]
    return {
        "family": "MATH", "parse_quality": quality, "sections": sections,
        "local_status": local_status, "local_evidence": local_evidence, "effective": {k: list(v) for k, v in effective.items()},
        "mrs": mrs, "root_origins": root_origins, "error_labels": labels,
        "seam_consistency": seam_cons, "seam_dependency": seam_deps, "seam_execution": seam_exec,
        "task_status": terminal_status, "n_claims": len(MATH_CLAIMS), "n_artifacts": len(MATH_ARTIFACTS),
        "claim_order": MATH_CLAIMS, "artifact_of": MATH_CLAIM_ARTIFACT, "depends_on": MATH_DEPENDS_ON,
    }


def analyze_d_code(task_id, text):
    task = CODE_TASKS_BY_ID[task_id]
    claim_order = CODE_CLAIM_ORDER[task_id]
    sections, quality, found_ids = parse_sections(text, CODE_ARTIFACTS)

    a1_text, a2_text, a3_text = sections.get("A1_CONTRACT"), sections.get("A2_EDGE_CASES"), sections.get("A3_IMPLEMENTATION")
    status1, a1_obj, claim_results = classify_code_claims(a1_text, a2_text, task, CODE_CLAIMS_BY_TASK[task_id])
    local_status = {c["claim_id"]: c["classification"] for c in claim_results}
    local_evidence = {c["claim_id"]: c["evidence"] for c in claim_results}

    code, _ = extract_code(a3_text or "", task.get("function_name"))
    c8_status, c8_evidence = classify_c8_tests_pass(code, task["reference_tests"])
    local_status["C8_TESTS_PASS"] = c8_status
    local_evidence["C8_TESTS_PASS"] = c8_evidence

    depends_on = CODE_DEPENDS_ON  # empty -- see schema8.py docstring
    effective = propagate(local_status, depends_on)
    mrs, root_origins = minimal_repair_set(effective, depends_on, claim_order)
    labels = {cid: error_taxonomy_label(cid, effective, root_origins) for cid in claim_order}
    seam_cons = seam_consistency_code(task, a1_obj, a3_text)
    seam_deps = [s for s in (seam_dependency_status(cid, local_status.get(cid, "OMITTED"), effective, depends_on) for cid in claim_order) if s]
    seam_exec = seam_execution_status(local_status["C8_TESTS_PASS"])

    terminal_status = effective["C8_TESTS_PASS"][0]
    return {
        "family": "CODE", "parse_quality": quality, "sections": sections,
        "local_status": local_status, "local_evidence": local_evidence, "effective": {k: list(v) for k, v in effective.items()},
        "mrs": mrs, "root_origins": root_origins, "error_labels": labels,
        "seam_consistency": seam_cons, "seam_dependency": seam_deps, "seam_execution": seam_exec,
        "task_status": terminal_status, "n_claims": len(claim_order), "n_artifacts": len(CODE_ARTIFACTS),
        "claim_order": claim_order, "artifact_of": CODE_CLAIM_ARTIFACT_BY_TASK[task_id], "depends_on": depends_on,
    }


def analyze_p(task_id, text):
    if TASK_FAMILY[task_id] == "MATH":
        status, evidence = check_p_mode_math(text, MATH_TASKS[task_id]["final_answer"])
    else:
        status, evidence = check_p_mode_code(text, CODE_TASKS_BY_ID[task_id])
    return {"task_status": status, "evidence": evidence}


def main():
    all_records = []
    for model_id in MODELS:
        path = os.path.join(RUNS_DIR, f"pd_index.{model_id}.jsonl")
        rows = load_jsonl(path)
        for row in rows:
            if row["generation_failed"]:
                continue
            text = read_raw(row)
            rec = {"run_id": row["run_id"], "task_id": row["task_id"], "mode": row["mode"], "model_id": model_id,
                   "seed": row["seed"], "output_tokens": row["output_tokens"], "generation_time_sec": row["generation_time_sec"]}
            if row["mode"] == "P":
                rec["analysis"] = analyze_p(row["task_id"], text)
            else:
                if TASK_FAMILY[row["task_id"]] == "MATH":
                    rec["analysis"] = analyze_d_math(row["task_id"], text)
                else:
                    rec["analysis"] = analyze_d_code(row["task_id"], text)
            all_records.append(rec)

    out_path = os.path.join(os.path.dirname(__file__), "analysis8_records.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    print(f"n_records={len(all_records)}")
    by_mode = collections.Counter(r["mode"] for r in all_records)
    print("by mode:", dict(by_mode))
    return all_records


if __name__ == "__main__":
    main()
