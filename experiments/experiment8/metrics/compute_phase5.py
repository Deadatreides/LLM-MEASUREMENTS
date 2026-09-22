"""
Section 14: D_RETRY evaluation. For every real error case retried
(configs/runner8_local_retry.py), merges the newly-generated MRS-section
text back into the ORIGINAL generation's untouched upstream sections,
re-runs the same frozen classification pipeline on the merged artifact,
and checks whether the terminal claim became CORRECT. Compares tokens
spent (N=3 short local retries) against:
  - tokens_full_P: that (task, model)'s own average FLAT (P mode) token
    cost -- the literal "regenerate the whole answer" comparison section
    14 asks for.
  - tokens_full_D: that (task, model)'s own average DECOMPOSED (D mode)
    token cost -- regenerating the whole structured answer instead.
COMPUTE_SAVING = 1 - tokens_local / tokens_full, computed against both,
reported separately (P is the more literal reading of section 14).
"""
import collections
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "configs"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tasks"))

from storage8 import EXP_DIR, load_jsonl
from schema8 import MATH_TASKS, MATH_ARTIFACTS, MATH_CLAIMS, MATH_DEPENDS_ON, TASK_FAMILY
from tasks_pipeline_code import TASKS_BY_ID as CODE_TASKS_BY_ID
from classifiers8 import classify_math_claim
from claims_code_wrapper import classify_code_claims
from claims_def_code import CLAIMS_BY_TASK as CODE_CLAIMS_BY_TASK
from mech_check_code import classify_c8_tests_pass
from ast_checks import extract_code
from parse8 import parse_sections
from dependency_engine import propagate, minimal_repair_set, error_taxonomy_label

RUNS_DIR = os.path.join(EXP_DIR, "runs")
METRICS_DIR = os.path.dirname(__file__)
MODELS = ["qwen3-1.7b-q4_0-unsloth", "llama-3.2-1b-instruct-q4_0", "qwen2.5-coder-1.5b-instruct-q4_0"]
CODE_CLAIM_ORDER = {"CODE_01": ["C1_NAME", "C2_ARGC", "C3_ARG_TYPE", "C4_RETURN_TYPE", "C5_SEMANTICS_CASE", "C6_EDGE_EMPTY", "C7_EDGE_SPACES", "C8_TESTS_PASS"],
                     "CODE_06": ["C1_NAME", "C2_ARGC", "C3_RETURN_TYPE", "C4_SEMANTICS_INDICES", "C5_SEMANTICS_SMALLEST_I", "C6_EDGE_ONE_SOLUTION", "C7_EDGE_NOT_SAME", "C8_TESTS_PASS"]}


def read_raw(raw_path):
    with open(os.path.join(RUNS_DIR, raw_path, "result.txt"), encoding="utf-8") as f:
        return f.read()


def evaluate_merged_math(task_id, merged_sections):
    section_map = {"C1_METHOD": "A1_METHOD", "C2_VALUES": "A2_VALUES", "C3_COMPUTATION": "A3_COMPUTATION", "C4_FINAL_ANSWER": "A4_FINAL_ANSWER"}
    local_status = {}
    for cid in MATH_CLAIMS:
        sec_text = merged_sections.get(section_map[cid])
        local_status[cid] = classify_math_claim(task_id, cid, sec_text)[0] if sec_text else "OMITTED"
    effective = propagate(local_status, MATH_DEPENDS_ON)
    return effective["C4_FINAL_ANSWER"][0]


def evaluate_merged_code(task_id, merged_sections):
    task = CODE_TASKS_BY_ID[task_id]
    claim_order = CODE_CLAIM_ORDER[task_id]
    a1_text, a2_text, a3_text = merged_sections.get("A1_CONTRACT"), merged_sections.get("A2_EDGE_CASES"), merged_sections.get("A3_IMPLEMENTATION")
    status1, a1_obj, claim_results = classify_code_claims(a1_text, a2_text, task, CODE_CLAIMS_BY_TASK[task_id])
    local_status = {c["claim_id"]: c["classification"] for c in claim_results}
    code, _ = extract_code(a3_text or "", task.get("function_name"))
    c8_status, _ = classify_c8_tests_pass(code, task["reference_tests"])
    local_status["C8_TESTS_PASS"] = c8_status
    effective = propagate(local_status, {})
    return effective["C8_TESTS_PASS"][0]


def main():
    original_recs = {r["run_id"]: r for r in json.load(open(os.path.join(METRICS_DIR, "analysis8_records.json"), encoding="utf-8"))}
    p_by_unit_tokens = collections.defaultdict(list)
    d_by_unit_tokens = collections.defaultdict(list)
    for r in original_recs.values():
        key = (r["task_id"], r["model_id"])
        if r["mode"] == "P":
            p_by_unit_tokens[key].append(r["output_tokens"])
        else:
            d_by_unit_tokens[key].append(r["output_tokens"])

    results = []
    for model_id in MODELS:
        path = os.path.join(RUNS_DIR, f"local_retry8.{model_id}.jsonl")
        for row in load_jsonl(path):
            if row["generation_failed"]:
                continue
            orig = original_recs[row["orig_run_id"]]
            orig_sections = orig["analysis"]["sections"]
            mrs_artifact_ids = row["mrs_artifact_ids"]
            retry_text = read_raw(row["raw_path"])
            expected_artifacts = MATH_ARTIFACTS if TASK_FAMILY[row["task_id"]] == "MATH" else ["A1_CONTRACT", "A2_EDGE_CASES", "A3_IMPLEMENTATION"]
            if len(mrs_artifact_ids) == 1 and TASK_FAMILY[row["task_id"]] == "CODE":
                # CODE retry prompt asks for ONLY A3_IMPLEMENTATION, no header requested for it explicitly in the reply necessarily -- try header parse, fall back to raw text as the whole section
                parsed, quality, _ = parse_sections(retry_text, mrs_artifact_ids)
                new_sections = {mrs_artifact_ids[0]: parsed.get(mrs_artifact_ids[0]) or retry_text}
            else:
                parsed, quality, _ = parse_sections(retry_text, mrs_artifact_ids)
                new_sections = parsed

            merged = dict(orig_sections)
            for aid in mrs_artifact_ids:
                if new_sections.get(aid):
                    merged[aid] = new_sections[aid]

            if TASK_FAMILY[row["task_id"]] == "MATH":
                terminal_status = evaluate_merged_math(row["task_id"], merged)
            else:
                terminal_status = evaluate_merged_code(row["task_id"], merged)

            key = (row["task_id"], row["model_id"])
            results.append({
                "orig_run_id": row["orig_run_id"], "retry_run_id": row["run_id"], "task_id": row["task_id"], "model_id": row["model_id"],
                "mrs_artifact_ids": mrs_artifact_ids, "retry_tokens": row["output_tokens"], "retry_time_sec": row["generation_time_sec"],
                "retry_terminal_status": terminal_status, "retry_succeeded": terminal_status == "CORRECT",
                "tokens_full_P_avg": round(statistics.mean(p_by_unit_tokens[key]), 1) if p_by_unit_tokens[key] else None,
                "tokens_full_D_avg": round(statistics.mean(d_by_unit_tokens[key]), 1) if d_by_unit_tokens[key] else None,
            })

    # aggregate per orig_run_id (N=3 retries -> did at least one succeed? mean tokens per attempt)
    by_case = collections.defaultdict(list)
    for r in results:
        by_case[r["orig_run_id"]].append(r)

    case_summaries = []
    for orig_run_id, attempts in by_case.items():
        n = len(attempts)
        n_success = sum(1 for a in attempts if a["retry_succeeded"])
        mean_tokens = statistics.mean(a["retry_tokens"] for a in attempts)
        tokens_full_P = attempts[0]["tokens_full_P_avg"]
        tokens_full_D = attempts[0]["tokens_full_D_avg"]
        case_summaries.append({
            "orig_run_id": orig_run_id, "task_id": attempts[0]["task_id"], "model_id": attempts[0]["model_id"],
            "mrs_artifact_ids": attempts[0]["mrs_artifact_ids"], "n_attempts": n, "n_success": n_success,
            "any_success": n_success > 0, "success_rate": round(n_success / n, 4),
            "mean_retry_tokens_per_attempt": round(mean_tokens, 1),
            "tokens_full_P_avg": tokens_full_P, "tokens_full_D_avg": tokens_full_D,
            "compute_saving_vs_P": round(1 - mean_tokens / tokens_full_P, 4) if tokens_full_P else None,
            "compute_saving_vs_D": round(1 - mean_tokens / tokens_full_D, 4) if tokens_full_D else None,
        })

    overall = {
        "n_cases": len(case_summaries),
        "any_success_rate": round(sum(1 for c in case_summaries if c["any_success"]) / len(case_summaries), 4),
        "mean_compute_saving_vs_P": round(statistics.mean(c["compute_saving_vs_P"] for c in case_summaries), 4),
        "mean_compute_saving_vs_D": round(statistics.mean(c["compute_saving_vs_D"] for c in case_summaries), 4),
        "mean_retry_tokens_per_attempt": round(statistics.mean(c["mean_retry_tokens_per_attempt"] for c in case_summaries), 1),
    }

    out = {"overall": overall, "cases": case_summaries}
    with open(os.path.join(METRICS_DIR, "phase5_local_retry8.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(json.dumps(overall, ensure_ascii=False, indent=2))
    for c in case_summaries:
        print(c["orig_run_id"], "success_rate=", c["success_rate"], "saving_vs_P=", c["compute_saving_vs_P"])


if __name__ == "__main__":
    main()
