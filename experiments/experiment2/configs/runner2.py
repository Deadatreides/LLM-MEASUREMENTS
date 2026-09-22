"""
Experiment 2 runner: baseline monolithic best-of-N + artifact pipeline
(A1 CONTRACT -> A2 TESTS / A3 IMPLEMENTATION -> seams -> optional A4 PATCH
repair), for one model, over a set of tasks. Loads the model once.

Fixed, disclosed methodological choice: TEMPERATURE = 0.3 (not 0.0) for
every call in this experiment (baseline samples, A1, A2, A3, A4, and the
3 A3 samples in pipeline-3). Rationale: at T=0 this backend is very close
to deterministic (see experiment 1's finding -- most cells identical
across seeds), which would starve best-of-N of any real candidate
diversity and bias the comparison in the pipeline's favor for free. A low
*non-zero* fixed temperature keeps temperature far from being "the main
mechanism" (per spec section 18) while giving both regimes a fair,
equal-diversity-budget starting point.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tasks"))

from llm_client import load_model, generate
from prompts import (
    BASELINE_TEMPLATE,
    A1_CONTRACT_TEMPLATE,
    A2_TESTS_TEMPLATE,
    A3_IMPLEMENTATION_TEMPLATE,
    A4_PATCH_TEMPLATE,
)
from contract import check_contract
from ast_checks import extract_code
from seams import run_test_suite, classify_test_run, seam_a1_a3
from ast_mutator import generate_mutants
from storage import save_raw, JsonlWriter, EXP_DIR
from tasks_pipeline import TASKS, TASKS_BY_ID

TEMPERATURE = 0.3
TOP_P = 1.0
TOP_K = 40

MAX_TOKENS = {"BASELINE": 700, "A1": 500, "A2": 600, "A3": 700, "A4": 700}

RUNS_DIR = os.path.join(EXP_DIR, "runs")


def failure_report_text(run_result, verdict):
    if verdict["error_class"] == "SYNTAX_ERROR":
        return f"SyntaxError while parsing the implementation: {run_result.get('impl_exec_error')}"
    if verdict["error_class"] == "RUNTIME_ERROR":
        return f"Error while loading the implementation: {run_result.get('impl_exec_error')}"
    if verdict["error_class"] == "TEST_FAILURE":
        lines = []
        for t in run_result.get("tests", []):
            if t["status"] != "PASS":
                lines.append(f"- {t['name']}: {t['status']} ({t['exception_type']}: {t['exception_message']})")
        return "The following tests failed:\n" + "\n".join(lines)
    return f"{verdict['error_class']}: {verdict['error_signature']}"


def mutation_score(test_code, function_name, reference_implementation):
    mutants = generate_mutants(reference_implementation)
    if not mutants:
        return {"n_mutants": 0, "n_killed": 0, "score": None}
    killed = 0
    for m in mutants:
        rr = run_test_suite(m["mutant_source"], test_code)
        v = classify_test_run(rr)
        if v["status"] != "MECHANICAL_PASS":
            killed += 1
    return {"n_mutants": len(mutants), "n_killed": killed, "score": round(killed / len(mutants), 3)}


class Ctx:
    """Bundles the per-model index writers so callers don't pass 5 args around."""

    def __init__(self, model_id):
        self.model_id = model_id
        self.artifacts = JsonlWriter(os.path.join(RUNS_DIR, f"artifacts.{model_id}.jsonl"), key_field="artifact_id")
        self.seams = JsonlWriter(os.path.join(RUNS_DIR, f"seams.{model_id}.jsonl"), key_field="seam_id")
        self.baseline = JsonlWriter(os.path.join(RUNS_DIR, f"baseline.{model_id}.jsonl"), key_field="run_id")
        self.pipeline_runs = JsonlWriter(os.path.join(RUNS_DIR, f"pipeline_runs.{model_id}.jsonl"), key_field="run_id")

    def close(self):
        for w in (self.artifacts, self.seams, self.baseline, self.pipeline_runs):
            w.close()


def call_role(ctx, llm, model_id, model_entry, role, artifact_type, prompt_text, seed, task_id,
              variant, attempt_idx, parent_artifact_ids, input_artifact_ids, seq):
    artifact_id = f"{model_id}--{task_id}--{variant}--{role}--a{attempt_idx}--{seq}"
    if ctx.artifacts.has(artifact_id):
        # resumability: reuse a previously generated artifact's RAW text
        raw_path = os.path.join(RUNS_DIR, "raw", artifact_id, "result.txt")
        with open(raw_path, encoding="utf-8") as f:
            raw_text = f.read()
        return artifact_id, raw_text, True

    gen = generate(llm, model_id, model_entry, prompt_text, temperature=TEMPERATURE, top_p=TOP_P,
                    top_k=TOP_K, seed=seed, max_tokens=MAX_TOKENS[artifact_type])
    raw_path = save_raw(artifact_id, gen["raw_text"], {
        "artifact_id": artifact_id, "role": role, "prompt_text": prompt_text,
        "rendered_prompt": gen["rendered_prompt"], "generation_error": gen["generation_error"],
    })
    row = {
        "artifact_id": artifact_id, "run_id": f"{model_id}--{task_id}--{variant}--a{attempt_idx}",
        "artifact_type": artifact_type, "task_id": task_id, "model_id": model_id, "role": role,
        "pipeline_variant": variant, "attempt_idx": attempt_idx,
        "temperature": TEMPERATURE, "top_p": TOP_P, "seed": seed,
        "parent_artifact_ids": parent_artifact_ids, "input_artifact_ids": input_artifact_ids,
        "output_artifact_id": artifact_id,
        "input_tokens": gen["input_tokens"], "output_tokens": gen["output_tokens"],
        "generation_time_sec": round(gen["generation_time_sec"], 4),
        "generation_failed": gen["generation_failed"], "generation_error": gen["generation_error"],
        "raw_path": raw_path,
    }
    ctx.artifacts.write(row, key=artifact_id)
    return artifact_id, gen["raw_text"], False


def write_seam(ctx, seam_type, run_id, task_id, model_id, involved_artifact_ids, status, error_class, error_signature, details=None):
    seam_id = f"{run_id}--SEAM-{seam_type}"
    if ctx.seams.has(seam_id):
        return None
    row = {
        "seam_id": seam_id, "seam_type": seam_type, "run_id": run_id, "task_id": task_id, "model_id": model_id,
        "involved_artifact_ids": involved_artifact_ids, "status": status,
        "error_class": error_class, "error_signature": error_signature, "details": details or {},
    }
    ctx.seams.write(row, key=seam_id)
    return row


# ---------------------------------------------------------------------------
# BASELINE: monolithic best-of-N
# ---------------------------------------------------------------------------

def run_baseline(ctx, llm, model_id, model_entry, task, n_samples=8):
    task_id = task["task_id"]
    prompt_text = BASELINE_TEMPLATE.format(question=task["question"])
    for i in range(1, n_samples + 1):
        run_id = f"{model_id}--{task_id}--BASELINE--sample{i:02d}"
        if ctx.baseline.has(run_id):
            continue
        artifact_id, raw_text, _ = call_role(
            ctx, llm, model_id, model_entry, "BASELINE", "BASELINE", prompt_text, seed=i,
            task_id=task_id, variant="BASELINE", attempt_idx=i, parent_artifact_ids=[],
            input_artifact_ids=[], seq="sol",
        )
        code, fmt = extract_code(raw_text, task["function_name"])
        if code is None:
            verdict = {"status": "MECHANICAL_FAIL", "error_class": "FORMAT_ERROR", "error_signature": "no_extractable_code"}
        else:
            rr = run_test_suite(code, task["reference_tests"])
            verdict = classify_test_run(rr)
        row = {
            "run_id": run_id, "task_id": task_id, "model_id": model_id, "sample_idx": i,
            "artifact_id": artifact_id, "has_code": code is not None, "answer_format": fmt,
            "reference_status": verdict["status"], "error_class": verdict["error_class"],
            "error_signature": verdict["error_signature"],
        }
        ctx.baseline.write(row, key=run_id)


# ---------------------------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------------------------

def gen_contract(ctx, llm, model_id, model_entry, task, variant, attempt_idx, seed):
    prompt_text = A1_CONTRACT_TEMPLATE.format(question=task["question"])
    aid, raw, _ = call_role(ctx, llm, model_id, model_entry, "A1_CONTRACT", "A1", prompt_text, seed,
                             task["task_id"], variant, attempt_idx, [], [], "a1")
    cr = check_contract(raw, task["function_name"])
    run_id = f"{model_id}--{task['task_id']}--{variant}--a{attempt_idx}"
    write_seam(ctx, "A1_SELF", run_id, task["task_id"], model_id, [aid], cr["seam_a1_self_status"],
               cr["error_class"], cr["error_signature"], details=cr)
    return aid, raw, cr


def gen_tests(ctx, llm, model_id, model_entry, task, variant, attempt_idx, seed, a1_id, contract_json_text):
    prompt_text = A2_TESTS_TEMPLATE.format(
        question=task["question"], contract_json=contract_json_text, function_name=task["function_name"]
    )
    aid, raw, _ = call_role(ctx, llm, model_id, model_entry, "A2_TESTS", "A2", prompt_text, seed,
                             task["task_id"], variant, attempt_idx, [a1_id], [a1_id], "a2")
    test_code, _ = extract_code(raw, None)
    return aid, raw, (test_code or "")


def gen_implementation(ctx, llm, model_id, model_entry, task, variant, attempt_idx, seed, a1_id, contract_json_text, seq="a3"):
    prompt_text = A3_IMPLEMENTATION_TEMPLATE.format(question=task["question"], contract_json=contract_json_text)
    aid, raw, _ = call_role(ctx, llm, model_id, model_entry, "A3_IMPLEMENTATION", "A3", prompt_text, seed,
                             task["task_id"], variant, attempt_idx, [a1_id], [a1_id], seq)
    code, _ = extract_code(raw, task["function_name"])
    return aid, raw, (code or "")


def gen_patch(ctx, llm, model_id, model_entry, task, variant, attempt_idx, seed, a1_id, a3_id, contract_json_text,
              previous_code, failure_report, seq):
    prompt_text = A4_PATCH_TEMPLATE.format(
        function_name=task["function_name"], question=task["question"], contract_json=contract_json_text,
        previous_code=previous_code or "(no code was extracted from the previous attempt)",
        failure_report=failure_report,
    )
    aid, raw, _ = call_role(ctx, llm, model_id, model_entry, "A4_PATCH", "A4", prompt_text, seed,
                             task["task_id"], variant, attempt_idx, [a1_id, a3_id], [a1_id, a3_id], seq)
    code, _ = extract_code(raw, task["function_name"])
    return aid, raw, (code or "")


def evaluate_implementation(ctx, run_id, task_id, model_id, a1_id, a2_id, a3_id, cr, code, test_code, task, label):
    """Runs every mechanical check for one candidate implementation and
    records the seams. label distinguishes INITIAL vs REPAIR1 vs REPAIR2."""
    out = {}

    seam13 = seam_a1_a3(cr, code or "")
    write_seam(ctx, f"A1_A3_{label}", run_id, task_id, model_id, [a1_id, a3_id],
               seam13["status"], seam13["error_class"], seam13["error_signature"], details=seam13["inspect"])
    out["seam_a1_a3"] = seam13["status"]

    rr_a2 = run_test_suite(code or "", test_code or "")
    v_a2 = classify_test_run(rr_a2)
    write_seam(ctx, f"A2_A3_{label}", run_id, task_id, model_id, [a2_id, a3_id],
               v_a2["status"], v_a2["error_class"], v_a2["error_signature"], details=rr_a2)
    out["seam_a2_a3"] = v_a2
    out["seam_a2_a3_raw"] = rr_a2

    rr_ref = run_test_suite(code or "", task["reference_tests"])
    v_ref = classify_test_run(rr_ref)
    write_seam(ctx, f"REFERENCE_A3_{label}", run_id, task_id, model_id, [a3_id],
               v_ref["status"], v_ref["error_class"], v_ref["error_signature"], details=rr_ref)
    out["seam_reference"] = v_ref

    return out


def run_pipeline_attempt(ctx, llm, model_id, model_entry, task, variant, attempt_idx, max_repairs):
    task_id = task["task_id"]
    run_id = f"{model_id}--{task_id}--{variant}--a{attempt_idx}"
    if ctx.pipeline_runs.has(run_id):
        return

    t_wall0 = time.time()
    seed = attempt_idx

    a1_id, a1_raw, cr = gen_contract(ctx, llm, model_id, model_entry, task, variant, attempt_idx, seed)
    # A2/A3/A4 see A1's raw output verbatim as "the contract" -- simplest and
    # most faithful (avoids silently repairing a malformed contract by
    # re-serializing a reconstructed object before passing it on).
    contract_text_for_roles = a1_raw

    a2_id, a2_raw, test_code = gen_tests(ctx, llm, model_id, model_entry, task, variant, attempt_idx, seed, a1_id, contract_text_for_roles)

    # A2 quality check: does A2's own test suite even accept the known-correct
    # reference implementation? (independent of A3 entirely)
    rr_a2_ref = run_test_suite(task["reference_implementation"], test_code)
    v_a2_ref = classify_test_run(rr_a2_ref)
    write_seam(ctx, "A2_REFERENCE", run_id, task_id, model_id, [a2_id], v_a2_ref["status"],
               v_a2_ref["error_class"], v_a2_ref["error_signature"], details=rr_a2_ref)

    mscore = mutation_score(test_code, task["function_name"], task["reference_implementation"])
    write_seam(ctx, "A2_MUTATION", run_id, task_id, model_id, [a2_id], "MECHANICAL_PASS" if mscore["score"] is not None else "TECHNICAL_FAILURE",
               None, None, details=mscore)

    if variant == "PIPELINE_3":
        a3_specs = [("a3_1", seed), ("a3_2", seed + 100), ("a3_3", seed + 200)]
    else:
        a3_specs = [("a3", seed)]

    candidates = []
    for seq, a3_seed in a3_specs:
        a3_id, a3_raw, code = gen_implementation(ctx, llm, model_id, model_entry, task, variant, attempt_idx, a3_seed, a1_id, contract_text_for_roles, seq=seq)
        ev = evaluate_implementation(ctx, run_id, task_id, model_id, a1_id, a2_id, a3_id, cr, code, test_code, task, label=f"INITIAL_{seq}")
        candidates.append({"seq": seq, "a3_id": a3_id, "code": code, "ev": ev})

    # selection among candidates (mechanical only: prefer one whose A2 seam
    # passes; break ties by taking the first such, else just take the first)
    passing = [c for c in candidates if c["ev"]["seam_a2_a3"]["status"] == "MECHANICAL_PASS"]
    chosen = passing[0] if passing else candidates[0]

    initial_reference_status = chosen["ev"]["seam_reference"]["status"]
    initial_seam_a2_a3_status = chosen["ev"]["seam_a2_a3"]["status"]

    n_repairs_used = 0
    final_code = chosen["code"]
    final_reference_status = initial_reference_status
    final_seam_a2_a3_status = initial_seam_a2_a3_status
    repair_regressed = False

    current_a3_id = chosen["a3_id"]
    current_code = chosen["code"]
    if variant in ("PIPELINE_2", "PIPELINE_2R2") and initial_seam_a2_a3_status != "MECHANICAL_PASS":
        max_r = 2 if variant == "PIPELINE_2R2" else 1
        max_r = min(max_r, max_repairs)
        for r in range(1, max_r + 1):
            failure_report = failure_report_text(chosen["ev"]["seam_a2_a3_raw"], chosen["ev"]["seam_a2_a3"])
            a4_id, a4_raw, patched_code = gen_patch(
                ctx, llm, model_id, model_entry, task, variant, attempt_idx, seed + 300 + r,
                a1_id, current_a3_id, contract_text_for_roles, current_code, failure_report, seq=f"a4_{r}",
            )
            ev = evaluate_implementation(ctx, run_id, task_id, model_id, a1_id, a2_id, a4_id, cr, patched_code, test_code, task, label=f"REPAIR{r}")
            n_repairs_used += 1
            pre_repair_ref_status = final_reference_status
            final_reference_status = ev["seam_reference"]["status"]
            final_seam_a2_a3_status = ev["seam_a2_a3"]["status"]
            if pre_repair_ref_status == "MECHANICAL_PASS" and final_reference_status != "MECHANICAL_PASS":
                repair_regressed = True
            current_a3_id = a4_id
            current_code = patched_code
            final_code = patched_code
            if final_seam_a2_a3_status == "MECHANICAL_PASS":
                break

    wall_time = time.time() - t_wall0

    art_rows = [r for r in json_load_all(os.path.join(RUNS_DIR, f"artifacts.{model_id}.jsonl")) if r["run_id"] == run_id]
    total_in = sum(a["input_tokens"] or 0 for a in art_rows)
    total_out = sum(a["output_tokens"] or 0 for a in art_rows)
    total_gen_time = sum(a["generation_time_sec"] or 0 for a in art_rows)
    n_calls = len(art_rows)

    row = {
        "run_id": run_id, "task_id": task_id, "model_id": model_id, "pipeline_variant": variant,
        "attempt_idx": attempt_idx, "n_llm_calls": n_calls, "total_input_tokens": total_in,
        "total_output_tokens": total_out, "total_generation_time_sec": round(total_gen_time, 3),
        "wall_time_sec": round(wall_time, 3),
        "a1_artifact_id": a1_id, "a2_artifact_id": a2_id,
        "a3_artifact_ids": [c["a3_id"] for c in candidates], "chosen_a3_artifact_id": chosen["a3_id"],
        "n_a3_candidates": len(candidates),
        "seam_a1_self_status": cr["seam_a1_self_status"],
        "seam_a2_reference_status": v_a2_ref["status"],
        "mutation_score_a2": mscore["score"], "mutation_n": mscore["n_mutants"],
        "initial_seam_a2_a3_status": initial_seam_a2_a3_status,
        "initial_reference_status": initial_reference_status,
        "n_repairs_used": n_repairs_used,
        "final_seam_a2_a3_status": final_seam_a2_a3_status,
        "final_reference_status": final_reference_status,
        "repair_regressed": repair_regressed,
        "initial_solved": initial_reference_status == "MECHANICAL_PASS",
        "final_solved": final_reference_status == "MECHANICAL_PASS",
    }
    ctx.pipeline_runs.write(row, key=run_id)


def json_load_all(path):
    from storage import load_jsonl
    return load_jsonl(path)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--tasks", default="ALL")
    ap.add_argument("--modes", default="BASELINE,PIPELINE_1,PIPELINE_2,PIPELINE_2R2,PIPELINE_3")
    ap.add_argument("--baseline-n", type=int, default=8)
    ap.add_argument("--attempts", type=int, default=3, help="independent pipeline attempts per task (per variant)")
    args = ap.parse_args()

    tasks = TASKS if args.tasks == "ALL" else [TASKS_BY_ID[t] for t in args.tasks.split(",")]
    modes = args.modes.split(",")

    llm, model_entry, load_time = load_model(args.model_id)
    ctx = Ctx(args.model_id)
    print(json.dumps({"event": "model_loaded", "model_id": args.model_id, "load_time_sec": round(load_time, 2)}))

    for task in tasks:
        if "BASELINE" in modes:
            run_baseline(ctx, llm, args.model_id, model_entry, task, n_samples=args.baseline_n)
        for variant in ("PIPELINE_1", "PIPELINE_2", "PIPELINE_2R2", "PIPELINE_3"):
            if variant not in modes:
                continue
            n_attempts = 1 if variant == "PIPELINE_3" else args.attempts
            for attempt_idx in range(1, n_attempts + 1):
                run_pipeline_attempt(ctx, llm, args.model_id, model_entry, task, variant, attempt_idx, max_repairs=2)
        print(json.dumps({"event": "task_done", "model_id": args.model_id, "task_id": task["task_id"]}))

    ctx.close()
    del llm
    print(json.dumps({"event": "model_done", "model_id": args.model_id}))


if __name__ == "__main__":
    main()
