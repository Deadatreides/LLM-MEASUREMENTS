"""
Section 14: D_RETRY generation for every real D-mode INCORRECT case found
by build_analysis8.py whose MRS is a proper subset of all claims (spec:
"сравнивать только случаи, где локальный retry действительно допустим по
dependency graph" -- true for all 30 real error cases found in this
run, since this task set's dependency chains never made the WHOLE answer
the MRS). Regenerates ONLY the MRS-affected artifacts, N=3 attempts per
case, showing the untouched upstream artifacts (taken verbatim from the
ORIGINAL failed generation) as fixed context.
"""
import argparse
import collections
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tasks"))

from llm_client import load_model, generate
from storage8 import save_raw, JsonlWriter, EXP_DIR
from schema8 import MATH_TASKS, MATH_ARTIFACTS, TASK_FAMILY
from tasks_pipeline_code import TASKS_BY_ID as CODE_TASKS_BY_ID
from prompts8_retry import (MATH_RETRY_TEMPLATE, MATH_SECTION_PROMPTS, MATH_CONTEXT_LABELS,
                             CODE_RETRY_TEMPLATE, CODE_CONTEXT_LABELS, MATH_MAX_TOKENS_RETRY, CODE_MAX_TOKENS_RETRY)

RUNS_DIR = os.path.join(EXP_DIR, "runs")
TEMPERATURE = 0.5
TOP_P = 1.0
TOP_K = 40
RETRY_SEEDS = [101, 102, 103]  # disjoint from the original 1-8 seed range


def mrs_artifacts_math(mrs_claims, artifact_of):
    ids = sorted({artifact_of[c] for c in mrs_claims}, key=lambda a: MATH_ARTIFACTS.index(a))
    return ids


def build_math_retry_prompt(task_id, sections, mrs_artifact_ids):
    task = MATH_TASKS[task_id]
    fixed = [a for a in MATH_ARTIFACTS if a not in mrs_artifact_ids]
    fixed_context = "\n".join(f"{MATH_CONTEXT_LABELS[a]}: {sections[a]}" for a in fixed if sections.get(a))
    remaining = "".join(MATH_SECTION_PROMPTS[a] for a in mrs_artifact_ids)
    prompt = MATH_RETRY_TEMPLATE.format(question=task["question"], fixed_context=fixed_context, remaining_sections=remaining)
    return prompt, MATH_MAX_TOKENS_RETRY


def build_code_retry_prompt(task_id, sections):
    task = CODE_TASKS_BY_ID[task_id]
    fixed_context = "\n".join(f"{CODE_CONTEXT_LABELS[a]}: {sections[a]}" for a in ("A1_CONTRACT", "A2_EDGE_CASES") if sections.get(a))
    prompt = CODE_RETRY_TEMPLATE.format(question=task["question"], fixed_context=fixed_context)
    return prompt, CODE_MAX_TOKENS_RETRY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--cases_json", required=True, help="path to retry_cases.json (filtered per model)")
    args = ap.parse_args()

    with open(args.cases_json, encoding="utf-8") as f:
        cases = json.load(f)
    cases = [c for c in cases if c["model_id"] == args.model_id]

    llm, model_entry, load_time = load_model(args.model_id)
    print(json.dumps({"event": "model_loaded", "model_id": args.model_id, "load_time_sec": round(load_time, 2), "n_cases": len(cases)}))

    index_path = os.path.join(RUNS_DIR, f"local_retry8.{args.model_id}.jsonl")
    writer = JsonlWriter(index_path, key_field="run_id")

    n_done = 0
    for case in cases:
        task_id = case["task_id"]
        orig_run_id = case["orig_run_id"]
        sections = case["sections"]
        mrs = case["mrs"]
        if TASK_FAMILY[task_id] == "MATH":
            artifact_of = case["artifact_of"]
            mrs_artifact_ids = mrs_artifacts_math(mrs, artifact_of)
            prompt, max_tokens = build_math_retry_prompt(task_id, sections, mrs_artifact_ids)
        else:
            mrs_artifact_ids = ["A3_IMPLEMENTATION"]
            prompt, max_tokens = build_code_retry_prompt(task_id, sections)

        for seed in RETRY_SEEDS:
            run_id = f"{orig_run_id}--RETRY-seed{seed}"
            if writer.has(run_id):
                continue
            gen = generate(llm, args.model_id, model_entry, prompt, temperature=TEMPERATURE,
                            top_p=TOP_P, top_k=TOP_K, seed=seed, max_tokens=max_tokens)
            raw_path = save_raw(run_id, gen["raw_text"], {"run_id": run_id, "rendered_prompt": gen["rendered_prompt"]})
            row = {
                "run_id": run_id, "orig_run_id": orig_run_id, "task_id": task_id, "model_id": args.model_id,
                "mrs_artifact_ids": mrs_artifact_ids, "seed": seed, "temperature": TEMPERATURE,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "input_tokens": gen["input_tokens"], "output_tokens": gen["output_tokens"],
                "generation_time_sec": round(gen["generation_time_sec"], 4),
                "generation_failed": gen["generation_failed"], "raw_path": raw_path,
            }
            writer.write(row, key=run_id)
            n_done += 1
        print(json.dumps({"event": "case_done", "orig_run_id": orig_run_id}))

    writer.close()
    del llm
    print(json.dumps({"event": "model_done", "model_id": args.model_id, "n_generated_this_run": n_done}))


if __name__ == "__main__":
    main()
