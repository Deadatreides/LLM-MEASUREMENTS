"""
Generates the CHANGE_PROMPT continuation arm: same model, same task, same
temperature, but a different prompt framing (reasoning-first, matching
experiment 1/2's P3 style) -- reused verbatim for consistency with prior
experiments, not invented fresh for this one.

Everything else (SAME_MODEL continuation, CHANGE_TEMPERATURE, SWITCH_MODEL)
is constructed later by re-slicing experiment 5's existing N=16 pools --
no new generation needed for those arms.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))

from llm_client import load_model, generate
from ast_checks import extract_code
from seams import run_test_suite, classify_test_run
from storage6 import save_raw, JsonlWriter, EXP_DIR
from tasks_pipeline import TASKS, TASKS_BY_ID
from error_taxonomy import classify as classify_error

ALT_PROMPT_TEMPLATE = (
    "{question}\nFirst think through the algorithm and any edge cases in plain text. "
    "Then provide the complete, final function definition in a single Python code block."
)
MAX_TOKENS = 900  # reasoning-first responses run longer than the plain baseline
TOP_P = 1.0
TOP_K = 40
RUNS_DIR = os.path.join(EXP_DIR, "runs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--tasks", default="ALL")
    ap.add_argument("--temperatures", required=True)
    ap.add_argument("--seeds", required=True)
    args = ap.parse_args()

    tasks = TASKS if args.tasks == "ALL" else [TASKS_BY_ID[t] for t in args.tasks.split(",")]
    temperatures = [float(t) for t in args.temperatures.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    llm, model_entry, load_time = load_model(args.model_id)
    print(json.dumps({"event": "model_loaded", "model_id": args.model_id, "load_time_sec": round(load_time, 2)}))

    index_path = os.path.join(RUNS_DIR, f"index.{args.model_id}.changeprompt.jsonl")
    writer = JsonlWriter(index_path, key_field="run_id")

    n_done = 0
    for task in tasks:
        prompt_text = ALT_PROMPT_TEMPLATE.format(question=task["question"])
        for temperature in temperatures:
            for seed in seeds:
                run_id = f"{args.model_id}--{task['task_id']}--T{temperature}--seed{seed}--CHANGEPROMPT"
                if writer.has(run_id):
                    continue
                gen = generate(llm, args.model_id, model_entry, prompt_text, temperature=temperature,
                                top_p=TOP_P, top_k=TOP_K, seed=seed, max_tokens=MAX_TOKENS)

                code, fmt = extract_code(gen["raw_text"], task["function_name"]) if gen["raw_text"] else (None, None)
                if code is None:
                    verification = {"status": "MECHANICAL_FAIL", "error_class": "FORMAT_ERROR", "error_signature": "no_extractable_code"}
                else:
                    rr = run_test_suite(code, task["reference_tests"])
                    v = classify_test_run(rr)
                    verification = {"status": v["status"], "error_class": v["error_class"], "error_signature": v["error_signature"]}

                status, err_type = classify_error(verification, code, task["function_name"])

                raw_path = save_raw(run_id, gen["raw_text"], {
                    "run_id": run_id, "rendered_prompt": gen["rendered_prompt"], "generation_error": gen["generation_error"],
                })

                row = {
                    "run_id": run_id, "generation_id": run_id, "task_id": task["task_id"], "claim_id": None,
                    "model": args.model_id, "model_size": model_entry["size_label"], "model_version": model_entry["model_version"],
                    "quantization": model_entry["quantization"], "temperature": temperature, "seed": seed,
                    "prompt_id": "CHANGEPROMPT", "timestamp": datetime.now(timezone.utc).isoformat(),
                    "has_code": code is not None, "answer_format": fmt,
                    "correctness_status": status, "error_type": err_type,
                    "correctness": status == "CORRECT",
                    "verification_status": verification["status"], "error_class": verification["error_class"],
                    "error_signature": verification["error_signature"],
                    "input_tokens": gen["input_tokens"], "output_tokens": gen["output_tokens"],
                    "generation_time_sec": round(gen["generation_time_sec"], 4),
                    "generation_failed": gen["generation_failed"], "raw_path": raw_path,
                }
                writer.write(row, key=run_id)
                n_done += 1
        print(json.dumps({"event": "task_done", "model_id": args.model_id, "task_id": task["task_id"]}))

    writer.close()
    del llm
    print(json.dumps({"event": "model_done", "model_id": args.model_id, "n_generated_this_run": n_done}))


if __name__ == "__main__":
    main()
