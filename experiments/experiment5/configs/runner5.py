"""
Experiment 5 main-data generator: independent monolithic solve attempts
for one model, one task, one temperature at a time -- exactly the
"не смешивать разные модели, разные промпты и разные температуры в одной
базовой выборке" requirement. Reuses experiment 2's verification
machinery (reference_tests, run_tests.py, error_signature) unmodified, so
error clustering here is directly comparable to experiments 1/2.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))

from llm_client import load_model, generate
from ast_checks import extract_code
from seams import run_test_suite, classify_test_run
from storage5 import save_raw, JsonlWriter, EXP_DIR
from tasks_pipeline import TASKS, TASKS_BY_ID

BASELINE_TEMPLATE = (
    "{question}\nProvide the complete function definition in a single Python code block."
)
MAX_TOKENS = 700
TOP_P = 1.0
TOP_K = 40
RUNS_DIR = os.path.join(EXP_DIR, "runs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--tasks", default="ALL")
    ap.add_argument("--temperatures", required=True, help="comma-separated, e.g. 0.0,0.5,1.0")
    ap.add_argument("--seeds", required=True, help="comma-separated seeds, e.g. 1,2,...,16")
    ap.add_argument("--logits", action="store_true", help="collect logits_all=True + logprobs (slow, small-N only)")
    ap.add_argument("--top_logprobs", type=int, default=10)
    args = ap.parse_args()

    tasks = TASKS if args.tasks == "ALL" else [TASKS_BY_ID[t] for t in args.tasks.split(",")]
    temperatures = [float(t) for t in args.temperatures.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    llm, model_entry, load_time = load_model(args.model_id, logits_all=args.logits)
    print(json.dumps({"event": "model_loaded", "model_id": args.model_id, "load_time_sec": round(load_time, 2), "logits": args.logits}))

    suffix = ".logits" if args.logits else ""
    index_path = os.path.join(RUNS_DIR, f"index.{args.model_id}{suffix}.jsonl")
    writer = JsonlWriter(index_path, key_field="run_id")

    n_done = 0
    for task in tasks:
        prompt_text = BASELINE_TEMPLATE.format(question=task["question"])
        for temperature in temperatures:
            for seed in seeds:
                run_id = f"{args.model_id}--{task['task_id']}--T{temperature}--seed{seed}"
                if writer.has(run_id):
                    continue
                gen_kwargs = dict(temperature=temperature, top_p=TOP_P, top_k=TOP_K, seed=seed, max_tokens=MAX_TOKENS)
                if args.logits:
                    gen_kwargs["logprobs"] = args.top_logprobs
                gen = generate(llm, args.model_id, model_entry, prompt_text, **gen_kwargs)

                code, fmt = extract_code(gen["raw_text"], task["function_name"]) if gen["raw_text"] else (None, None)
                if code is None:
                    verification = {"status": "MECHANICAL_FAIL", "error_class": "FORMAT_ERROR", "error_signature": "no_extractable_code"}
                    test_detail = None
                else:
                    rr = run_test_suite(code, task["reference_tests"])
                    v = classify_test_run(rr)
                    verification = {"status": v["status"], "error_class": v["error_class"], "error_signature": v["error_signature"]}
                    test_detail = rr

                meta = {
                    "run_id": run_id, "rendered_prompt": gen["rendered_prompt"],
                    "generation_error": gen["generation_error"], "test_detail": test_detail,
                    "logprobs": gen.get("logprobs"),
                }
                raw_path = save_raw(run_id, gen["raw_text"], meta)

                row = {
                    "run_id": run_id, "generation_id": run_id, "task_id": task["task_id"],
                    "claim_id": None, "model": args.model_id, "model_size": model_entry["size_label"],
                    "model_version": model_entry["model_version"], "quantization": model_entry["quantization"],
                    "temperature": temperature, "seed": seed, "prompt_id": "BASELINE",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "has_code": code is not None, "answer_format": fmt,
                    "correctness": verification["status"] == "MECHANICAL_PASS",
                    "verification_status": verification["status"],
                    "error_class": verification["error_class"], "error_signature": verification["error_signature"],
                    "input_tokens": gen["input_tokens"], "output_tokens": gen["output_tokens"],
                    "generation_time_sec": round(gen["generation_time_sec"], 4),
                    "generation_failed": gen["generation_failed"],
                    "raw_path": raw_path,
                    "has_logprobs": bool(gen.get("logprobs")),
                }
                writer.write(row, key=run_id)
                n_done += 1
        print(json.dumps({"event": "task_done", "model_id": args.model_id, "task_id": task["task_id"]}))

    writer.close()
    del llm
    print(json.dumps({"event": "model_done", "model_id": args.model_id, "n_generated_this_run": n_done}))


if __name__ == "__main__":
    main()
