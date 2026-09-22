"""
Experiment 3 generator: independent A1/CONTRACT samples only. No pipeline,
no repair, no comparison during generation -- each call is fully
independent (spec section 4: no previous A1 is ever shown to a new
generation, no iterative refinement, no reference at generation time).

Loads one model, then generates the cross-product of
(temperature x prompt x seed) for each task passed in, appending to that
model's own index shard. Different invocations (with different cell
specs) build up the combined design described in REPORT3.md /
configs/design.py.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tasks"))

from llm_client import load_model, generate
from prompts3 import PROMPTS
from storage7 import save_raw, JsonlWriter, EXP_DIR
from tasks_pipeline import TASKS, TASKS_BY_ID

MAX_TOKENS = 500
TOP_P = 1.0
TOP_K = 40
RUNS_DIR = os.path.join(EXP_DIR, "runs")


def text_hash(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--tasks", default="ALL")
    ap.add_argument("--temperatures", required=True, help="comma-separated, e.g. 0.0,0.2,0.5")
    ap.add_argument("--prompts", required=True, help="comma-separated PROMPT_IDs, e.g. P1,P2,P3")
    ap.add_argument("--seeds", required=True, help="comma-separated seeds, e.g. 1,2,3,4,5,6")
    args = ap.parse_args()

    tasks = TASKS if args.tasks == "ALL" else [TASKS_BY_ID[t] for t in args.tasks.split(",")]
    temperatures = [float(t) for t in args.temperatures.split(",")]
    prompt_ids = args.prompts.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    llm, model_entry, load_time = load_model(args.model_id)
    print(json.dumps({"event": "model_loaded", "model_id": args.model_id, "load_time_sec": round(load_time, 2)}))

    index_path = os.path.join(RUNS_DIR, f"a1_index.{args.model_id}.jsonl")
    writer = JsonlWriter(index_path, key_field="run_id")

    n_done = 0

    for task in tasks:
        for prompt_id in prompt_ids:
            template = PROMPTS[prompt_id]
            question_text = template.format(question=task["question"])
            for temperature in temperatures:
                for seed in seeds:
                    run_id = f"{args.model_id}--{task['task_id']}--{prompt_id}--T{temperature}--seed{seed}"
                    if writer.has(run_id):
                        continue
                    gen = generate(
                        llm, args.model_id, model_entry, question_text,
                        temperature=temperature, top_p=TOP_P, top_k=TOP_K, seed=seed,
                        max_tokens=MAX_TOKENS,
                    )
                    raw_path = save_raw(run_id, gen["raw_text"], {
                        "run_id": run_id, "question_text": question_text,
                        "rendered_prompt": gen["rendered_prompt"], "generation_error": gen["generation_error"],
                    })
                    row = {
                        "run_id": run_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "task_id": task["task_id"], "model_id": args.model_id,
                        "model_family": model_entry["family"], "model_version": model_entry["model_version"],
                        "quantization": model_entry["quantization"],
                        "prompt_id": prompt_id, "prompt_text_hash": text_hash(question_text),
                        "temperature": temperature, "top_p": TOP_P, "top_k": TOP_K, "seed": seed,
                        "input_tokens": gen["input_tokens"], "output_tokens": gen["output_tokens"],
                        "generation_time_sec": round(gen["generation_time_sec"], 4),
                        "generation_failed": gen["generation_failed"], "generation_error": gen["generation_error"],
                        "raw_path": raw_path,
                        "result_hash": text_hash(gen["raw_text"]) if gen["raw_text"] else None,
                    }
                    writer.write(row, key=run_id)
                    n_done += 1
        print(json.dumps({"event": "task_done", "model_id": args.model_id, "task_id": task["task_id"]}))

    writer.close()
    del llm
    print(json.dumps({"event": "model_done", "model_id": args.model_id, "n_generated_this_run": n_done}))


if __name__ == "__main__":
    main()
