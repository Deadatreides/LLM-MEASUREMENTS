"""Section 16/23: real D-mode generation for all 20 tasks, one model per
invocation, fixed T=0.5, seeds 1-4 (smaller N than experiments 7/8 --
this experiment is not re-testing model quality or P-vs-D, only whether
the dependency/affected-subgraph machinery works on real errors)."""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tasks"))

from llm_client import load_model, generate
from storage9 import save_raw, JsonlWriter, EXP_DIR
from schema9 import TASKS, TASK_IDS
from prompts9 import D_TEMPLATE, MAX_TOKENS_D

RUNS_DIR = os.path.join(EXP_DIR, "runs")
TEMPERATURE = 0.5
TOP_P = 1.0
TOP_K = 40


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--seeds", default="1,2,3,4")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    llm, model_entry, load_time = load_model(args.model_id)
    print(json.dumps({"event": "model_loaded", "model_id": args.model_id, "load_time_sec": round(load_time, 2)}))

    index_path = os.path.join(RUNS_DIR, f"d_index9.{args.model_id}.jsonl")
    writer = JsonlWriter(index_path, key_field="run_id")

    n_done = 0
    for task_id in TASK_IDS:
        prompt_text = D_TEMPLATE.format(question=TASKS[task_id]["question"])
        for seed in seeds:
            run_id = f"{args.model_id}--{task_id}--D--T{TEMPERATURE}--seed{seed}"
            if writer.has(run_id):
                continue
            gen = generate(llm, args.model_id, model_entry, prompt_text, temperature=TEMPERATURE,
                            top_p=TOP_P, top_k=TOP_K, seed=seed, max_tokens=MAX_TOKENS_D)
            raw_path = save_raw(run_id, gen["raw_text"], {"run_id": run_id, "rendered_prompt": gen["rendered_prompt"]})
            row = {
                "run_id": run_id, "task_id": task_id, "model_id": args.model_id, "temperature": TEMPERATURE, "seed": seed,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "input_tokens": gen["input_tokens"], "output_tokens": gen["output_tokens"],
                "generation_time_sec": round(gen["generation_time_sec"], 4),
                "generation_failed": gen["generation_failed"], "raw_path": raw_path,
            }
            writer.write(row, key=run_id)
            n_done += 1
        print(json.dumps({"event": "task_done", "model_id": args.model_id, "task_id": task_id}))

    writer.close()
    del llm
    print(json.dumps({"event": "model_done", "model_id": args.model_id, "n_generated_this_run": n_done}))


if __name__ == "__main__":
    main()
