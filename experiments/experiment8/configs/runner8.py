"""
Generates P (flat) and D (decomposed) mode answers for one model, across
all 6 tasks (MATH_01..04, CODE_01, CODE_06), fixed temperature=0.5, fixed
seeds. One invocation loads the model once. Section 23 budget: T=0.5,
seeds 1-8, MAX_TOKENS per prompts8.py, matching seeds used for both P and
D (same random draws requested from the model, not literal output
identity -- prompts differ by design).
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tasks"))

from llm_client import load_model, generate
from storage8 import save_raw, JsonlWriter, EXP_DIR
from schema8 import MATH_TASKS
from tasks_pipeline_code import TASKS_BY_ID as CODE_TASKS_BY_ID
from prompts8 import (MATH_P_TEMPLATE, MATH_D_TEMPLATE, CODE_P_TEMPLATE, CODE_D_TEMPLATE,
                       MATH_MAX_TOKENS_P, MATH_MAX_TOKENS_D, CODE_MAX_TOKENS_P, CODE_MAX_TOKENS_D)

RUNS_DIR = os.path.join(EXP_DIR, "runs")
TEMPERATURE = 0.5
TOP_P = 1.0
TOP_K = 40

TASK_IDS = ["MATH_01", "MATH_02", "MATH_03", "MATH_04", "CODE_01", "CODE_06"]


def build_prompt(task_id, mode):
    if task_id.startswith("MATH"):
        q = MATH_TASKS[task_id]["question"]
        template = MATH_P_TEMPLATE if mode == "P" else MATH_D_TEMPLATE
        max_tokens = MATH_MAX_TOKENS_P if mode == "P" else MATH_MAX_TOKENS_D
    else:
        q = CODE_TASKS_BY_ID[task_id]["question"]
        template = CODE_P_TEMPLATE if mode == "P" else CODE_D_TEMPLATE
        max_tokens = CODE_MAX_TOKENS_P if mode == "P" else CODE_MAX_TOKENS_D
    return template.format(question=q), max_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--tasks", default="ALL")
    ap.add_argument("--modes", default="P,D")
    ap.add_argument("--seeds", default="1,2,3,4,5,6,7,8")
    args = ap.parse_args()

    task_ids = TASK_IDS if args.tasks == "ALL" else args.tasks.split(",")
    modes = args.modes.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    llm, model_entry, load_time = load_model(args.model_id)
    print(json.dumps({"event": "model_loaded", "model_id": args.model_id, "load_time_sec": round(load_time, 2)}))

    index_path = os.path.join(RUNS_DIR, f"pd_index.{args.model_id}.jsonl")
    writer = JsonlWriter(index_path, key_field="run_id")

    n_done = 0
    for task_id in task_ids:
        for mode in modes:
            prompt_text, max_tokens = build_prompt(task_id, mode)
            for seed in seeds:
                run_id = f"{args.model_id}--{task_id}--{mode}--T{TEMPERATURE}--seed{seed}"
                if writer.has(run_id):
                    continue
                gen = generate(llm, args.model_id, model_entry, prompt_text, temperature=TEMPERATURE,
                                top_p=TOP_P, top_k=TOP_K, seed=seed, max_tokens=max_tokens)
                raw_path = save_raw(run_id, gen["raw_text"], {"run_id": run_id, "rendered_prompt": gen["rendered_prompt"]})
                row = {
                    "run_id": run_id, "task_id": task_id, "mode": mode, "model_id": args.model_id,
                    "temperature": TEMPERATURE, "seed": seed,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "input_tokens": gen["input_tokens"], "output_tokens": gen["output_tokens"],
                    "generation_time_sec": round(gen["generation_time_sec"], 4),
                    "generation_failed": gen["generation_failed"], "raw_path": raw_path,
                }
                writer.write(row, key=run_id)
                n_done += 1
            print(json.dumps({"event": "task_mode_done", "model_id": args.model_id, "task_id": task_id, "mode": mode}))

    writer.close()
    del llm
    print(json.dumps({"event": "model_done", "model_id": args.model_id, "n_generated_this_run": n_done}))


if __name__ == "__main__":
    main()
