"""
Section 8/9/13: LOCAL_RETRY action. For a unit whose target claim was
selected by metrics/target_claim.py (from the first-8 pool only, computed
BEFORE this script ever runs), generate N=8 short, narrowly-scoped
continuations that address ONLY that one claim, keeping the rest of the
original first-pass answer untouched (spec section 8: "the rest of the
original solution is preserved" -- here that means the other 6 claims'
classifications, from the existing first-8 pool, are carried forward
unchanged into the outcome table; only the target claim's classification
can change).

One invocation loads one model and can process several (task_id, claim_id)
target pairs for that model in one session (avoids repeated model loads).
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tasks"))

from llm_client import load_model, generate
from storage7 import save_raw, JsonlWriter, EXP_DIR
from tasks_pipeline import TASKS_BY_ID
from claims_def import CLAIMS_BY_TASK
from local_classifiers import classify_local_answer

RUNS_DIR = os.path.join(EXP_DIR, "runs")
MAX_TOKENS = 200
TOP_P = 1.0
TOP_K = 40
TEMPERATURE = 0.5

LOCAL_RETRY_TEMPLATE = (
    "You previously produced a specification for this task:\n\n{question}\n\n"
    "Focus ONLY on this specific aspect of the specification: \"{claim_text}\"\n\n"
    "State clearly, in one or two sentences, whether this is true or false for the "
    "function's required behavior, and briefly why. Do not restate the rest of the "
    "specification, do not write code."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--targets", required=True, help="comma-separated task_id:claim_id pairs, e.g. CODE_01:C7_EDGE_SPACES,CODE_02:C6_EDGE_EMPTY")
    ap.add_argument("--seeds", default="1,2,3,4,5,6,7,8")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    targets = [tuple(t.split(":")) for t in args.targets.split(",")]

    llm, model_entry, load_time = load_model(args.model_id)
    print(json.dumps({"event": "model_loaded", "model_id": args.model_id, "load_time_sec": round(load_time, 2)}))

    index_path = os.path.join(RUNS_DIR, f"local_retry.{args.model_id}.jsonl")
    writer = JsonlWriter(index_path, key_field="run_id")

    n_done = 0
    for task_id, claim_id in targets:
        task = TASKS_BY_ID[task_id]
        claim_def = next(c for c in CLAIMS_BY_TASK[task_id] if c["claim_id"] == claim_id)
        prompt_text = LOCAL_RETRY_TEMPLATE.format(question=task["question"], claim_text=claim_def["claim_text"])

        for seed in seeds:
            run_id = f"{args.model_id}--{task_id}--{claim_id}--T{TEMPERATURE}--seed{seed}--LOCALRETRY"
            if writer.has(run_id):
                continue
            gen = generate(llm, args.model_id, model_entry, prompt_text, temperature=TEMPERATURE,
                            top_p=TOP_P, top_k=TOP_K, seed=seed, max_tokens=MAX_TOKENS)

            text = gen["raw_text"] or ""
            cls, evidence = classify_local_answer(task_id, claim_id, text)

            raw_path = save_raw(run_id, gen["raw_text"], {"run_id": run_id, "rendered_prompt": gen["rendered_prompt"]})
            row = {
                "run_id": run_id, "task_id": task_id, "claim_id": claim_id, "model_id": args.model_id,
                "temperature": TEMPERATURE, "seed": seed, "prompt_id": "LOCAL_RETRY",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "classification": cls, "evidence": evidence,
                "input_tokens": gen["input_tokens"], "output_tokens": gen["output_tokens"],
                "generation_time_sec": round(gen["generation_time_sec"], 4),
                "generation_failed": gen["generation_failed"], "raw_path": raw_path,
            }
            writer.write(row, key=run_id)
            n_done += 1
        print(json.dumps({"event": "target_done", "model_id": args.model_id, "task_id": task_id, "claim_id": claim_id}))

    writer.close()
    del llm
    print(json.dumps({"event": "model_done", "model_id": args.model_id, "n_generated_this_run": n_done}))


if __name__ == "__main__":
    main()
