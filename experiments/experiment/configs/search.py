"""
Simple filter/search tool over the compact index. Prints matching RUN_IDs
(one per line) by default, or full JSON rows with --full. Use the RUN_IDs to
open the corresponding RAW folder at runs/raw/<run_id>/ when a specific case
needs inspection.

Examples:
  python search.py --model qwen3-1.7b-q4_0-unsloth --task MATH_01
  python search.py --error_class WRONG_NUMERIC_ANSWER --temperature 1.0
  python search.py --correct false --prompt P3
  python search.py --error_signature "wrong_numeric:55"
"""
import argparse
import json
import os

EXP_DIR = r"<PROJECT_ROOT>\trace-probe\experiment"

PASS_STATUSES = {"REFERENCE_PASS", "MECHANICAL_PASS"}
FAIL_STATUSES = {"REFERENCE_FAIL", "MECHANICAL_FAIL"}


def load_rows():
    path = os.path.join(EXP_DIR, "runs", "index.jsonl")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--task")
    ap.add_argument("--prompt")
    ap.add_argument("--condition")
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--top_p", type=float)
    ap.add_argument("--error_class")
    ap.add_argument("--error_signature", help="substring match")
    ap.add_argument("--verification_status")
    ap.add_argument("--correct", choices=["true", "false"])
    ap.add_argument("--domain")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()

    rows = load_rows()
    out = []
    for r in rows:
        if args.model and r["model_id"] != args.model:
            continue
        if args.task and r["task_id"] != args.task:
            continue
        if args.prompt and r["prompt_id"] != args.prompt:
            continue
        if args.condition and r["condition_id"] != args.condition:
            continue
        if args.temperature is not None and r["temperature"] != args.temperature:
            continue
        if args.top_p is not None and r["top_p"] != args.top_p:
            continue
        if args.domain and r["domain"] != args.domain:
            continue
        v = r["verification"]
        if args.error_class and v["error_class"] != args.error_class:
            continue
        if args.error_signature and (
            not v["error_signature"] or args.error_signature not in v["error_signature"]
        ):
            continue
        if args.verification_status and v["status"] != args.verification_status:
            continue
        if args.correct == "true" and v["status"] not in PASS_STATUSES:
            continue
        if args.correct == "false" and v["status"] not in FAIL_STATUSES:
            continue
        out.append(r)

    print(f"# {len(out)} matching rows (showing up to {args.limit})", flush=True)
    for r in out[: args.limit]:
        if args.full:
            print(json.dumps(r, ensure_ascii=False))
        else:
            print(r["run_id"])


if __name__ == "__main__":
    main()
