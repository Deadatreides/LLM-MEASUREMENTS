"""
Single-model runner. Loads one GGUF model once (GPU-resident), then executes
every (task, prompt, condition, repeat) job assigned to it, writing:
  - one RAW folder per run under runs/raw/<run_id>/ (result.txt + meta.json)
  - one compact JSON line per run appended to runs/index.<model_id>.jsonl

Designed to be launched as an independent OS process (see run_parallel.py),
so several models can be resident on the GPU and generating concurrently.

Resumable: if a RUN_ID already exists in this model's index shard, it is
skipped, so re-running (e.g. pilot -> main) only fills in new jobs.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
from env_fix import fix_cuda_dll_path

fix_cuda_dll_path()
from llama_cpp import Llama  # noqa: E402
from chat_render import render_prompt, STOP_SEQUENCES  # noqa: E402
from verification import verify  # noqa: E402
from extraction import text_characteristics  # noqa: E402

MODELS_BASE = r"<PROJECT_ROOT>\trace-probe\models"
EXP_DIR = r"<PROJECT_ROOT>\trace-probe\experiment"

EXPERIMENT_ID = "trace-probe-llm-variance"
EXPERIMENT_VERSION = "1.0.0"

MAX_TOKENS_BY_DOMAIN = {"math": 512, "logic": 512, "code": 768}
N_CTX = 2048


def text_hash(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--tasks", default="ALL", help="comma-separated TASK_IDs or ALL")
    ap.add_argument("--prompts", default="P1,P2,P3", help="comma-separated PROMPT_IDs")
    ap.add_argument("--conditions", default="ALL", help="comma-separated CONDITION_IDs or ALL")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--max-calls", type=int, default=None)
    ap.add_argument("--max-time-sec", type=float, default=None)
    args = ap.parse_args()

    registry = load_json(os.path.join(EXP_DIR, "configs", "models.json"))
    model_entry = next((m for m in registry["models"] if m["model_id"] == args.model_id), None)
    if model_entry is None:
        raise SystemExit(f"unknown model_id {args.model_id}")

    all_tasks = load_jsonl(os.path.join(EXP_DIR, "tasks", "tasks.jsonl"))
    if args.tasks != "ALL":
        wanted = set(args.tasks.split(","))
        all_tasks = [t for t in all_tasks if t["task_id"] in wanted]

    prompt_rows = load_jsonl(os.path.join(EXP_DIR, "prompts", "prompts.jsonl"))
    prompts_by_key = {(p["prompt_id"], p["domain"]): p for p in prompt_rows}
    wanted_prompt_ids = args.prompts.split(",")

    all_conditions = load_json(os.path.join(EXP_DIR, "configs", "conditions.json"))
    if args.conditions != "ALL":
        wanted_c = set(args.conditions.split(","))
        all_conditions = [c for c in all_conditions if c["condition_id"] in wanted_c]

    raw_dir_base = os.path.join(EXP_DIR, "runs", "raw")
    os.makedirs(raw_dir_base, exist_ok=True)
    index_path = os.path.join(EXP_DIR, "runs", f"index.{args.model_id}.jsonl")

    existing_run_ids = set()
    if os.path.exists(index_path):
        for row in load_jsonl(index_path):
            existing_run_ids.add(row["run_id"])

    model_path = os.path.join(MODELS_BASE, model_entry["path"])
    t_load0 = time.time()
    llm = Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=N_CTX, verbose=False)
    load_time_sec = time.time() - t_load0

    stop = STOP_SEQUENCES.get(args.model_id)
    enable_thinking = False if model_entry["family"] == "qwen3" else None

    out_f = open(index_path, "a", encoding="utf-8")
    n_done = 0
    n_skipped = 0
    t_start = time.time()

    jobs = []
    for task in all_tasks:
        for prompt_id in wanted_prompt_ids:
            key = (prompt_id, task["domain"])
            if key not in prompts_by_key:
                continue
            for cond in all_conditions:
                for rep in range(1, args.repeats + 1):
                    jobs.append((task, prompt_id, cond, rep))

    for task, prompt_id, cond, rep in jobs:
        run_id = f"{args.model_id}--{task['task_id']}--{prompt_id}--{cond['condition_id']}--rep{rep:02d}"
        if run_id in existing_run_ids:
            n_skipped += 1
            continue
        if args.max_calls is not None and n_done >= args.max_calls:
            break
        if args.max_time_sec is not None and (time.time() - t_start) >= args.max_time_sec:
            break

        prompt_tpl = prompts_by_key[(prompt_id, task["domain"])]["template"]
        question_text = prompt_tpl.format(question=task["question"])
        try:
            rendered_prompt = render_prompt(
                llm,
                args.model_id,
                [{"role": "user", "content": question_text}],
                enable_thinking=enable_thinking,
            )
        except Exception as e:
            rendered_prompt = None
            render_error = f"{type(e).__name__}: {e}"
        else:
            render_error = None

        max_tokens = MAX_TOKENS_BY_DOMAIN[task["domain"]]
        seed = rep  # deterministic per-repeat seed, recorded explicitly below

        gen_failed = False
        gen_error = None
        raw_text = ""
        usage = {}
        t0 = time.time()
        if rendered_prompt is not None:
            try:
                out = llm.create_completion(
                    prompt=rendered_prompt,
                    max_tokens=max_tokens,
                    temperature=cond["temperature"],
                    top_p=cond["top_p"],
                    top_k=cond["top_k"],
                    seed=seed,
                    stop=stop,
                )
                raw_text = out["choices"][0]["text"]
                usage = out.get("usage", {}) or {}
            except Exception as e:
                gen_failed = True
                gen_error = f"{type(e).__name__}: {e}"
        else:
            gen_failed = True
            gen_error = render_error
        gen_time_sec = time.time() - t0

        raw_run_dir = os.path.join(raw_dir_base, run_id)
        os.makedirs(raw_run_dir, exist_ok=True)
        with open(os.path.join(raw_run_dir, "result.txt"), "w", encoding="utf-8") as rf:
            rf.write(raw_text)
        meta = {
            "run_id": run_id,
            "rendered_prompt": rendered_prompt,
            "question_text": question_text,
            "generation_params": {
                "temperature": cond["temperature"],
                "top_p": cond["top_p"],
                "top_k": cond["top_k"],
                "seed": seed,
                "max_tokens": max_tokens,
                "stop": stop,
                "n_ctx": N_CTX,
                "n_gpu_layers": -1,
                "enable_thinking": enable_thinking,
            },
            "generation_failed": gen_failed,
            "generation_error": gen_error,
            "usage": usage,
        }
        with open(os.path.join(raw_run_dir, "meta.json"), "w", encoding="utf-8") as mf:
            json.dump(meta, mf, ensure_ascii=False, indent=2)

        if gen_failed:
            verification = {
                "verification_status": "TECHNICAL_FAILURE",
                "error_class": "MODEL_FAILURE",
                "error_signature": f"generation_exception:{gen_error[:80] if gen_error else 'unknown'}",
                "extracted_final_answer": None,
                "answer_format": "unknown",
                "has_final_answer": False,
                "has_multiple_candidates": False,
                "has_code": False,
                "code_length": None,
                "syntax_valid": None,
                "execution_status": None,
                "tests_passed": None,
                "tests_failed": None,
                "exception_type": None,
            }
            chars = {
                "length_chars": 0,
                "has_explicit_uncertainty": None,
                "has_self_correction_language": None,
                "has_contradiction": None,
            }
        else:
            verification = verify(task["domain"], task, raw_text)
            chars = text_characteristics(raw_text)

        has_final_conclusion = (
            verification["has_code"] if task["domain"] == "code" else verification["has_final_answer"]
        )

        input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
        tokens_per_sec = (
            round(output_tokens / gen_time_sec, 2) if output_tokens and gen_time_sec > 0 else None
        )

        row = {
            "run_id": run_id,
            "experiment_id": EXPERIMENT_ID,
            "experiment_version": EXPERIMENT_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "condition_id": cond["condition_id"],
            "task_id": task["task_id"],
            "domain": task["domain"],
            "model_id": args.model_id,
            "model_family": model_entry["family"],
            "model_version": model_entry["model_version"],
            "model_size_label": model_entry["size_label"],
            "quantization": model_entry["quantization"],
            "architecture": model_entry["architecture"],
            "prompt_id": prompt_id,
            "prompt_text_hash": text_hash(rendered_prompt) if rendered_prompt else None,
            "temperature": cond["temperature"],
            "top_p": cond["top_p"],
            "top_k": cond["top_k"],
            "seed": seed,
            "max_tokens": max_tokens,
            "repeat_index": rep,
            "result_id": run_id,
            "result_status": "generation_failed" if gen_failed else "completed",
            "result": {
                "length_chars": chars["length_chars"],
                "has_final_answer": verification["has_final_answer"],
                "extracted_final_answer": verification["extracted_final_answer"],
                "answer_format": verification["answer_format"],
                "has_code": verification["has_code"],
                "code_length": verification["code_length"],
                "has_multiple_candidates": verification["has_multiple_candidates"],
                "has_explicit_uncertainty": chars["has_explicit_uncertainty"],
                "has_self_correction_language": chars["has_self_correction_language"],
                "has_contradiction": chars["has_contradiction"],
                "has_final_conclusion": has_final_conclusion,
            },
            "verification": {
                "status": verification["verification_status"],
                "error_class": verification["error_class"],
                "error_signature": verification["error_signature"],
                "syntax_valid": verification["syntax_valid"],
                "execution_status": verification["execution_status"],
                "tests_passed": verification["tests_passed"],
                "tests_failed": verification["tests_failed"],
                "exception_type": verification["exception_type"],
            },
            "performance": {
                "generation_time_sec": round(gen_time_sec, 4),
                "load_time_sec": round(load_time_sec, 4) if n_done == 0 else None,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "tokens_per_sec": tokens_per_sec,
            },
            "raw_path": f"raw/{run_id}/",
        }
        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        out_f.flush()
        n_done += 1

    out_f.close()
    del llm
    print(
        json.dumps(
            {
                "model_id": args.model_id,
                "jobs_total": len(jobs),
                "jobs_done_this_run": n_done,
                "jobs_skipped_existing": n_skipped,
                "wall_time_sec": round(time.time() - t_start, 2),
                "load_time_sec": round(load_time_sec, 2),
            }
        )
    )


if __name__ == "__main__":
    main()
