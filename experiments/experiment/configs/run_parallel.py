"""
Orchestrator: launches run_batch.py as one OS subprocess per model, capping
concurrency (several small GGUF models comfortably co-reside on the single
6GB GPU). After all finish, merges the per-model index shards into
runs/index.jsonl (skipping run_ids already present, so merging is safe to
re-run).
"""
import argparse
import json
import os
import subprocess
import sys
import time

EXP_DIR = r"<PROJECT_ROOT>\trace-probe\experiment"
CONFIGS = os.path.join(EXP_DIR, "configs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="ALL")
    ap.add_argument("--tasks", default="ALL")
    ap.add_argument("--prompts", default="P1,P2,P3")
    ap.add_argument("--conditions", default="ALL")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--max-calls", type=int, default=None)
    ap.add_argument("--max-time-sec", type=float, default=None)
    ap.add_argument("--max-parallel", type=int, default=4)
    args = ap.parse_args()

    with open(os.path.join(CONFIGS, "models.json"), encoding="utf-8") as f:
        registry = json.load(f)
    model_ids = [m["model_id"] for m in registry["models"]]
    if args.models != "ALL":
        wanted = set(args.models.split(","))
        model_ids = [m for m in model_ids if m in wanted]

    def build_cmd(model_id):
        cmd = [
            sys.executable,
            os.path.join(CONFIGS, "run_batch.py"),
            "--model_id", model_id,
            "--tasks", args.tasks,
            "--prompts", args.prompts,
            "--conditions", args.conditions,
            "--repeats", str(args.repeats),
        ]
        if args.max_calls is not None:
            cmd += ["--max-calls", str(args.max_calls)]
        if args.max_time_sec is not None:
            cmd += ["--max-time-sec", str(args.max_time_sec)]
        return cmd

    pending = list(model_ids)
    running = {}  # model_id -> Popen
    logs = {}
    t0 = time.time()

    def kill_all_running():
        for mid, p in list(running.items()):
            if p.poll() is None:
                print(f"[cleanup] killing still-running worker for {mid} (pid={p.pid})")
                try:
                    p.kill()
                    p.wait(timeout=10)
                except Exception as e:
                    print(f"[cleanup] failed to kill {mid}: {e}")
        for lf in logs.values():
            try:
                lf.close()
            except Exception:
                pass

    try:
        while pending or running:
            while pending and len(running) < args.max_parallel:
                mid = pending.pop(0)
                log_path = os.path.join(EXP_DIR, "runs", f"log.{mid}.txt")
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                lf = open(log_path, "w", encoding="utf-8")
                logs[mid] = lf
                print(f"[launch] {mid}")
                running[mid] = subprocess.Popen(build_cmd(mid), stdout=lf, stderr=subprocess.STDOUT)

            time.sleep(2)
            finished = [mid for mid, p in running.items() if p.poll() is not None]
            for mid in finished:
                rc = running[mid].returncode
                logs[mid].close()
                print(f"[done] {mid} rc={rc} elapsed={round(time.time()-t0,1)}s")
                del running[mid]
    except BaseException:
        # any interruption (Ctrl+C, crash, tool timeout) must not leave GPU
        # worker processes orphaned and holding VRAM.
        kill_all_running()
        raise

    print(f"All done in {round(time.time()-t0,1)}s")

    # merge shards
    merged_path = os.path.join(EXP_DIR, "runs", "index.jsonl")
    seen = set()
    rows = []
    if os.path.exists(merged_path):
        with open(merged_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    if row["run_id"] not in seen:
                        seen.add(row["run_id"])
                        rows.append(row)
    for mid in model_ids:
        shard_path = os.path.join(EXP_DIR, "runs", f"index.{mid}.jsonl")
        if not os.path.exists(shard_path):
            continue
        with open(shard_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    if row["run_id"] not in seen:
                        seen.add(row["run_id"])
                        rows.append(row)
    with open(merged_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Merged index: {len(rows)} rows -> {merged_path}")


if __name__ == "__main__":
    main()
