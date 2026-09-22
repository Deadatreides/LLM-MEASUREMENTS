"""run_e3.py — E3: rho_FILTER (F2, T_hard) and rho_SUM. 0 new calls.
PROTOCOL_E1E2E3.md SS4: rho_FILTER reconstructed from E1/E2's own
`metrics/filter_calls.jsonl` (already-logged `extracted` per call, no
re-query). rho_SUM reused from FORK-1's own already-collected
`path_b_whole_detail.json` (read-only). Run only after run_e1.py AND
run_e2.py have both completed.

Same formula as `rho_holdout.py`:
  rho = [P(both wrong) - P(wrong)^2] / [P(wrong)(1-P(wrong))]
For FILTER, "wrong" is pooled over every (task, candidate_id) pair (ids
are positional/task-local, no cross-task field identity to key on --
same reasoning PROTOCOL_E1E2E3.md SS1 gives for balanced-accuracy
calibration). Falsification (ESW_THEORY.md SS7): rho_FILTER < 0.4
(variance-type), rho_SUM > 0.8 (bias-type).
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import task_generator_f2 as TF2       # noqa: E402
import task_generator_hard as THARD   # noqa: E402
import consensus_filter as CF         # noqa: E402
import model_registry_11 as MR        # noqa: E402

AGENT_DIR = Path(__file__).resolve().parents[1]
METRICS_DIR = AGENT_DIR / "metrics"
FORK1_WHOLE_DETAIL = AGENT_DIR.parent / "agent_fork1_three_paths" / "metrics" / "path_b_whole_detail.json"

THR_FILTER = 0.4
THR_SUM = 0.8


def load_filter_votes(family: str, task_ids: list, model_ids) -> dict:
    path = METRICS_DIR / "filter_calls.jsonl"
    votes = {t: {} for t in task_ids}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["family"] != family or row["task_id"] not in votes:
            continue
        votes[row["task_id"]][row["model"]] = {"extracted": frozenset(row["extracted"])}
    missing = [t for t in task_ids for m in model_ids if m not in votes[t]]
    if missing:
        raise RuntimeError(f"filter_calls.jsonl missing {len(missing)} (task,model) rows for "
                           f"family={family} -- run_e1.py/run_e2.py must both finish before run_e3.py")
    return votes


def rho_pooled(wrong: dict, model_ids) -> tuple:
    p_bar = statistics.mean(statistics.mean(wrong[m]) for m in model_ids)
    pair_both = []
    for i, m1 in enumerate(model_ids):
        for m2 in model_ids[i + 1:]:
            pair_both.append(statistics.mean(a and b for a, b in zip(wrong[m1], wrong[m2])))
    p_both = statistics.mean(pair_both)
    denom = p_bar * (1 - p_bar)
    rho = (p_both - p_bar ** 2) / denom if denom > 1e-9 else float("nan")
    return rho, p_bar


def rho_filter(family: str, tasks_by_id: dict, task_ids: list, model_ids) -> dict:
    votes = load_filter_votes(family, task_ids, model_ids)
    wrong = {m: [] for m in model_ids}
    for tid in task_ids:
        task = tasks_by_id[tid]
        golden = CF.golden_map(task)
        pv = CF.per_id_votes(task, votes[tid])
        for cid, g in golden.items():
            for m in model_ids:
                wrong[m].append(1 if pv[m][cid] != g else 0)
    rho, p_bar = rho_pooled(wrong, model_ids)
    return {"rho": rho, "p_wrong": p_bar, "n_trials_per_model": len(next(iter(wrong.values())))}


def rho_sum(model_ids) -> dict:
    detail = json.loads(FORK1_WHOLE_DETAIL.read_text(encoding="utf-8"))
    task_ids = sorted(next(iter(detail.values())).keys())
    wrong = {m: [1 - detail[m][t]["v"] for t in task_ids] for m in model_ids}
    rho, p_bar = rho_pooled(wrong, model_ids)
    return {"rho": rho, "p_wrong": p_bar, "n_trials_per_model": len(task_ids)}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    f2 = TF2.build_tasks()
    hard = THARD.build_tasks()
    model_ids = list(MR.MODEL_IDS)

    r_f2 = rho_filter("E1-F2", f2["tasks"], f2["test"], model_ids)
    r_hard = rho_filter("E2-Thard", hard["tasks"], hard["test"], model_ids)
    r_sum = rho_sum(model_ids)

    def verdict_filter(r):
        if r != r:   # NaN check (P(wrong) degenerate at 0 or 1)
            return "NAN"
        return "OK_BELOW_0.4" if r < THR_FILTER else "ABOVE_0.4_variance_claim_falsified"

    def verdict_sum(r):
        return "OK_ABOVE_0.8" if (r == r and r > THR_SUM) else "BELOW_0.8_or_NAN_bias_claim_falsified"

    result = {
        "rho_FILTER_F2": r_f2, "rho_FILTER_Thard": r_hard, "rho_SUM": r_sum,
        "threshold_FILTER": THR_FILTER, "threshold_SUM": THR_SUM,
        "verdict_FILTER_F2": verdict_filter(r_f2["rho"]),
        "verdict_FILTER_Thard": verdict_filter(r_hard["rho"]),
        "verdict_SUM": verdict_sum(r_sum["rho"]),
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "e3_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"rho_FILTER_F2    = {r_f2['rho']:+.3f}  (P(wrong)={r_f2['p_wrong']:.3f})  "
         f"-> {result['verdict_FILTER_F2']}", flush=True)
    print(f"rho_FILTER_Thard = {r_hard['rho']:+.3f}  (P(wrong)={r_hard['p_wrong']:.3f})  "
         f"-> {result['verdict_FILTER_Thard']}", flush=True)
    print(f"rho_SUM          = {r_sum['rho']:+.3f}  (P(wrong)={r_sum['p_wrong']:.3f})  "
         f"-> {result['verdict_SUM']}", flush=True)


if __name__ == "__main__":
    main()
