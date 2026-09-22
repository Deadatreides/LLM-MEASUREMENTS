"""run_experiment11.py — CLI: --phase 0|1|analyze, --pilot.

Чекпоинты пишутся атомарно (tmp + rename) в runs11/, агрегат -- в
metrics11/action_outcomes.json. Разделение фаз позволяет возобновление:
`--phase 1` перечитывает уже сохранённый `phase0_*` файл, не повторяя GPU-работу.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from configs.model_registry import MODEL_IDS, load_model  # noqa: E402
from tasks.arithmetic_tasks import TASK_IDS as ARITH_TASK_IDS  # noqa: E402
from tasks.code_tasks import CODE_TASK_IDS  # noqa: E402
import harness  # noqa: E402
from harness import (  # noqa: E402
    ARMS,
    EXPERIMENT_ID,
    MIN_CATEGORY_N,
    CollisionState,
    best_of_n,
    bootstrap_ratio_diff,
    build_arm_configs,
    find_collision_states,
    paired_sign_test,
    run_arm,
    run_seek_evidence,
    run_stop,
)

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs11"
METRICS_DIR = ROOT / "metrics11"


def _atomic_write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    tmp.replace(path)


def _load_json(path: Path, default=None):
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _phase0_path(tag: str) -> Path:
    return RUNS_DIR / f"phase0_{tag}_{EXPERIMENT_ID}.json"


def _phase1_path(tag: str) -> Path:
    return RUNS_DIR / f"phase1_{tag}_{EXPERIMENT_ID}.json"


# -- phase 0 ------------------------------------------------------------------


def cmd_phase0(args) -> None:
    tag = "pilot" if args.pilot else "full"
    model_ids = list(MODEL_IDS)[:3] if args.pilot else list(MODEL_IDS)
    seeds = list(harness.INITIAL_SEEDS[:2]) if args.pilot else list(harness.INITIAL_SEEDS)
    arith_ids = list(ARITH_TASK_IDS)[:6] if args.pilot else list(ARITH_TASK_IDS)
    code_ids = list(CODE_TASK_IDS)[:4] if args.pilot else list(CODE_TASK_IDS)

    print(f"[phase0:{tag}] models={model_ids}")
    print(f"[phase0:{tag}] seeds={seeds} arith_tasks={len(arith_ids)} code_tasks={len(code_ids)}")
    t0 = time.time()
    states, records = find_collision_states(model_ids, seeds, arith_ids, code_ids, log=print)
    elapsed = time.time() - t0
    n_pass = sum(1 for r in records if r["status"] == "PASS")
    print(
        f"[phase0:{tag}] done in {elapsed:.1f}s -- {len(records)} initial attempts, "
        f"{len(states)} FAIL collision-states, {n_pass} PASS on first try"
    )

    out_path = _phase0_path(tag)
    _atomic_write_json(
        out_path,
        {
            "experiment_id": EXPERIMENT_ID,
            "tag": tag,
            "model_ids": model_ids,
            "seeds": seeds,
            "arithmetic_task_ids": arith_ids,
            "code_task_ids": code_ids,
            "elapsed_sec": elapsed,
            "states": [dataclasses.asdict(s) for s in states],
            "initial_records": records,
        },
    )
    print(f"[phase0:{tag}] saved -> {out_path}")


# -- phase 1 ------------------------------------------------------------------


def cmd_phase1(args) -> None:
    tag = "pilot" if args.pilot else "full"
    data = _load_json(_phase0_path(tag))
    if data is None:
        print(f"[phase1] no phase0 data at {_phase0_path(tag)} -- run --phase 0 first")
        return

    states = [CollisionState(**s) for s in data["states"]]
    all_model_ids = data["model_ids"]

    cap = args.cap or (20 if args.pilot else 150)
    if len(states) > cap:
        rng = random.Random(20260816)
        states = rng.sample(states, cap)
    print(f"[phase1:{tag}] {len(states)} collision-states selected (cap={cap})")

    config_rng = random.Random(20260817)
    tasks_by_model: dict = {}
    for s in states:
        configs = build_arm_configs(s.model_id, all_model_ids, config_rng)
        for arm_name, config in configs.items():
            tasks_by_model.setdefault(config.model_id, []).append((s, arm_name, config))

    arm_results: list = []
    out_path = _phase1_path(tag)
    t0 = time.time()

    for model_id, work in tasks_by_model.items():
        llm, load_t = load_model(model_id)
        print(f"[phase1:{tag}] loaded {model_id} in {load_t:.1f}s -- {len(work)} generations queued")
        for s, arm_name, config in work:
            res = run_arm(s, config, llm)
            arm_results.append(res)
        del llm
        _atomic_write_json(
            out_path,
            {"experiment_id": EXPERIMENT_ID, "tag": tag, "n_states": len(states), "arm_results": arm_results},
        )
        print(f"[phase1:{tag}] checkpoint -- {len(arm_results)} arm results so far")

    for s in states:
        arm_results.append(run_seek_evidence(s))
        arm_results.append(run_stop(s))

    elapsed = time.time() - t0
    _atomic_write_json(
        out_path,
        {
            "experiment_id": EXPERIMENT_ID,
            "tag": tag,
            "n_states": len(states),
            "elapsed_sec": elapsed,
            "arm_results": arm_results,
        },
    )
    print(f"[phase1:{tag}] done in {elapsed:.1f}s -- {len(arm_results)} arm results -> {out_path}")


# -- analyze ------------------------------------------------------------------


def _paired(by_collision: dict, arm_a: str, arm_b: str):
    ids = [cid for cid, arms in by_collision.items() if arm_a in arms and arm_b in arms]
    return ids, [by_collision[c][arm_a] for c in ids], [by_collision[c][arm_b] for c in ids]


def cmd_analyze(args) -> None:
    tag = "pilot" if args.pilot else "full"
    phase0 = _load_json(_phase0_path(tag))
    phase1 = _load_json(_phase1_path(tag))
    if phase0 is None or phase1 is None:
        print(f"[analyze:{tag}] missing phase0/phase1 data -- run --phase 0 and --phase 1 first")
        return

    states = phase0["states"]
    initial_records = phase0["initial_records"]
    arm_results = phase1["arm_results"]
    state_by_id = {s["collision_id"]: s for s in states}

    by_collision: dict = {}
    for r in arm_results:
        by_collision.setdefault(r["collision_id"], {})[r["arm"]] = r

    arm_metrics = {}
    for arm in list(ARMS) + ["SEEK_EVIDENCE", "STOP"]:
        attempts = [by_collision[cid][arm] for cid in by_collision if arm in by_collision[cid]]
        n = len(attempts)
        if n == 0:
            continue
        n_solved = sum(1 for a in attempts if a.get("solved"))
        n_unknown = sum(1 for a in attempts if a.get("status") == "UNKNOWN")
        n_inapplicable = sum(1 for a in attempts if a.get("status") == "INAPPLICABLE")
        n_still_fail = sum(1 for a in attempts if a.get("status") == "FAIL")
        n_error = sum(1 for a in attempts if a.get("status") == "ERROR")
        n_applicable = n - n_inapplicable
        total_tokens = sum(a.get("tokens", 0) or 0 for a in attempts)
        latencies = sorted(a["latency"] for a in attempts if a.get("latency") is not None)
        same_fail_attempts = [a for a in attempts if a.get("same_failure") is not None]
        n_no_gain = sum(1 for a in attempts if not a.get("solved") and a.get("same_failure") == "same_failure")
        signatures = [tuple(a["config"].values()) for a in attempts if "config" in a]
        n_repeated_configs = len(signatures) - len(set(signatures)) if signatures else 0

        arm_metrics[arm] = {
            "n_attempts": n,
            "n_model_calls": n if arm not in ("SEEK_EVIDENCE", "STOP") else 0,
            "resolution_rate": n_solved / n,
            "conditional_resolution_rate": (n_solved / n_applicable) if n_applicable else None,
            "unknown_rate": n_unknown / n,
            "inapplicable_rate": n_inapplicable / n,
            "n_still_fail": n_still_fail,
            "n_error": n_error,
            "total_tokens": total_tokens,
            "tokens_per_resolved_task": (total_tokens / n_solved) if n_solved else None,
            "median_latency_sec": latencies[len(latencies) // 2] if latencies else None,
            "p90_latency_sec": latencies[int(0.9 * len(latencies))] if latencies else None,
            "same_failure_rate": (
                sum(1 for a in same_fail_attempts if a["same_failure"] == "same_failure") / len(same_fail_attempts)
                if same_fail_attempts
                else None
            ),
            "no_gain_rate": n_no_gain / n,
            "n_repeated_identical_configurations": n_repeated_configs,
        }

    # -- primary + secondary paired tests (метод и пары фиксированы planом ДО прогона) --
    tests = {}
    for arm, label in (
        ("CHANGE_MODEL", "primary: CHANGE_MODEL vs REGENERATE_SAME"),
        ("CHANGE_PROMPT", "secondary: CHANGE_PROMPT vs REGENERATE_SAME"),
        ("CHANGE_TEMPERATURE", "secondary: CHANGE_TEMPERATURE vs REGENERATE_SAME"),
    ):
        ids, a_att, b_att = _paired(by_collision, arm, "REGENERATE_SAME")
        if not ids:
            continue
        sign_test = paired_sign_test([a["solved"] for a in a_att], [b["solved"] for b in b_att])
        boot = bootstrap_ratio_diff(a_att, b_att)
        tests[label] = {"n_paired": len(ids), "sign_test": sign_test, "bootstrap_token_cost_diff": boot}

    # -- same-failure correlation (п.8): P(same failure | REGENERATE_SAME) vs P(.. | CHANGE_MODEL) --
    ids, a_att, b_att = _paired(by_collision, "CHANGE_MODEL", "REGENERATE_SAME")
    same_failure_corr = None
    if ids:
        a_same = [x.get("same_failure") == "same_failure" for x in a_att]
        b_same = [x.get("same_failure") == "same_failure" for x in b_att]
        same_failure_corr = {
            "n_paired": len(ids),
            "p_same_failure_given_REGENERATE_SAME": sum(b_same) / len(b_same),
            "p_same_failure_given_CHANGE_MODEL": sum(a_same) / len(a_same),
        }

    # -- breakdown by collision type (п.9) --
    # ВАЖНО: только states, реально прошедшие через phase 1 (by_collision) --
    # не все найденные в phase 0. Иначе "n" здесь считал бы состояния, для
    # которых ни одно плечо не запускалось (обрезаны --cap), что задним
    # числом искажает читаемое "n" в отчёте относительно фактических данных.
    by_type: dict = {}
    for cid in by_collision:
        state = state_by_id[cid]
        ctype = (state.get("initial_seam_result") or {}).get("collision_type") or "format"
        by_type.setdefault(ctype, []).append(cid)

    breakdown = {}
    for ctype, cids in by_type.items():
        if len(cids) < MIN_CATEGORY_N:
            breakdown[ctype] = {"n": len(cids), "note": f"insufficient data, no conclusion drawn (n < {MIN_CATEGORY_N})"}
            continue
        per_arm = {}
        for arm in ARMS:
            attempts = [by_collision[cid][arm] for cid in cids if arm in by_collision.get(cid, {})]
            if not attempts:
                continue
            n_solved = sum(1 for a in attempts if a.get("solved"))
            per_arm[arm] = {"n": len(attempts), "resolution_rate": n_solved / len(attempts)}
        breakdown[ctype] = {"n": len(cids), "per_arm": per_arm}

    bon = best_of_n(initial_records)

    analyzed_ids = set(by_collision)
    n_arith_found = sum(1 for s in states if s["task_family"] == "arithmetic")
    n_code_found = sum(1 for s in states if s["task_family"] == "code")
    n_arith_analyzed = sum(1 for cid in analyzed_ids if state_by_id[cid]["task_family"] == "arithmetic")
    n_code_analyzed = sum(1 for cid in analyzed_ids if state_by_id[cid]["task_family"] == "code")

    result = {
        "experiment_id": EXPERIMENT_ID,
        "tag": tag,
        "n_initial_attempts": len(initial_records),
        "n_collision_states_found": len(states),
        "n_collision_states_analyzed": len(analyzed_ids),
        "n_arithmetic_states_found": n_arith_found,
        "n_code_states_found": n_code_found,
        "n_arithmetic_states_analyzed": n_arith_analyzed,
        "n_code_states_analyzed": n_code_analyzed,
        "arm_metrics": arm_metrics,
        "statistical_tests": tests,
        "same_failure_correlation": same_failure_corr,
        "breakdown_by_collision_type": breakdown,
        "best_of_n": bon,
    }

    out_path = METRICS_DIR / f"action_outcomes_{tag}.json"
    _atomic_write_json(out_path, result)
    print(f"[analyze:{tag}] saved -> {out_path}")
    print(json.dumps({k: v for k, v in result.items() if k != "arm_metrics"}, ensure_ascii=False, indent=2)[:3000])
    print("arm_metrics:")
    print(json.dumps(arm_metrics, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 11 CLI")
    parser.add_argument("--phase", choices=["0", "1", "analyze"], required=True)
    parser.add_argument("--pilot", action="store_true", help="small pilot run (~20 states, 3 models, 2 seeds)")
    parser.add_argument("--cap", type=int, default=None, help="max collision-states to process in phase 1")
    args = parser.parse_args()

    if args.phase == "0":
        cmd_phase0(args)
    elif args.phase == "1":
        cmd_phase1(args)
    else:
        cmd_analyze(args)


if __name__ == "__main__":
    main()
