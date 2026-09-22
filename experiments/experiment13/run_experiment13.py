"""run_experiment13.py — CLI: --phase select|1|analyze, --pilot.

Чекпоинты -- атомарно (tmp+rename) в runs13/, агрегат -- в
metrics13/action_outcomes_*.json. Та же схема фаз/резюмируемости, что
run_experiment12.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from configs.model_registry import load_model  # noqa: E402
import harness  # noqa: E402
from harness import (  # noqa: E402
    ALL_MODES,
    EXPERIMENT_ID,
    MODES_MAIN,
    CollisionState13,
    PromptViolation,
    bootstrap_rate_diff,
    load_exp12_states,
    paired_sign_test,
    run_mode,
    select_states,
    unpaired_rate_diff,
)

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs13"
METRICS_DIR = ROOT / "metrics13"

MIN_CATEGORY_N = 15  # ниже этого порога разбивка по типу коллизии не даёт вывода (меньшая выборка эксп.13, порог мягче эксп.12)


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


def _select_path(tag: str) -> Path:
    return RUNS_DIR / f"select_{tag}_{EXPERIMENT_ID}.json"


def _phase1_path(tag: str) -> Path:
    return RUNS_DIR / f"phase1_{tag}_{EXPERIMENT_ID}.json"


# -- select ---------------------------------------------------------------


def cmd_select(args) -> None:
    tag = "pilot" if args.pilot else "full"
    n = args.n or (40 if args.pilot else 150)
    seed = args.seed or (20260101 if args.pilot else 20260102)

    all_states = load_exp12_states()
    print(f"[select:{tag}] loaded {len(all_states)} states from experiment12's phase-0 data")
    selected = select_states(all_states, n, seed=seed, arith_fraction=0.7)
    n_arith = sum(1 for s in selected if s.task_family == "arithmetic")
    n_code = len(selected) - n_arith
    print(f"[select:{tag}] selected {len(selected)} states ({n_arith} arithmetic, {n_code} code), seed={seed}")

    out_path = _select_path(tag)
    _atomic_write_json(
        out_path,
        {
            "experiment_id": EXPERIMENT_ID, "tag": tag, "n": n, "seed": seed,
            "states": [s.__dict__ for s in selected],
        },
    )
    print(f"[select:{tag}] saved -> {out_path}")


# -- phase 1 ----------------------------------------------------------------


def cmd_phase1(args) -> None:
    tag = "pilot" if args.pilot else "full"
    data = _load_json(_select_path(tag))
    if data is None:
        print(f"[phase1] no selection at {_select_path(tag)} -- run --phase select first")
        return

    states = [CollisionState13(**s) for s in data["states"]]
    modes = list(MODES_MAIN) + (["R0B"] if args.include_r0b else [])
    print(f"[phase1:{tag}] {len(states)} states x {len(modes)} modes ({modes})")

    by_model: dict = defaultdict(list)
    for s in states:
        by_model[s.model_id].append(s)

    results: list = []
    out_path = _phase1_path(tag)
    t0 = time.time()

    try:
        for model_id, model_states in by_model.items():
            llm, load_t = load_model(model_id)
            print(f"[phase1:{tag}] loaded {model_id} in {load_t:.1f}s -- {len(model_states)} states, {len(model_states) * len(modes)} generations queued")
            for s in model_states:
                for mode in modes:
                    res = run_mode(s, mode, llm)
                    results.append(res)
            del llm
            _atomic_write_json(
                out_path,
                {"experiment_id": EXPERIMENT_ID, "tag": tag, "n_states": len(states), "modes": modes, "complete": False, "results": results},
            )
            print(f"[phase1:{tag}] checkpoint -- {len(results)} results so far")
    except PromptViolation as exc:
        _atomic_write_json(
            out_path,
            {"experiment_id": EXPERIMENT_ID, "tag": tag, "n_states": len(states), "modes": modes, "complete": False,
             "stopped_on_violation": str(exc), "results": results},
        )
        print(f"[phase1:{tag}] STOPPED -- prompt validation caught a violation: {exc}")
        print("Per protocol: fix the defect, then start a NEW campaign with a NEW experiment_id -- do not patch over this data.")
        raise SystemExit(1)

    elapsed = time.time() - t0
    _atomic_write_json(
        out_path,
        {"experiment_id": EXPERIMENT_ID, "tag": tag, "n_states": len(states), "modes": modes, "complete": True,
         "elapsed_sec": elapsed, "results": results},
    )
    print(f"[phase1:{tag}] done in {elapsed:.1f}s -- {len(results)} results -> {out_path}")


# -- analyze ------------------------------------------------------------------


def _paired(by_collision: dict, mode_a: str, mode_b: str):
    ids = [cid for cid, modes in by_collision.items() if mode_a in modes and mode_b in modes]
    return ids, [by_collision[c][mode_a] for c in ids], [by_collision[c][mode_b] for c in ids]


def cmd_analyze(args) -> None:
    tag = "pilot" if args.pilot else "full"
    phase1 = _load_json(_phase1_path(tag))
    if phase1 is None:
        print(f"[analyze:{tag}] no phase1 data -- run --phase 1 first")
        return
    if not phase1.get("complete", False):
        print(f"[analyze:{tag}] WARNING: phase1 data not marked complete -- analyzing partial data anyway")

    results = phase1["results"]
    modes = phase1["modes"]

    by_collision: dict = {}
    for r in results:
        by_collision.setdefault(r["collision_id"], {})[r["mode"]] = r
    n_states = len(by_collision)

    # -- per-mode metrics --
    FINAL_STATUSES = ("PASS", "SAME_FAILURE", "DIFFERENT_FAILURE", "INAPPLICABLE", "UNVERIFIED", "ERROR")
    mode_metrics = {}
    for mode in modes:
        attempts = [by_collision[cid][mode] for cid in by_collision if mode in by_collision[cid]]
        n = len(attempts)
        if n == 0:
            continue
        status_counts = {s: sum(1 for a in attempts if a.get("final_status") == s) for s in FINAL_STATUSES}
        n_solved = status_counts["PASS"]
        total_tokens = sum(a.get("tokens", 0) or 0 for a in attempts)
        latencies = sorted(a["latency"] for a in attempts if a.get("latency") is not None)
        pass_latencies = sorted(a["latency"] for a in attempts if a.get("solved") and a.get("latency") is not None)
        novelty_counts = defaultdict(int)
        for a in attempts:
            if a.get("representation_novelty"):
                novelty_counts[a["representation_novelty"]] += 1

        mode_metrics[mode] = {
            "n_attempts": n,
            "pass_rate": n_solved / n,
            "same_failure_rate": status_counts["SAME_FAILURE"] / n,
            "different_failure_rate": status_counts["DIFFERENT_FAILURE"] / n,
            "inapplicable_rate": status_counts["INAPPLICABLE"] / n,
            "unverified_rate": status_counts["UNVERIFIED"] / n,
            "error_rate": status_counts["ERROR"] / n,
            "total_tokens": total_tokens,
            "tokens_per_pass": (total_tokens / n_solved) if n_solved else None,
            "calls_per_pass": (n / n_solved) if n_solved else None,
            "median_latency_sec": latencies[len(latencies) // 2] if latencies else None,
            "median_time_to_first_pass_sec": pass_latencies[len(pass_latencies) // 2] if pass_latencies else None,
            "novelty_distribution": dict(novelty_counts),
        }

    if "R0" in mode_metrics and mode_metrics["R0"].get("tokens_per_pass"):
        base = mode_metrics["R0"]["tokens_per_pass"]
        for mode, m in mode_metrics.items():
            if mode != "R0" and m.get("tokens_per_pass"):
                m["efficiency_gain_vs_r0"] = base / m["tokens_per_pass"]

    # -- primary paired tests: DeltaR1, DeltaR2 (vs R0) --
    delta_tests = {}
    for mode, label in (("R1", "DeltaR1: R1 vs R0"), ("R2", "DeltaR2: R2 vs R0")):
        if mode not in modes:
            continue
        ids, a_att, b_att = _paired(by_collision, mode, "R0")
        if not ids:
            continue
        a_solved = [a["solved"] for a in a_att]
        b_solved = [b["solved"] for b in b_att]
        delta_tests[label] = {
            "n_paired": len(ids),
            "sign_test": paired_sign_test(a_solved, b_solved),
            "pass_rate_diff": bootstrap_rate_diff(a_solved, b_solved),
        }

    # -- DeltaNovel: P(PASS|NOVEL_STRUCTURE) vs P(PASS|SAME_STRUCTURE), pooled over R1+R2 (unpaired subgroup comparison) --
    novel_solved, same_solved = [], []
    for mode in ("R1", "R2"):
        if mode not in modes:
            continue
        for cid, m in by_collision.items():
            a = m.get(mode)
            if not a:
                continue
            if a.get("representation_novelty") == "NOVEL_STRUCTURE":
                novel_solved.append(a["solved"])
            elif a.get("representation_novelty") == "SAME_STRUCTURE":
                same_solved.append(a["solved"])
    delta_novel = unpaired_rate_diff(novel_solved, same_solved)
    delta_novel["pass_rate_novel"] = (sum(novel_solved) / len(novel_solved)) if novel_solved else None
    delta_novel["pass_rate_same"] = (sum(same_solved) / len(same_solved)) if same_solved else None

    # -- breakdown by original collision failure_type --
    by_type: dict = defaultdict(list)
    for cid, m in by_collision.items():
        ftype = next(iter(m.values()))["failure_type"] or "other"
        by_type[ftype].append(cid)

    breakdown = {}
    for ftype, cids in by_type.items():
        if len(cids) < MIN_CATEGORY_N:
            breakdown[ftype] = {"n": len(cids), "note": f"insufficient data, no conclusion drawn (n < {MIN_CATEGORY_N})"}
            continue
        per_mode = {}
        for mode in modes:
            attempts = [by_collision[cid][mode] for cid in cids if mode in by_collision.get(cid, {})]
            if not attempts:
                continue
            n_solved = sum(1 for a in attempts if a["solved"])
            per_mode[mode] = {"n": len(attempts), "pass_rate": n_solved / len(attempts)}
        breakdown[ftype] = {"n": len(cids), "per_mode": per_mode}

    # -- A/B/C/D outcome table (задание §16) --
    def _novel_rate(mode):
        attempts = [by_collision[cid][mode] for cid in by_collision if mode in by_collision.get(cid, {})]
        novel = sum(1 for a in attempts if a.get("representation_novelty") == "NOVEL_STRUCTURE")
        return (novel / len(attempts)) if attempts else None

    outcome_table = {
        mode: {
            "novel_structure_rate": _novel_rate(mode),
            "pass_rate": mode_metrics.get(mode, {}).get("pass_rate"),
            "pass_rate_vs_r0": (mode_metrics.get(mode, {}).get("pass_rate", 0) - mode_metrics.get("R0", {}).get("pass_rate", 0))
            if mode in mode_metrics and "R0" in mode_metrics else None,
        }
        for mode in ("R1", "R2") if mode in modes
    }

    # -- §17: 5 gating criteria (checked, not decided automatically -- report writes the verdict) --
    gate_facts = {
        "measurable_novel_structure_share": {m: outcome_table[m]["novel_structure_rate"] for m in outcome_table},
        "delta_novel": delta_novel,
        "delta_vs_r0": {k: v["pass_rate_diff"] for k, v in delta_tests.items()},
        "efficiency_gain": {m: mode_metrics.get(m, {}).get("efficiency_gain_vs_r0") for m in ("R1", "R2") if m in mode_metrics},
    }

    result = {
        "experiment_id": EXPERIMENT_ID, "tag": tag,
        "n_states": n_states, "modes": modes,
        "mode_metrics": mode_metrics,
        "delta_tests": delta_tests,
        "delta_novel": delta_novel,
        "breakdown_by_failure_type": breakdown,
        "outcome_table_A_B_C_D": outcome_table,
        "gate_facts_section17": gate_facts,
    }

    out_path = METRICS_DIR / f"action_outcomes_{tag}.json"
    _atomic_write_json(out_path, result)
    print(f"[analyze:{tag}] saved -> {out_path}")
    print(json.dumps({k: v for k, v in result.items() if k not in ("mode_metrics", "breakdown_by_failure_type")}, ensure_ascii=False, indent=2, default=str))
    print("mode_metrics:")
    print(json.dumps(mode_metrics, ensure_ascii=False, indent=2, default=str))
    print("breakdown_by_failure_type:")
    print(json.dumps(breakdown, ensure_ascii=False, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 13 CLI")
    parser.add_argument("--phase", choices=["select", "1", "analyze"], required=True)
    parser.add_argument("--pilot", action="store_true", help="small pilot run (30-50 states)")
    parser.add_argument("--n", type=int, default=None, help="number of states to select")
    parser.add_argument("--seed", type=int, default=None, help="selection seed")
    parser.add_argument("--include-r0b", action="store_true", help="also run the R0+B control (deferred by default per spec §13)")
    args = parser.parse_args()

    if args.phase == "select":
        cmd_select(args)
    elif args.phase == "1":
        cmd_phase1(args)
    else:
        cmd_analyze(args)


if __name__ == "__main__":
    main()
