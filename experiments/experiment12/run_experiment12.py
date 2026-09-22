"""run_experiment12.py — CLI: --phase 0|1|analyze, --pilot.

Чекпоинты -- атомарно (tmp+rename) в runs12/, агрегат -- в
metrics12/action_outcomes_*.json. Та же схема фаз/резюмируемости, что
run_experiment11.py.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Сырой текст генерации модели может содержать произвольный Unicode;
# консоль Windows по умолчанию не UTF-8 -- без этого печать диагностики
# (включая LeakDetected) может сама упасть с UnicodeEncodeError на
# критичном пути остановки кампании (поймано на пилоте).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from configs.model_registry import MODEL_IDS, load_model  # noqa: E402
from tasks.arithmetic_multistep_tasks import TASK_IDS as ARITH_TASK_IDS, TASKS as ARITH_TASKS  # noqa: E402
from tasks.code_tasks import CODE_TASK_IDS, CODE_TASKS  # noqa: E402
import context_manifest  # noqa: E402
import harness  # noqa: E402
import seams  # noqa: E402
from harness import (  # noqa: E402
    ALL_ARMS,
    ARM_H_SYN,
    ARMS_MAIN,
    EXPERIMENT_ID,
    MIN_CATEGORY_N,
    MIN_HSYN_N,
    CollisionState,
    LeakDetected,
    blind_resample_baseline,
    bootstrap_interaction,
    bootstrap_rate_diff,
    bootstrap_ratio_diff,
    build_arm_configs,
    find_collision_states,
    paired_sign_test,
    run_arm,
)
from tests.test_experiment12_v2_sanity import run_sanity_checklist  # noqa: E402

FINAL_STATUSES = ("PASS", "SAME_FAILURE", "DIFFERENT_FAILURE", "INAPPLICABLE", "UNVERIFIED", "ERROR")
_SOURCE_EXPERIMENT_ID = "exp12-v1"  # фаза 0 переиспользуется от v1 -- см. план ("не перегенерировать заново")

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs12"
METRICS_DIR = ROOT / "metrics12"


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


def _phase0_path(tag: str, experiment_id: str = EXPERIMENT_ID) -> Path:
    return RUNS_DIR / f"phase0_{tag}_{experiment_id}.json"


def _phase1_path(tag: str) -> Path:
    return RUNS_DIR / f"phase1_{tag}_{EXPERIMENT_ID}.json"


def _load_or_reuse_phase0(tag: str):
    """v2 не меняет протокол фазы 0 (тот же датасет задач, те же модели,
    то же извлечение) -- только протокол ремонта/классификации в фазе 1.
    Поэтому фаза 0 переиспользуется от v1 буквально (план: "не
    перегенерировать заново"), с явной пометкой источника в самом файле,
    а не тихим копированием."""
    own_path = _phase0_path(tag)
    data = _load_json(own_path)
    if data is not None:
        return data, own_path
    source_path = _phase0_path(tag, _SOURCE_EXPERIMENT_ID)
    source_data = _load_json(source_path)
    if source_data is None:
        return None, own_path
    source_data = dict(source_data)
    source_data["experiment_id"] = EXPERIMENT_ID
    source_data["reused_phase0_from"] = _SOURCE_EXPERIMENT_ID
    _atomic_write_json(own_path, source_data)
    print(f"[phase0:{tag}] reused phase-0 data from {source_path} -> {own_path} (same task/model/seed protocol, unchanged by v2)")
    return source_data, own_path


# -- phase 0 ------------------------------------------------------------------


def cmd_phase0(args) -> None:
    tag = "pilot" if args.pilot else "full"
    model_ids = list(MODEL_IDS)[:3] if args.pilot else list(MODEL_IDS)
    seeds = list(harness.INITIAL_SEEDS[:2]) if args.pilot else list(harness.INITIAL_SEEDS)
    arith_ids = list(ARITH_TASK_IDS)[:8] if args.pilot else list(ARITH_TASK_IDS)
    code_ids = list(CODE_TASK_IDS)[:6] if args.pilot else list(CODE_TASK_IDS)

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
            "experiment_id": EXPERIMENT_ID, "tag": tag, "model_ids": model_ids, "seeds": seeds,
            "arithmetic_task_ids": arith_ids, "code_task_ids": code_ids, "elapsed_sec": elapsed,
            "states": [dataclasses.asdict(s) for s in states], "initial_records": records,
        },
    )
    print(f"[phase0:{tag}] saved -> {out_path}")


# -- phase 1 ------------------------------------------------------------------


def cmd_phase1(args) -> None:
    tag = "pilot" if args.pilot else "full"
    data, _ = _load_or_reuse_phase0(tag)
    if data is None:
        print(f"[phase1] no phase0 data for tag={tag} under {EXPERIMENT_ID} or {_SOURCE_EXPERIMENT_ID} -- run --phase 0 first")
        return

    states = [CollisionState(**s) for s in data["states"]]
    all_model_ids = data["model_ids"]

    cap = args.cap or (20 if args.pilot else 500)
    if len(states) > cap:
        rng = random.Random(20260817)
        states = rng.sample(states, cap)
    print(f"[phase1:{tag}] {len(states)} collision-states selected (cap={cap})")

    config_rng = random.Random(20260818)
    tasks_by_model: dict = {}
    for s in states:
        configs = build_arm_configs(s.model_id, all_model_ids, config_rng)
        for arm_id, config in configs.items():
            tasks_by_model.setdefault(config.model_id, []).append((s, arm_id, config))

    arm_results: list = []
    out_path = _phase1_path(tag)
    t0 = time.time()

    try:
        for model_id, work in tasks_by_model.items():
            llm, load_t = load_model(model_id)
            print(f"[phase1:{tag}] loaded {model_id} in {load_t:.1f}s -- {len(work)} generations queued")
            for s, arm_id, config in work:
                res = run_arm(s, config, llm)
                arm_results.append(res)
            del llm
            _atomic_write_json(
                out_path,
                {"experiment_id": EXPERIMENT_ID, "tag": tag, "n_states": len(states), "complete": False, "arm_results": arm_results},
            )
            print(f"[phase1:{tag}] checkpoint -- {len(arm_results)} arm results so far")
    except LeakDetected as exc:
        _atomic_write_json(
            out_path,
            {"experiment_id": EXPERIMENT_ID, "tag": tag, "n_states": len(states), "complete": False,
             "stopped_on_leak": str(exc), "arm_results": arm_results},
        )
        print(f"[phase1:{tag}] STOPPED -- antileak check caught a violation: {exc}")
        print("Per protocol (TASK_EXPERIMENT12.md §13): fix the defect, then start a NEW campaign with a NEW experiment_id -- do not patch over this data.")
        raise SystemExit(1)

    elapsed = time.time() - t0
    _atomic_write_json(
        out_path,
        {"experiment_id": EXPERIMENT_ID, "tag": tag, "n_states": len(states), "complete": True,
         "elapsed_sec": elapsed, "arm_results": arm_results},
    )
    print(f"[phase1:{tag}] done in {elapsed:.1f}s -- {len(arm_results)} arm results -> {out_path}")


# -- analyze ------------------------------------------------------------------


def _paired(by_collision: dict, arm_a: str, arm_b: str):
    ids = [cid for cid, arms in by_collision.items() if arm_a in arms and arm_b in arms]
    return ids, [by_collision[c][arm_a] for c in ids], [by_collision[c][arm_b] for c in ids]


def _total_and_confirmed(state: dict) -> tuple:
    seam = state["initial_seam_result"]
    key = "step_results" if state["task_family"] == "arithmetic" else "requirement_results"
    results = seam[key]
    total = len(results)
    confirmed = sum(1 for r in results if r["status"] == "PASS")
    return total, confirmed


def cmd_analyze(args) -> None:
    tag = "pilot" if args.pilot else "full"
    phase0, _ = _load_or_reuse_phase0(tag)
    phase1 = _load_json(_phase1_path(tag))
    if phase0 is None or phase1 is None:
        print(f"[analyze:{tag}] missing phase0/phase1 data -- run --phase 0 and --phase 1 first")
        return
    if not phase1.get("complete", False):
        print(f"[analyze:{tag}] WARNING: phase1 data is not marked complete (stopped early?) -- analyzing partial data anyway")

    states = phase0["states"]
    initial_records = phase0["initial_records"]
    arm_results = phase1["arm_results"]
    state_by_id = {s["collision_id"]: s for s in states}
    state_objs = [CollisionState(**s) for s in states]

    by_collision: dict = {}
    for r in arm_results:
        by_collision.setdefault(r["collision_id"], {})[r["arm"]] = r
    analyzed_ids = set(by_collision)

    # -- per-arm metrics (пятисоставный final_status: PASS/SAME_FAILURE/
    # DIFFERENT_FAILURE/INAPPLICABLE/UNVERIFIED, плюс ERROR отдельно) --
    arm_metrics = {}
    for arm in ALL_ARMS:
        attempts = [by_collision[cid][arm] for cid in by_collision if arm in by_collision[cid]]
        n = len(attempts)
        if n == 0:
            continue
        status_counts = {s: sum(1 for a in attempts if a.get("final_status") == s) for s in FINAL_STATUSES}
        n_solved = status_counts["PASS"]
        n_inapplicable = status_counts["INAPPLICABLE"]
        n_applicable = n - n_inapplicable
        n_fail_total = status_counts["SAME_FAILURE"] + status_counts["DIFFERENT_FAILURE"]
        n_with_regression = sum(1 for a in attempts if a.get("regressions"))
        n_with_improvement_not_pass = sum(1 for a in attempts if a.get("improvements") and a.get("final_status") != "PASS")
        total_tokens = sum(a.get("tokens", 0) or 0 for a in attempts)
        latencies = sorted(a["latency"] for a in attempts if a.get("latency") is not None)
        signatures = [tuple(a["config"].values()) for a in attempts if "config" in a]
        n_repeated_configs = len(signatures) - len(set(signatures)) if signatures else 0

        arm_metrics[arm] = {
            "n_attempts": n,
            "pass_rate": n_solved / n,
            "conditional_pass_rate": (n_solved / n_applicable) if n_applicable else None,
            "same_failure_rate": status_counts["SAME_FAILURE"] / n,
            "different_failure_rate": status_counts["DIFFERENT_FAILURE"] / n,
            "same_failure_rate_of_fails": (status_counts["SAME_FAILURE"] / n_fail_total) if n_fail_total else None,
            "inapplicable_rate": n_inapplicable / n,
            "unverified_rate": status_counts["UNVERIFIED"] / n,
            "error_rate": status_counts["ERROR"] / n,
            "regression_rate": n_with_regression / n,
            "improved_but_not_pass_rate": n_with_improvement_not_pass / n,
            "total_tokens": total_tokens,
            "tokens_per_resolved_task": (total_tokens / n_solved) if n_solved else None,
            "median_latency_sec": latencies[len(latencies) // 2] if latencies else None,
            "p90_latency_sec": latencies[int(0.9 * len(latencies))] if latencies else None,
            "n_repeated_identical_configurations": n_repeated_configs,
        }

    # -- economic control: each arm vs blind 1-for-1 resample baseline --
    blind = blind_resample_baseline(state_objs, initial_records)
    for arm, m in arm_metrics.items():
        if blind.get("tokens_per_resolved") and m.get("tokens_per_resolved_task"):
            m["beats_blind_resample"] = m["tokens_per_resolved_task"] < blind["tokens_per_resolved"]

    # -- primary tests (задание v2: DeltaK1 = P(PASS|K1,O0)-P(PASS|K0,O0);
    # DeltaK2 = P(PASS|K2,O0)-P(PASS|K1,O0), инкрементально к K1, не к K0;
    # K2-vs-K0 совокупно тоже считается отдельно для непрерывности с v1) --
    primary_tests = {}
    for level, control, label in (
        ("K1O0", "K0O0", "DeltaK1: K1O0 vs K0O0"),
        ("K2O0", "K1O0", "DeltaK2 (incremental): K2O0 vs K1O0"),
        ("K2O0", "K0O0", "K2O0 vs K0O0 (cumulative, continuity with v1)"),
    ):
        ids, a_att, b_att = _paired(by_collision, level, control)
        if not ids:
            continue
        a_solved = [a["solved"] for a in a_att]
        b_solved = [b["solved"] for b in b_att]
        sign_test = paired_sign_test(a_solved, b_solved)
        rate_diff = bootstrap_rate_diff(a_solved, b_solved)
        boot = bootstrap_ratio_diff(a_att, b_att)
        primary_tests[label] = {
            "n_paired": len(ids), "sign_test": sign_test,
            "pass_rate_diff": rate_diff, "bootstrap_token_cost_diff": boot,
        }

    # -- secondary: Delta_hetero(K) = P(PASS|K,O1)-P(PASS|K,O0), для K0/K1/K2 --
    secondary_tests = {}
    for level, label in (("K0O1", "Delta_hetero(K0): K0O1 vs K0O0 (CHANGE_MODEL at K0)"),
                          ("K1O1", "Delta_hetero(K1): K1O1 vs K1O0 (CHANGE_MODEL at K1)"),
                          ("K2O1", "Delta_hetero(K2): K2O1 vs K2O0 (CHANGE_MODEL at K2)")):
        control = level[:2] + "O0"
        ids, a_att, b_att = _paired(by_collision, level, control)
        if not ids:
            continue
        a_solved = [a["solved"] for a in a_att]
        b_solved = [b["solved"] for b in b_att]
        sign_test = paired_sign_test(a_solved, b_solved)
        rate_diff = bootstrap_rate_diff(a_solved, b_solved)
        boot = bootstrap_ratio_diff(a_att, b_att)
        secondary_tests[label] = {
            "n_paired": len(ids), "sign_test": sign_test,
            "pass_rate_diff": rate_diff, "bootstrap_token_cost_diff": boot,
        }

    # -- interaction tests --
    interaction_tests = {}
    for level, label in (("K1", "interaction: does CHANGE_MODEL help more at K1 than at K0?"),
                          ("K2", "interaction: does CHANGE_MODEL help more at K2 than at K0?")):
        ids, k0o0, k0o1 = _paired(by_collision, "K0O0", "K0O1")
        ids2, kio0, kio1 = _paired(by_collision, f"{level}O0", f"{level}O1")
        common = [cid for cid in ids if cid in set(ids2)]
        if not common:
            continue
        k0o0_b = [by_collision[c]["K0O0"]["solved"] for c in common]
        k0o1_b = [by_collision[c]["K0O1"]["solved"] for c in common]
        kio0_b = [by_collision[c][f"{level}O0"]["solved"] for c in common]
        kio1_b = [by_collision[c][f"{level}O1"]["solved"] for c in common]
        interaction_tests[label] = bootstrap_interaction(k0o0_b, k0o1_b, kio0_b, kio1_b)
        interaction_tests[label]["n_paired"] = len(common)

    # -- задание v2: таксономия коллизий syntax/contract_interface/dependency/
    # semantic/value/other -- вычисляется по СОБСТВЕННОЙ первичной коллизии
    # состояния (не по исходу конкретного плеча). dependency/value применимы
    # только к арифметике (реальная цепочка зависимостей между шагами) --
    # для кода честно не подгоняются, см. seams.classify_code_subtype.
    def _state_failure_type(cid: str):
        state = state_by_id[cid]
        family = state["task_family"]
        seam = state["initial_seam_result"]
        task_steps = ARITH_TASKS[state["task_id"]]["steps"] if family == "arithmetic" else None
        return seams.collision_subtype(family, seam, task_steps)

    failure_type_by_id = {cid: _state_failure_type(cid) for cid in analyzed_ids}

    # -- H-syn: K0O0_promptB vs K0O0, syntax-типа состояния (задание v2) --
    syntactic_ids = [cid for cid in analyzed_ids if failure_type_by_id[cid] == "syntax"]
    hsyn_ids = [cid for cid in syntactic_ids if ARM_H_SYN in by_collision[cid] and "K0O0" in by_collision[cid]]
    h_syn_test = None
    if hsyn_ids:
        a_att = [by_collision[c][ARM_H_SYN] for c in hsyn_ids]
        b_att = [by_collision[c]["K0O0"] for c in hsyn_ids]
        h_syn_test = {
            "n_syntactic_paired": len(hsyn_ids),
            "meets_preregistered_n_threshold": len(hsyn_ids) >= MIN_HSYN_N,
            "sign_test": paired_sign_test([a["solved"] for a in a_att], [b["solved"] for b in b_att]),
            "bootstrap_token_cost_diff": bootstrap_ratio_diff(a_att, b_att),
        }

    # -- §8: applicability of structural context --
    buckets = {"none": [], "partial": [], "near_complete": []}
    for cid in analyzed_ids:
        total, confirmed = _total_and_confirmed(state_by_id[cid])
        if confirmed == 0:
            buckets["none"].append(cid)
        elif confirmed >= total - 1:
            buckets["near_complete"].append(cid)
        else:
            buckets["partial"].append(cid)

    def _rate(cids, arm):
        attempts = [by_collision[cid][arm] for cid in cids if arm in by_collision.get(cid, {})]
        if not attempts:
            return None
        return sum(1 for a in attempts if a["solved"]) / len(attempts)

    applicability = {
        "n_total_analyzed": len(analyzed_ids),
        "bucket_sizes": {k: len(v) for k, v in buckets.items()},
        "p_has_any_confirmed_step": (len(buckets["partial"]) + len(buckets["near_complete"])) / len(analyzed_ids) if analyzed_ids else None,
        "k2o0_resolution_rate": {k: _rate(v, "K2O0") for k, v in buckets.items()},
        "k0o0_resolution_rate": {k: _rate(v, "K0O0") for k, v in buckets.items()},
    }
    gain_none = None
    gain_partial = None
    if applicability["k2o0_resolution_rate"]["none"] is not None and applicability["k0o0_resolution_rate"]["none"] is not None:
        gain_none = applicability["k2o0_resolution_rate"]["none"] - applicability["k0o0_resolution_rate"]["none"]
    if applicability["k2o0_resolution_rate"]["partial"] is not None and applicability["k0o0_resolution_rate"]["partial"] is not None:
        gain_partial = applicability["k2o0_resolution_rate"]["partial"] - applicability["k0o0_resolution_rate"]["partial"]
    applicability["k2_vs_k0_gain_when_no_confirmed_step"] = gain_none
    applicability["k2_vs_k0_gain_when_partial_confirmed"] = gain_partial
    applicability["note_near_complete_excluded_from_headline"] = (
        "near_complete (all steps confirmed except one) is reported separately per TASK_EXPERIMENT12.md §8 "
        "-- close to handing over the answer, excluded from the main P(applicable) x gain product"
    )
    if gain_partial is not None:
        p_partial = len(buckets["partial"]) / len(analyzed_ids) if analyzed_ids else 0
        applicability["expected_gain_product_partial_only"] = p_partial * gain_partial

    # -- breakdown by collision subtype (задание v2, n>=MIN_CATEGORY_N) --
    by_type: dict = {}
    for cid in analyzed_ids:
        by_type.setdefault(failure_type_by_id[cid] or "other", []).append(cid)

    breakdown = {}
    for ctype, cids in by_type.items():
        if len(cids) < MIN_CATEGORY_N:
            breakdown[ctype] = {"n": len(cids), "note": f"insufficient data, no conclusion drawn (n < {MIN_CATEGORY_N})"}
            continue
        per_arm = {}
        for arm in ARMS_MAIN:
            attempts = [by_collision[cid][arm] for cid in cids if arm in by_collision.get(cid, {})]
            if not attempts:
                continue
            n_solved = sum(1 for a in attempts if a.get("solved"))
            n_same = sum(1 for a in attempts if a.get("final_status") == "SAME_FAILURE")
            n_diff = sum(1 for a in attempts if a.get("final_status") == "DIFFERENT_FAILURE")
            per_arm[arm] = {
                "n": len(attempts), "pass_rate": n_solved / len(attempts),
                "same_failure_rate": n_same / len(attempts), "different_failure_rate": n_diff / len(attempts),
            }
        breakdown[ctype] = {"n": len(cids), "per_arm": per_arm}
    breakdown["_note"] = "dependency/value applicable only to arithmetic (real step-chain); code collisions map to syntax/contract_interface/semantic/other (TASK_EXPERIMENT12.md v2, taxonomy section)"

    # -- best-of-N (same combinatorial method as Experiment 11) --
    import itertools

    by_key: dict = {}
    for r in initial_records:
        key = (r["model"], r["seam_type"], r["task_id"])
        by_key.setdefault(key, []).append(r)
    best_of_n = {}
    for n in (1, 2, 3, 4):
        oracle_ok = cnt = 0
        token_sum = 0
        for samples in by_key.values():
            if len(samples) < n:
                continue
            for combo in itertools.combinations(samples, n):
                cnt += 1
                token_sum += sum(s["total_tokens"] for s in combo)
                if any(s["status"] == "PASS" for s in combo):
                    oracle_ok += 1
        if cnt:
            best_of_n[f"N={n}"] = {
                "n_combinations": cnt, "mean_tokens": token_sum / cnt, "oracle_success_rate": oracle_ok / cnt,
                "oracle_tokens_per_success": (token_sum / cnt) / (oracle_ok / cnt) if oracle_ok else None,
            }

    n_arith_found = sum(1 for s in states if s["task_family"] == "arithmetic")
    n_code_found = sum(1 for s in states if s["task_family"] == "code")
    n_arith_analyzed = sum(1 for cid in analyzed_ids if state_by_id[cid]["task_family"] == "arithmetic")
    n_code_analyzed = sum(1 for cid in analyzed_ids if state_by_id[cid]["task_family"] == "code")

    # -- финальная сверка sanity-чеклиста на ВСЕЙ собранной кампании (не
    # только на пилоте) -- задание v2: "если хотя бы один пункт
    # нарушается -- не запускать основной эксперимент"; здесь это
    # постфактум-подтверждение, что ни один пункт не нарушился по факту --
    sanity = run_sanity_checklist(state_objs, arm_results)

    result = {
        "experiment_id": EXPERIMENT_ID, "tag": tag,
        "n_initial_attempts": len(initial_records),
        "n_collision_states_found": len(states),
        "n_collision_states_analyzed": len(analyzed_ids),
        "n_arithmetic_states_found": n_arith_found, "n_code_states_found": n_code_found,
        "n_arithmetic_states_analyzed": n_arith_analyzed, "n_code_states_analyzed": n_code_analyzed,
        "n_syntactic_states_analyzed": len(syntactic_ids),
        "sanity_checklist": sanity,
        "blind_resample_baseline": blind,
        "arm_metrics": arm_metrics,
        "primary_tests": primary_tests,
        "secondary_tests": secondary_tests,
        "interaction_tests": interaction_tests,
        "h_syn_test": h_syn_test,
        "applicability_of_structural_context": applicability,
        "breakdown_by_collision_type": breakdown,
        "best_of_n": best_of_n,
    }
    if not sanity["overall_ok"]:
        print(f"[analyze:{tag}] WARNING: sanity checklist failed on collected data: {sanity}")

    out_path = METRICS_DIR / f"action_outcomes_{tag}.json"
    _atomic_write_json(out_path, result)
    print(f"[analyze:{tag}] saved -> {out_path}")
    print(json.dumps({k: v for k, v in result.items() if k not in ("arm_metrics", "breakdown_by_collision_type")}, ensure_ascii=False, indent=2, default=str))
    print("arm_metrics:")
    print(json.dumps(arm_metrics, ensure_ascii=False, indent=2, default=str))
    print("breakdown_by_collision_type:")
    print(json.dumps(breakdown, ensure_ascii=False, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 12 CLI")
    parser.add_argument("--phase", choices=["0", "1", "analyze"], required=True)
    parser.add_argument("--pilot", action="store_true", help="small pilot run")
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
