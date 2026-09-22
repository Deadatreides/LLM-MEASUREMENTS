"""run_e4.py — E4: structural decoding instead of flat conjunction.
0 NEW GPU CALLS -- replays metrics/filter_calls.jsonl (516 rows already
on disk from E1/E2). Thresholds pre-registered in PROTOCOL_E4.md SS4,
never moved.

STRUCTURE_DET=true on every E4 number (PROTOCOL_E4.md SS5): class
enumeration and the aggregate/derive expansion are deterministic. The
pure-tool ceiling (filter_det=1.000, FORK-1) is reported alongside so E4
is never mistaken for a swarm beating a tool.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import task_generator_f2 as TF2       # noqa: E402
import task_generator_hard as THARD   # noqa: E402
import consensus_filter as CF         # noqa: E402
import structure_decode as SD         # noqa: E402
import det_atoms as DA                # noqa: E402
import oracles as OR                  # noqa: E402
import model_registry_11 as MR        # noqa: E402

AGENT_DIR = Path(__file__).resolve().parents[1]
METRICS_DIR = AGENT_DIR / "metrics"

# read-only reference constants (DELTA-0 / FORK-1 / E1 / E2)
R_M_STAR_F2 = 0.525        # DELTA-0 whole-task best single
R_UNION_LLM_F2 = 0.025     # DELTA-0 chained union
R_FLAT_E1 = 0.025          # E1 flat consensus exact-set
R_M_STAR_B = 0.000         # FORK-1 T_hard whole-task, all 6 models
R_FLAT_E2 = 0.025          # E2 flat consensus + det-SUM
FILTER_DET = 1.000         # FORK-1 pure deterministic ceiling

# pre-registered thresholds (PROTOCOL_E4.md SS4)
THR_E4A_SET_PREDICT, THR_E4A_SET_FALSIFY = 0.30, 0.15
THR_DELTA = 0.05
THR_E4B_PREDICT = 0.25


def load_votes(family: str, task_ids: list, model_ids) -> dict:
    path = METRICS_DIR / "filter_calls.jsonl"
    votes = {t: {} for t in task_ids}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["family"] != family or row["task_id"] not in votes:
            continue
        votes[row["task_id"]][row["model"]] = {"extracted": frozenset(row["extracted"])}
    missing = [(t, m) for t in task_ids for m in model_ids if m not in votes[t]]
    if missing:
        raise RuntimeError(f"HARD_STOP: filter_calls.jsonl missing {len(missing)} rows "
                           f"for family={family} -- E4 never re-collects (PROTOCOL_E4.md SS6)")
    return votes


def run_family(family: str, gen_mod, model_ids, final_fn) -> dict:
    d = gen_mod.build_tasks()
    tasks_by_id, test_ids = d["tasks"], d["test"]
    votes = load_votes(family, test_ids, model_ids)

    n = len(test_ids)
    stats = {
        "swarm_flat_set": 0, "swarm_dec_set": 0, "swarm_dec_final": 0,
        "single_honest_dec_set": 0, "single_honest_dec_final": 0,
    }
    per_model_dec_set = {m: 0 for m in model_ids}
    per_model_flat_set = {m: 0 for m in model_ids}
    per_task = {}
    class_counts = []

    for tid in test_ids:
        task = tasks_by_id[tid]
        golden = frozenset(task["matched_ids"])
        cids = CF.candidate_ids(task)
        classes = SD.enumerate_classes(task)
        class_counts.append(len(classes))

        loo_rel = CF.loo_reliabilities(tasks_by_id, test_ids, model_ids, votes, tid)
        admitted = sorted(m for m in model_ids if loo_rel[m] > 0.5)

        # --- swarm: flat (E1/E2 replay) vs decoded ---
        flat_set = CF.consensus_ids_for_task(task, votes[tid], admitted, loo_rel)
        stats["swarm_flat_set"] += int(frozenset(flat_set) == golden)

        support = SD.swarm_support(task, votes[tid], admitted)
        _key, dec_set, _score = SD.decode(support, cids, classes)
        stats["swarm_dec_set"] += int(dec_set == golden)
        swarm_final_ok = final_fn(task, dec_set)
        stats["swarm_dec_final"] += int(swarm_final_ok)

        # --- honest best single (LOO-picked, never sees this task's label) ---
        best_m = max(sorted(model_ids), key=lambda m: loo_rel[m])
        s_sup = SD.single_support(task, votes[tid], best_m)
        _k2, s_dec, _s2 = SD.decode(s_sup, cids, classes)
        stats["single_honest_dec_set"] += int(s_dec == golden)
        stats["single_honest_dec_final"] += int(final_fn(task, s_dec))

        # --- every model, decoded + flat (so the oracle max is visible) ---
        for m in model_ids:
            ms = SD.single_support(task, votes[tid], m)
            _k3, m_dec, _s3 = SD.decode(ms, cids, classes)
            per_model_dec_set[m] += int(m_dec == golden)
            per_model_flat_set[m] += int(votes[tid][m]["extracted"] == golden)

        per_task[tid] = {
            "golden": sorted(golden), "decoded": sorted(dec_set),
            "flat": sorted(flat_set), "n_classes": len(classes),
            "admitted": admitted, "best_single": best_m,
            "final_ok": bool(swarm_final_ok),
        }

    out = {k: v / n for k, v in stats.items()}
    out["n"] = n
    out["mean_n_classes"] = sum(class_counts) / n
    out["per_model_dec_set"] = {m: c / n for m, c in per_model_dec_set.items()}
    out["per_model_flat_set"] = {m: c / n for m, c in per_model_flat_set.items()}
    out["oracle_best_model_dec_set"] = max(out["per_model_dec_set"].values())
    out["per_task"] = per_task
    return out


def final_f2(task: dict, ids) -> bool:
    agg = DA.aggregate_det(ids, task["op"], task["id_to_amount"])
    return DA.derive_det(agg, task["threshold"], task["comparator"]) == task["final_oracle"]


def final_hard(task: dict, ids) -> bool:
    return OR.within_tolerance(DA.aggregate_det(ids, "SUM", task["id_to_amount"]),
                               task["final_oracle"])


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    model_ids = list(MR.MODEL_IDS)

    f2 = run_family("E1-F2", TF2, model_ids, final_f2)
    hard = run_family("E2-Thard", THARD, model_ids, final_hard)

    # ---- pre-registered verdicts (PROTOCOL_E4.md SS4), thresholds untouched ----
    e4a_set = f2["swarm_dec_set"]
    e4a_fin = f2["swarm_dec_final"]
    delta_f2 = e4a_fin - R_M_STAR_F2
    e4b = hard["swarm_dec_final"]
    delta_hard = e4b - R_M_STAR_B
    gain_f2 = e4a_set - f2["single_honest_dec_set"]
    gain_hard = hard["swarm_dec_set"] - hard["single_honest_dec_set"]

    v_e4a_set = ("FALSIFIED" if e4a_set < THR_E4A_SET_FALSIFY
                 else "CONFIRMED" if e4a_set >= THR_E4A_SET_PREDICT
                 else "ABOVE_FLOOR_BELOW_PREDICTION")
    v_e4a_fin = "DELTA_POSITIVE" if delta_f2 >= THR_DELTA else "DELTA_NONPOS"
    v_e4b = ("CONFIRMED" if delta_hard >= THR_E4B_PREDICT
             else "ABOVE_FLOOR_BELOW_PREDICTION" if delta_hard >= THR_DELTA
             else "FALSIFIED")
    v_gain = "SWARM_NULL" if gain_f2 <= 0 else "SWARM_POSITIVE"

    result = {
        "STRUCTURE_DET": True,
        "new_gpu_calls": 0,
        "tool_ceiling_filter_det": FILTER_DET,
        "reference": {
            "r_m_star_F2": R_M_STAR_F2, "r_union_llm_F2": R_UNION_LLM_F2,
            "r_flat_E1": R_FLAT_E1, "r_m_star_B": R_M_STAR_B, "r_flat_E2": R_FLAT_E2,
        },
        "thresholds": {
            "E4a_set_predict": THR_E4A_SET_PREDICT, "E4a_set_falsify": THR_E4A_SET_FALSIFY,
            "delta": THR_DELTA, "E4b_predict": THR_E4B_PREDICT,
        },
        "F2": f2, "T_hard": hard,
        "verdicts": {
            "E4a_set": v_e4a_set, "E4a_final": v_e4a_fin, "E4b": v_e4b, "E4_gain": v_gain,
        },
        "delta_F2_final": delta_f2, "delta_Thard": delta_hard,
        "gain_set_F2": gain_f2, "gain_set_Thard": gain_hard,
        "fork1_basket_flips_to_FORK_LLM_SWARM": delta_hard >= 0.05,
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "e4_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    p = print
    p("=== E4: structural decoding (STRUCTURE_DET=true, 0 new GPU calls) ===\n")
    p(f"mean classes per task: F2={f2['mean_n_classes']:.1f}  T_hard={hard['mean_n_classes']:.1f}")
    p(f"                       (vs K=18 / K=14 flat bits -- this is the dimensionality drop)\n")

    p("--- F2 (DELTA-0 polygon) ---")
    p(f"  exact-set  flat consensus (E1 replay) : {f2['swarm_flat_set']:.3f}")
    p(f"  exact-set  swarm + DECODE             : {e4a_set:.3f}   [{v_e4a_set}]")
    p(f"  exact-set  best single (honest) + DEC : {f2['single_honest_dec_set']:.3f}")
    p(f"  exact-set  best model  (oracle) + DEC : {f2['oracle_best_model_dec_set']:.3f}")
    p(f"  GAIN swarm - best single (honest)     : {gain_f2:+.3f}   [{v_gain}]")
    p(f"  FINAL yes/no  swarm + DECODE          : {e4a_fin:.3f}")
    p(f"  Delta vs r_m*_F2={R_M_STAR_F2:.3f}            : {delta_f2:+.3f}   [{v_e4a_fin}]\n")

    p("--- T_hard (FORK-1 polygon) ---")
    p(f"  exact-set  flat consensus (E2 replay) : {hard['swarm_flat_set']:.3f}")
    p(f"  exact-set  swarm + DECODE             : {hard['swarm_dec_set']:.3f}")
    p(f"  exact-set  best single (honest) + DEC : {hard['single_honest_dec_set']:.3f}")
    p(f"  GAIN swarm - best single (honest)     : {gain_hard:+.3f}")
    p(f"  SUM r      swarm + DECODE             : {e4b:.3f}")
    p(f"  Delta vs r_m*_B={R_M_STAR_B:.3f}              : {delta_hard:+.3f}   [{v_e4b}]")
    p(f"  FORK-1 basket flips to FORK_LLM_SWARM : {delta_hard >= 0.05}\n")

    p("--- per-model exact-set, flat -> decoded (F2) ---")
    for m in model_ids:
        p(f"  {m:35s} {f2['per_model_flat_set'][m]:.3f} -> {f2['per_model_dec_set'][m]:.3f}")
    p(f"\n  tool ceiling filter_det (FORK-1)      : {FILTER_DET:.3f}  "
      "<- E4 does NOT beat this and does not claim to")


if __name__ == "__main__":
    main()
