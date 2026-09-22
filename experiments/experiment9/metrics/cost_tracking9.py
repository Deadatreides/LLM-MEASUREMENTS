"""
Section 22: cost is tracked per action type, not assumed free for
anything (explicitly including graph analysis itself). This experiment's
actual cost ledger:

  initial_generation_cost : real GPU generation, runs/d_index9.*.jsonl
                             (output_tokens, generation_time_sec per run)
  detection_cost           : classification (regex over already-generated
                              text) -- zero GPU tokens, but not zero wall
                              time; measured here as wall-clock seconds
                              for the classification+propagation pass
  graph_analysis_cost      : propagation + affected-set/MRS computation --
                              also zero GPU tokens, measured as wall-clock
                              seconds (brute-force MRS is the expensive
                              part: 2^6 subset evaluations per case)
  repair_cost               : not separately re-generated with a live model
                              in this experiment (section 28 explicitly
                              excludes building a retry/repair mechanism
                              here -- experiment 8 already measured real
                              repair-generation cost; this experiment
                              reuses that finding rather than re-spending
                              GPU budget on it) -- reported as a TOKEN
                              ESTIMATE only (proportional to number of
                              artifacts in predicted_affected_set vs all 6),
                              explicitly marked as an estimate, not measured
  full_retry_cost_estimate : token estimate for regenerating all 6 sections
"""
import json
import os
import time
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "configs"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tasks"))

METRICS_DIR = os.path.dirname(__file__)


def measure_detection_and_graph_cost():
    """Re-runs the mechanical pipeline (classification + propagation +
    brute-force MRS) with wall-clock timers, to report these as non-zero,
    real (if small) costs rather than silently treating them as free."""
    t0 = time.perf_counter()
    import case_abcd_evaluation9
    case_abcd_evaluation9.main()
    t1 = time.perf_counter()
    detection_and_propagation_sec = t1 - t0

    t2 = time.perf_counter()
    import importlib
    import mrs_bruteforce9
    importlib.reload(mrs_bruteforce9)
    mrs_bruteforce9.main()
    t3 = time.perf_counter()
    bruteforce_mrs_sec = t3 - t2

    return {
        "detection_and_propagation_sec_for_120_cases": round(detection_and_propagation_sec, 4),
        "detection_and_propagation_sec_per_case": round(detection_and_propagation_sec / 120, 6),
        "bruteforce_mrs_sec_for_120_cases": round(bruteforce_mrs_sec, 4),
        "bruteforce_mrs_sec_per_case": round(bruteforce_mrs_sec / 120, 6),
        "note": "both are pure-Python, zero GPU tokens; brute-force MRS is the more expensive of the two "
                "(64 subset evaluations per case) but still sub-second per case at this DAG size (2^6=64) -- "
                "would NOT stay this cheap at larger claim counts (2^n growth), a real scaling limit for section 17's "
                "approach, noted honestly rather than glossed over.",
    }


def repair_token_estimates():
    """Coarse token estimate (not measured): assumes each artifact costs
    roughly output_tokens/6 of a full D-mode generation (from
    analysis9_records.json's real average, if available)."""
    records_path = os.path.join(METRICS_DIR, "analysis9_records.json")
    if not os.path.exists(records_path):
        return {"note": "analysis9_records.json not yet built -- run build_analysis9.py first"}
    records = json.load(open(records_path, encoding="utf-8"))
    if not records:
        return {"note": "no records"}
    avg_full_tokens = sum(r["output_tokens"] for r in records) / len(records)
    per_artifact_tokens = avg_full_tokens / 6

    case_results = json.load(open(os.path.join(METRICS_DIR, "case_abcd_results9.json"), encoding="utf-8"))
    mrs_results = json.load(open(os.path.join(METRICS_DIR, "mrs_bruteforce9_results.json"), encoding="utf-8"))
    mrs_by_key = {(m["task_id"], m["injected_claim"]): m for m in mrs_results}

    full_repair_tokens = avg_full_tokens
    graph_repair_tokens = []
    mrs_repair_tokens = []
    for r in case_results:
        key = (r["task_id"], r["injected_claim"])
        graph_repair_tokens.append(len(r["predicted_affected_set"]) * per_artifact_tokens)
        mrs_repair_tokens.append(len(mrs_by_key[key]["mrs_ground_truth_candidates"][0]) * per_artifact_tokens)

    import statistics
    return {
        "avg_full_generation_tokens_real": round(avg_full_tokens, 1),
        "per_artifact_token_estimate": round(per_artifact_tokens, 1),
        "full_repair_tokens_estimate": round(full_repair_tokens, 1),
        "graph_repair_tokens_estimate_mean": round(statistics.mean(graph_repair_tokens), 1),
        "mrs_repair_tokens_estimate_mean": round(statistics.mean(mrs_repair_tokens), 1),
        "compute_saving_graph_repair_vs_full": round(1 - statistics.mean(graph_repair_tokens) / full_repair_tokens, 4),
        "compute_saving_mrs_repair_vs_full": round(1 - statistics.mean(mrs_repair_tokens) / full_repair_tokens, 4),
        "note": "TOKEN ESTIMATE ONLY -- proportional to artifact count, not independently measured by live "
                "regeneration (section 28 explicitly excludes building a new repair mechanism in this experiment; "
                "experiment 8 already measured REAL repair-generation cost/reliability for a comparable task family).",
    }


def main():
    out = {"detection_and_graph_analysis_cost": measure_detection_and_graph_cost(), "repair_token_estimates": repair_token_estimates()}
    with open(os.path.join(METRICS_DIR, "cost_tracking9_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
