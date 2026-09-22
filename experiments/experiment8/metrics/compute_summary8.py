"""
PHASE 0-3 aggregation over analysis8_records.json. Pure aggregation, no
new classification logic -- everything here just summarizes what
build_analysis8.py already computed with the frozen protocol.
"""
import collections
import json
import os
import statistics

METRICS_DIR = os.path.dirname(__file__)


def load_records():
    with open(os.path.join(METRICS_DIR, "analysis8_records.json"), encoding="utf-8") as f:
        return json.load(f)


def phase0_decomposition_quality(d_records):
    claim_statuses = []
    section_word_counts = collections.defaultdict(list)
    n_artifacts_by_family = {}
    n_claims_by_family = {}
    for r in d_records:
        a = r["analysis"]
        n_artifacts_by_family[a["family"]] = a["n_artifacts"]
        n_claims_by_family[a["family"]] = a["n_claims"]
        for cid, status in a["local_status"].items():
            claim_statuses.append(status)
        for sec, text in a["sections"].items():
            if text:
                section_word_counts[sec].append(len(text.split()))

    counts = collections.Counter(claim_statuses)
    total = sum(counts.values())
    return {
        "n_artifacts_by_family": n_artifacts_by_family, "n_claims_by_family": n_claims_by_family,
        "n_claim_observations": total, "raw_status_distribution": dict(counts),
        "unknown_rate": round((counts.get("OMITTED", 0) + counts.get("UNKNOWN", 0)) / total, 4) if total else None,
        "ambiguous_rate": round(counts.get("AMBIGUOUS", 0) / total, 4) if total else None,
        "correct_rate": round(counts.get("CORRECT", 0) / total, 4) if total else None,
        "incorrect_rate": round(counts.get("INCORRECT", 0) / total, 4) if total else None,
        "avg_section_word_count": {k: round(statistics.mean(v), 1) for k, v in section_word_counts.items()},
    }


def phase1_stability(d_records):
    by_unit = collections.defaultdict(list)
    for r in d_records:
        by_unit[(r["task_id"], r["model_id"])].append(r["analysis"]["parse_quality"])

    unit_full_rate = {}
    for unit, qualities in by_unit.items():
        unit_full_rate[f"{unit[0]}/{unit[1]}"] = round(sum(1 for q in qualities if q == "FULL") / len(qualities), 4)

    all_qualities = [r["analysis"]["parse_quality"] for r in d_records]
    overall = collections.Counter(all_qualities)
    total = len(all_qualities)
    return {
        "overall_parse_quality": dict(overall), "overall_full_rate": round(overall.get("FULL", 0) / total, 4) if total else None,
        "unit_full_rate": unit_full_rate,
        "units_below_full_100pct": [u for u, r in unit_full_rate.items() if r < 1.0],
    }


def phase2_p_vs_d(all_records):
    p_records = [r for r in all_records if r["mode"] == "P"]
    d_records = [r for r in all_records if r["mode"] == "D"]

    def per_unit_oracle(records):
        by_unit = collections.defaultdict(list)
        for r in records:
            by_unit[(r["task_id"], r["model_id"])].append(r["analysis"]["task_status"])
        out = {}
        for unit, statuses in by_unit.items():
            n = len(statuses)
            out[f"{unit[0]}/{unit[1]}"] = {
                "n": n, "correct_rate": round(statuses.count("CORRECT") / n, 4),
                "oracle_at_n": 1.0 if "CORRECT" in statuses else 0.0,
                "unknown_rate": round(statuses.count("UNKNOWN") / n, 4),
            }
        return out

    p_by_unit = per_unit_oracle(p_records)
    d_by_unit = per_unit_oracle(d_records)

    p_tokens = [r["output_tokens"] for r in p_records]
    d_tokens = [r["output_tokens"] for r in d_records]
    p_time = [r["generation_time_sec"] for r in p_records]
    d_time = [r["generation_time_sec"] for r in d_records]

    return {
        "p_by_unit": p_by_unit, "d_by_unit": d_by_unit,
        "overall_p_correct_rate": round(sum(1 for r in p_records if r["analysis"]["task_status"] == "CORRECT") / len(p_records), 4),
        "overall_d_correct_rate": round(sum(1 for r in d_records if r["analysis"]["task_status"] == "CORRECT") / len(d_records), 4),
        "overall_p_oracle_at_8": round(sum(v["oracle_at_n"] for v in p_by_unit.values()) / len(p_by_unit), 4),
        "overall_d_oracle_at_8": round(sum(v["oracle_at_n"] for v in d_by_unit.values()) / len(d_by_unit), 4),
        "tokens_flat_mean": round(statistics.mean(p_tokens), 1), "tokens_decomposed_mean": round(statistics.mean(d_tokens), 1),
        "time_flat_mean": round(statistics.mean(p_time), 3), "time_decomposed_mean": round(statistics.mean(d_time), 3),
        "tokens_flat_total": sum(p_tokens), "tokens_decomposed_total": sum(d_tokens),
    }


def phase3_localization(d_records):
    incorrect = [r for r in d_records if r["analysis"]["task_status"] == "INCORRECT"]
    mrs_fractions = []
    origin_claim_types = collections.Counter()
    seam_consistency_violations = 0
    seam_consistency_checked = 0
    error_local_count = 0
    error_dependency_count = 0
    unknown_count = 0
    no_error_count = 0

    for r in d_records:
        a = r["analysis"]
        for label in a["error_labels"].values():
            if label == "ERROR_LOCAL":
                error_local_count += 1
            elif label == "ERROR_DEPENDENCY":
                error_dependency_count += 1
            elif label == "UNKNOWN":
                unknown_count += 1
            else:
                no_error_count += 1
        sc = a["seam_consistency"]
        if sc["status"] != "NOT_APPLICABLE":
            seam_consistency_checked += 1
            if sc["status"] == "VIOLATED":
                seam_consistency_violations += 1

    for r in incorrect:
        a = r["analysis"]
        if a["n_claims"]:
            mrs_fractions.append(len(a["mrs"]) / a["n_claims"])
        for cid in a["root_origins"]:
            origin_claim_types[cid] += 1

    return {
        "n_incorrect_d_generations": len(incorrect),
        "mrs_fraction_of_task_mean": round(statistics.mean(mrs_fractions), 4) if mrs_fractions else None,
        "mrs_fraction_of_task_median": round(statistics.median(mrs_fractions), 4) if mrs_fractions else None,
        "mrs_never_equals_full_task": all(f < 1.0 for f in mrs_fractions) if mrs_fractions else None,
        "origin_claim_type_distribution": dict(origin_claim_types),
        "seam_consistency_violation_rate": round(seam_consistency_violations / seam_consistency_checked, 4) if seam_consistency_checked else None,
        "seam_consistency_checked": seam_consistency_checked, "seam_consistency_violations": seam_consistency_violations,
        "claim_level_error_taxonomy": {"ERROR_LOCAL": error_local_count, "ERROR_DEPENDENCY": error_dependency_count,
                                        "UNKNOWN": unknown_count, "NO_ERROR": no_error_count},
    }


def main():
    all_records = load_records()
    d_records = [r for r in all_records if r["mode"] == "D"]

    summary = {
        "phase0_decomposition_quality": phase0_decomposition_quality(d_records),
        "phase1_stability": phase1_stability(d_records),
        "phase2_p_vs_d": phase2_p_vs_d(all_records),
        "phase3_localization": phase3_localization(d_records),
    }

    with open(os.path.join(METRICS_DIR, "summary8.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
