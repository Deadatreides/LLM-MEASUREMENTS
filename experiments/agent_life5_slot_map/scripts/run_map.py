"""run_map.py — Phase B: mandatory 24 archives (life3+life4) -> per-slot
map; auto-detects and separately maps the Phase B.2 `life5_default_s*`
runs if they exist on disk (never blended into the mandatory pool -- see
`slot_map.py` module docstring). Writes `metrics/slot_map.json` (headline
summary) + `metrics/per_slot_signatures.json` (full per-signature detail).
Pure CPU/JSON -- reads the grid (already-copied JSON, no new generate())
only to recompute Imp/coverage.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
AGENT_DIR = SCRIPTS.parent
SRC = AGENT_DIR / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import live_dataset as LD        # noqa: E402
import unfold_metrics as ML      # noqa: E402
import slot_map as SM            # noqa: E402
import orchestrator as O         # noqa: E402  -- only for build_panel

METRICS_DIR = AGENT_DIR / "metrics"
RUNS_DIR = AGENT_DIR / "runs"


def _serialize_signatures(sigs: dict) -> list:
    out = []
    for (op, models), bucket in sigs.items():
        lifespans = bucket["lifespans"]
        out.append({
            "op": op, "models": list(models), "imp": bucket["imp"],
            "n_carriers": len(bucket["carriers"]), "n_seed_appeared": len(bucket["seeds"]),
            "seeds": sorted(bucket["seeds"]),
            "median_L": statistics.median(lifespans) if lifespans else None,
            "max_L": max(lifespans) if lifespans else None,
            "n_occurrences": len(lifespans),
        })
    out.sort(key=lambda d: (-d["n_carriers"], -d["n_seed_appeared"]))
    return out


def _serialize_agg(agg: dict) -> dict:
    return {str(s): {
        "n_distinct_signatures": agg[s]["n_distinct_signatures"],
        "n_signatures_with_imp": agg[s]["n_signatures_with_imp"],
        "signatures": _serialize_signatures(agg[s]["signatures"]),
    } for s in agg}


def build_map_for(runs: list, ds, panel: list, best_single: dict, label: str) -> dict:
    full = SM.build_full_map(runs, ds, panel, best_single)
    return {
        "label": label, "n_runs": full["n_runs"], "run_ids": full["run_ids"],
        "format_sanity_ok": full["format_sanity_ok"],
        "format_sanity_violations": full["format_sanity_violations"],
        "delta_tally_pooled": full["delta_tally_pooled"],
        "agg_gen0": _serialize_agg(full["agg_gen0"]),
        "agg_gen1": _serialize_agg(full["agg_gen1"]),
    }


def _headline(m: dict) -> dict:
    """`top5` — top 5 by n_carriers AMONG Imp=1 signatures only (matches
    the report's own tables). `_serialize_signatures` sorts ALL
    signatures (Imp true or false) by n_carriers, so naively slicing
    `[:5]` here would let a huge-carrier-count Imp=0 signature (e.g. a
    single model that just matches, not beats, `best_single` on FORMAT)
    crowd out the actually-interesting Imp=1 entries — filter first."""
    return {
        "label": m["label"], "n_runs": m["n_runs"],
        "format_sanity_ok": m["format_sanity_ok"],
        "delta_tally_pooled": m["delta_tally_pooled"],
        "per_slot_gen1_headline": {
            s: {"n_distinct_signatures": v["n_distinct_signatures"],
               "n_signatures_with_imp": v["n_signatures_with_imp"],
               "top5_imp1": [sig for sig in v["signatures"] if sig["imp"]][:5]}
            for s, v in m["agg_gen1"].items()
        },
    }


def run_phase_b() -> dict:
    ds = LD.default_dataset()
    panel_info = O.build_panel(ds)
    panel = panel_info["panel"]
    best_single = ML.best_single_per_slot(ds, panel)

    mandatory_runs = SM.discover_mandatory_runs()
    print(f"[run_map] mandatory runs discovered: {len(mandatory_runs)}", flush=True)
    assert len(mandatory_runs) == 24, f"expected 24 mandatory runs, found {len(mandatory_runs)}"

    mandatory_map = build_map_for(mandatory_runs, ds, panel, best_single,
                                  "mandatory_life3_life4_pooled")
    print(f"[run_map] mandatory format_sanity_ok={mandatory_map['format_sanity_ok']}", flush=True)

    result = {"mandatory": mandatory_map, "optional": None}

    optional_runs = SM.discover_optional_runs(RUNS_DIR)
    print(f"[run_map] optional (Phase B.2) runs discovered: {len(optional_runs)}", flush=True)
    if optional_runs:
        optional_map = build_map_for(optional_runs, ds, panel, best_single,
                                     "optional_life5_default_crosscheck")
        print(f"[run_map] optional format_sanity_ok={optional_map['format_sanity_ok']}", flush=True)
        result["optional"] = optional_map

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    slot_map_summary = {"mandatory": _headline(result["mandatory"]),
                        "optional": _headline(result["optional"]) if result["optional"] else None}
    with open(METRICS_DIR / "slot_map.json", "w", encoding="utf-8") as f:
        json.dump(slot_map_summary, f, ensure_ascii=False, indent=2)
    with open(METRICS_DIR / "per_slot_signatures.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


if __name__ == "__main__":
    r = run_phase_b()
    print(f"[run_map] wrote {METRICS_DIR / 'slot_map.json'}", flush=True)
    print(f"[run_map] wrote {METRICS_DIR / 'per_slot_signatures.json'}", flush=True)
