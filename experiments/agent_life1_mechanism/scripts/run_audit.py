"""run_audit.py — PROTOCOL.md: Gate R1 -> Gate R2 -> Phase A audit -> корзина
MECH_*. Не запускает generate() -- всё поверх уже собранной живой сетки
`agent_a5_live_m/metrics/live_grid/` (только чтение).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import live_dataset as LD          # noqa: E402
import deterministic_route as DR   # noqa: E402
import orchestrator as ORCH        # noqa: E402
import lineage_audit as LA         # noqa: E402

AGENT_DIR = Path(__file__).resolve().parents[1]
METRICS_DIR = AGENT_DIR / "metrics"
RUNS_DIR = METRICS_DIR / "runs_instrumented"
A5_RUNS = AGENT_DIR.parent / "agent_a5_live_m" / "runs"

SEEDS = [20261201, 20261202, 20261203, 20261204, 20261205]

# Восстановлено из трасс агивного прогона A5 (agent_a5_live_m, seed 20261201) --
# PROTOCOL.md §3, Gate R1 сравнивает с этими id.
TARGET_REFERENCE_IDS = {
    "REF_B1_route": "cx-7b8668",
    "REF_B2_greedy2": "cx-7ddd67",
    "REF_B3_greedy_cover": "cx-df47df",
}


def gate_r1(ds) -> dict:
    train_ids = ds.split["train"]
    res = DR.find_matching_variant(ds, train_ids, TARGET_REFERENCE_IDS)
    print(f"[gate-r1] matched={res['matched']}", flush=True)
    if not res["matched"]:
        raise RuntimeError("Gate R1 failed: no tie-break variant reproduces the archived "
                           "A5 reference complex_ids -- see BLOCKERS.md")
    print(f"[gate-r1] orders={res['orders']}", flush=True)
    return res["orders"]


def gate_r2(ds, orders: dict) -> dict:
    report = {}
    all_match = True
    for seed in SEEDS:
        t0 = time.time()
        out_dir = RUNS_DIR / f"life1_s{seed}"
        summary = ORCH.run_seed_campaign(seed, ds, orders, out_dir)
        with open(out_dir / "archive.json", encoding="utf-8") as f:
            got_ids = {g["complex_id"] for g in json.load(f)["genotypes"]}
        a5_ids = {p.stem for p in (A5_RUNS / f"a5_live_s{seed}").glob("cx-*.jsonl")}
        match = got_ids == a5_ids
        all_match = all_match and match
        dt = time.time() - t0
        print(f"[gate-r2] seed {seed}: got={len(got_ids)} a5={len(a5_ids)} "
              f"match={match} extinct={summary['extinct']} gens={summary['n_generations']} "
              f"{dt:.0f}s", flush=True)
        report[seed] = {"match": match, "n_got": len(got_ids), "n_a5": len(a5_ids),
                        "extinct": summary["extinct"], "n_generations": summary["n_generations"],
                        "wall_seconds": dt}
    if not all_match:
        raise RuntimeError("Gate R2 failed: at least one seed's complex_id set does not match "
                           "the archived A5 run -- see BLOCKERS.md, proceed on best-effort only "
                           "with explicit disclosure")
    return report


def gain_loss_diagnostic(ds, seeds: list) -> dict:
    """Доп. диагностика для §8 отчёта (обязательна при MECH_MIXED/UNKNOWN,
    PROTOCOL.md/plan: 'deeper audit + explicit list of what to add')."""
    fmt_violations = 0
    delta_hist: dict = {}
    by_op: dict = {}
    total = 0

    for seed in seeds:
        art = LA.load_seed_artifacts(RUNS_DIR / f"life1_s{seed}")
        genotypes, summary, births = art["genotypes"], art["summary"], art["births"]
        panel = summary["panel"]
        best_single = LA.best_single_per_slot(ds, panel)
        imp_cache: dict = {}

        def imp(cid, _gt=genotypes, _cache=imp_cache):
            if cid not in _cache:
                g = _gt.get(cid)
                _cache[cid] = LA.imp_set(ds, panel, best_single, g) if g else frozenset()
            return _cache[cid]

        for cid in genotypes:
            if 1 in imp(cid):     # FORMAT = slot index 1, structurally saturated (PROTOCOL.md §1.3)
                fmt_violations += 1

        for b in births:
            if not b.get("parent_id"):
                continue
            cid, pid = b["child_id"], b["parent_id"]
            if cid not in genotypes or pid not in genotypes:
                continue
            total += 1
            d = len(imp(cid)) - len(imp(pid))
            delta_hist[d] = delta_hist.get(d, 0) + 1
            op = b.get("operator") or "?"
            rec = by_op.setdefault(op, {"n": 0, "gain": 0, "loss": 0, "unchanged": 0})
            rec["n"] += 1
            if d > 0:
                rec["gain"] += 1
            elif d < 0:
                rec["loss"] += 1
            else:
                rec["unchanged"] += 1

    return {"format_imp_violations": fmt_violations, "total_evolved_births": total,
           "delta_impset_histogram": delta_hist, "gain_loss_by_operator": by_op}


def main() -> int:
    ds = LD.default_dataset()

    print("=== Gate R1 ===", flush=True)
    orders = gate_r1(ds)

    print("=== Gate R2 ===", flush=True)
    r2_report = gate_r2(ds, orders)

    print("=== Phase A: lineage audit ===", flush=True)
    per_seed = [LA.audit_seed(seed, RUNS_DIR / f"life1_s{seed}", ds) for seed in SEEDS]
    pooled = LA.pool_seeds(per_seed)
    basket = LA.classify_basket(pooled, per_seed)
    print(f"[basket] {basket}", flush=True)

    print("=== Deeper diagnostic (gain/loss by operator) ===", flush=True)
    diagnostic = gain_loss_diagnostic(ds, SEEDS)
    print(json.dumps(diagnostic, indent=2), flush=True)

    out = {
        "gate_r1_orders": {k: list(v) for k, v in orders.items()},
        "gate_r2": r2_report,
        "per_seed": [{k: v for k, v in r.items() if not k.startswith("_")} for r in per_seed],
        "pooled": pooled,
        "basket": basket,
        "diagnostic": diagnostic,
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = METRICS_DIR / "lineage_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f"[run_audit] {out_path} записан", flush=True)
    print(f"[run_audit] ИТОГ: {basket['basket']}", flush=True)
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
