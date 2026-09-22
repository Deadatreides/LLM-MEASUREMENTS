"""Gather the numbers task 2 needs, from whatever analyses exist."""

import json
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "out"


def load(tag):
    p = OUT / f"{tag}.analysis.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


print("=== A.3: EPOCH - STATIC by corpus part ===")
for tag, what in (("G", "весь корпус"), ("G-A", "часть A, 510 запросов"),
                  ("G-B", "часть B, 20 сессий"), ("B", "dense, весь корпус"),
                  ("B-A", "dense, часть A"), ("B-B", "dense, часть B")):
    d = load(tag)
    if not d:
        print(f"  {tag:5s} —")
        continue
    t = d["q_table"]["5"]
    s, e = t["STATIC"]["q_steady"], t["EPOCH"]["q_steady"]
    print(f"  {tag:5s} {what:24s} tokens={d['trace']['tokens']:6d} "
          f"STATIC={s:.4f} EPOCH={e:.4f} gap={100*(e-s):+6.2f} pp "
          f"ceiling={t['EPOCH']['q_ceiling']:.3f}")

print("\n=== A.1: band vs channel unit ===")
for tag, what in (("B", "полосы s=48"), ("Bch", "каналы s=1")):
    d = load(tag)
    if not d:
        print(f"  {tag:5s} {what:14s} — анализ ещё не готов")
        continue
    tr = d["trace"]
    t5 = d["q_table"]["5"]
    iso = d.get("iso_ratio_point", {})
    core = d.get("core", {})
    print(f"  {tag:5s} {what:14s} units={tr['units_total']:6d} "
          f"active/tok={tr['mean_active_per_token']:9.1f} "
          f"({100*tr['mean_active_per_token']/tr['units_total']:5.2f} %)")
    print(f"        q(M=5%,EPOCH)={t5['EPOCH']['q_steady']:.4f} "
          f"ceiling={t5['EPOCH']['q_ceiling']:.3f} "
          f"STATIC={t5['STATIC']['q_steady']:.4f}")
    print(f"        iso M={iso.get('M_percent')} % "
          f"q={iso.get('q_steady', float('nan')):.4f} "
          f"clamped={iso.get('clamped')}")
    print(f"        core={core.get('n_core')} units "
          f"({100*core.get('core_of_units', 0):.1f} % of model)  "
          f"L_persist={d['persistence']['L_persist']} "
          f"o_1={d['persistence']['o_k']['1']:.3f}")

print("\n=== A.2: S(k) ===")
p = OUT / "B_clusters.json"
if p.exists():
    d = json.loads(p.read_text(encoding="utf-8"))
    print(f"  p = {d['p']:.4f}, tokens sampled {d['tokens_sampled']}")
    for k, e in d["k"].items():
        print(f"  k={int(k):3d}  a_contig={e['a_contig']:.4f} "
              f"a_clust={e['a_clust']:.4f} a_cond={e['a_clust_cond']:.4f} "
              f"a_random={e['a_random']:.4f}  S={e['S']:.3f} "
              f"S_cond={e['S_cond']:.3f}  S_random={e['S_random']:+.4f}")
else:
    print("  нет")

print("\n=== C: needle ===")
p = OUT / "needle_raw.json"
if p.exists():
    rows = json.loads(p.read_text(encoding="utf-8"))
    print(f"  {len(rows)} строк собрано")
else:
    print("  ещё идёт")
