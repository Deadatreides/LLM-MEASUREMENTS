# -*- coding: utf-8 -*-
"""reanalyse_a2_reference.py — 0 new calls. Correction check for REPORT_PIPELINE (2026-10-05).

A reader on Reddit pointed out that a test suite which fails the correct reference will usually fail
every mutant of it too, so its mutation score is near 1.0 for free. This script splits the 168 generated
A2 suites by whether they accept the reference, recomputes the mutation score on each side, and splits
the 129 rejections by error class (wrong assert vs a suite that does not run at all).

Input: runs/pipeline_runs.*.jsonl and runs/seams.*.jsonl (compact indexes of the run).
Run from the experiment folder:  python metrics/reanalyse_a2_reference.py
"""
import collections
import glob
import json
import os
import statistics

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(pattern):
    rows = []
    for f in glob.glob(os.path.join(HERE, "runs", pattern)):
        with open(f, encoding="utf-8") as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    return rows


runs = load("pipeline_runs.*.jsonl")
seams = load("seams.*.jsonl")
acc = [r for r in runs if r["seam_a2_reference_status"] == "MECHANICAL_PASS"]
rej = [r for r in runs if r["seam_a2_reference_status"] != "MECHANICAL_PASS"]


def mut(group):
    m = [r["mutation_score_a2"] for r in group if r.get("mutation_score_a2") is not None]
    return statistics.mean(m), sum(1 for x in m if x == 1.0), len(m)


for name, g in (("accept the reference", acc), ("reject the reference", rej), ("all", runs)):
    mean, perfect, n = mut(g)
    print(f"{name:22s} n={len(g):3d}  mean mutation score {mean:.3f}  perfect {perfect}/{n}")

fails = [s for s in seams if s["seam_type"] == "A2_REFERENCE" and s["status"] != "MECHANICAL_PASS"]
print("\nwhy the 129 suites rejected the reference:")
for cls, n in collections.Counter(s.get("error_class") for s in fails).most_common():
    print(f"  {cls:26s} {n}")
ran = len(runs) - sum(1 for s in fails if s.get("error_class") != "TEST_FAILURE")
wrong = sum(1 for s in fails if s.get("error_class") == "TEST_FAILURE")
print(f"\nsuites that did not run at all: {len(runs) - ran}/{len(runs)} = {(len(runs) - ran) / len(runs):.1%}")
print(f"of the suites that ran, rejected the correct reference: {wrong}/{ran} = {wrong / ran:.1%}")
