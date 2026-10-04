# -*- coding: utf-8 -*-
"""yes_rate_by_position.py — 0 new calls. Correction check for K1 §2.1 (2026-10-04).

K1 explained the positional recall decay as "the model says 'no' to late records more and more
often". If that were the mechanism, the share of "yes" answers would fall with position. This
script counts, for every record position p = 0..17 in F2 (same data, same models, same order as
probe_lost_middle.py):

  yes_rate     share of (task, model) votes that included the record at p
  golden_rate  share of records at p that truly match (the task generator's ground truth)
  recall       yes on matching records / matching records
  false_yes    yes on non-matching records / non-matching records

Input: metrics/filter_calls.jsonl (per-call extracted id sets of E1/E5, F2 family).
Run from the experiment folder:  python scripts/yes_rate_by_position.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))
import task_generator_f2 as TF2      # noqa: E402
import model_registry_11 as MR       # noqa: E402

FAMILIES = (("E1-F2", "test"), ("E5-F2", "train"))


def load(fam, ids):
    votes = {t: {} for t in ids}
    for line in (HERE / "metrics" / "filter_calls.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["family"] == fam and r["task_id"] in votes:
            votes[r["task_id"]][r["model"]] = frozenset(r["extracted"])
    return votes


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    models = list(MR.MODEL_IDS)
    d = TF2.build_tasks()
    said_yes = defaultdict(lambda: [0, 0])
    gold = defaultdict(lambda: [0, 0])
    tp = defaultdict(lambda: [0, 0])
    fp = defaultdict(lambda: [0, 0])
    for fam, split in FAMILIES:
        ids = d[split]
        votes = load(fam, ids)
        for tid in ids:
            if any(m not in votes[tid] for m in models):
                continue
            task = d["tasks"][tid]
            g = set(task["matched_ids"])
            for p, rec in enumerate(task["records"]):
                truth = rec["id"] in g
                for m in models:
                    said = rec["id"] in votes[tid][m]
                    said_yes[p][0] += said
                    said_yes[p][1] += 1
                    gold[p][0] += truth
                    gold[p][1] += 1
                    if truth:
                        tp[p][0] += said
                        tp[p][1] += 1
                    else:
                        fp[p][0] += said
                        fp[p][1] += 1

    def r(a):
        return a[0] / a[1] if a[1] else float("nan")

    print("pos  yes_rate  golden_rate  recall  false_yes")
    for p in sorted(said_yes):
        print(f"{p:3d}  {r(said_yes[p]):.3f}     {r(gold[p]):.3f}       {r(tp[p]):.3f}   {r(fp[p]):.3f}")
    head = [p for p in said_yes if p <= 5]
    tail = [p for p in said_yes if p >= 12]

    def avg(ps, src):
        return sum(r(src[p]) for p in ps) / len(ps)

    for name, ps in (("head 0-5", head), ("tail 12-17", tail)):
        print(f"{name}: yes={avg(ps, said_yes):.3f} golden={avg(ps, gold):.3f} "
              f"recall={avg(ps, tp):.3f} false_yes={avg(ps, fp):.3f}")


if __name__ == "__main__":
    main()
