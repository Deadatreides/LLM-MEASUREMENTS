"""run_m2_oos.py — out-of-sample test of ONE architecture, fixed in advance.

Post-hoc on the M2 TEST split, six hop-1 architectures were compared and
`best_single + МДП fallback into the reliability-ranked pool` came first
(0.600 vs 0.525 for the best single alone). Six comparisons on 40 tasks is
multiple-comparison territory and +0.075 is three tasks, so it counts as a
HYPOTHESIS, not a result.

This script tests exactly that one architecture on the never-used M2 TRAIN
split. Constants below are frozen before the run:

  ARCH      : answer = best model's own answer if it passes the seam check,
              else the first pool candidate (ranked by reliability mass)
              that passes; if none passes -> INAPPLICABLE (scored FAIL,
              never golden-substituted)
  RELIAB    : per-model hop-1 reliability estimated on the TEST split
              (a disjoint set of tasks), applied unchanged to TRAIN
  THRESHOLD : ARCH > best_single on TRAIN, else the hypothesis is rejected

Only hop 1 is re-run: it is where the fixed seam made МДП discriminating,
and it is the measured bottleneck (finding a maximum among 8 rows).
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import task_generator_m2 as M2     # noqa: E402
import model_registry_11 as MR     # noqa: E402
import call_log                    # noqa: E402

METRICS = Path(__file__).resolve().parents[1] / "metrics"
_DEP = re.compile(r"DEP-\d{4}")
SEED = 5151
MAX_ID = 16


def read_test_reliability(models, tb, test_ids) -> dict:
    h1 = defaultdict(dict)
    for line in (METRICS / "filter_calls.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["family"] == "M2-h1" and r["task_id"] in set(test_ids):
            m = _DEP.search(r["raw_text"] or "")
            h1[r["task_id"]][r["model"]] = m.group(0) if m else None
    n = len(test_ids)
    return {m: sum(1 for t in test_ids if h1[t].get(m) == tb[t]["hop1_oracle"]) / n
            for m in models}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    models = list(MR.MODEL_IDS)
    d = M2.build_tasks()
    tb, train_ids, test_ids = d["tasks"], d["train"], d["test"]

    rel = read_test_reliability(models, tb, test_ids)
    best_m = max(models, key=lambda m: rel[m])
    print("reliability from TEST split (frozen, applied to TRAIN):")
    for m in models:
        print(f"  {m:35s} {rel[m]:.3f}{'   <- best' if m == best_m else ''}")
    print(f"\nOOS on TRAIN: {len(train_ids)} tasks x {len(models)} models\n", flush=True)

    answers = defaultdict(dict)
    t0 = time.time()
    for model_id in models:
        llm, _ = MR.load_model(model_id)
        for tid in train_ids:
            r = MR.generate(llm, model_id, M2.prompt_hop1(tb[tid]), temperature=0.0,
                            top_p=1.0, seed=SEED, max_tokens=MAX_ID)
            txt = r.get("raw_text", "")
            m = _DEP.search(txt)
            answers[tid][model_id] = m.group(0) if m else None
            call_log.log_filter_call(model=model_id, task_id=tid, family="M2OOS-h1", seed=SEED,
                                     n_in=r.get("input_tokens") or 0,
                                     n_out=r.get("output_tokens") or 0, ms=0.0,
                                     failed=bool(r.get("generation_failed")),
                                     extracted=[answers[tid][model_id]], raw_text=txt)
        del llm
        print(f"  {model_id} done ({time.time()-t0:.0f}s)", flush=True)

    n = len(train_ids)
    n_single = n_arch = n_mode = n_fired = n_broke = n_inapp = 0
    for tid in train_ids:
        task, gold = tb[tid], tb[tid]["hop1_oracle"]
        wsum, votes = defaultdict(float), defaultdict(int)
        for m in models:
            v = answers[tid][m]
            if v:
                wsum[v] += rel[m]
                votes[v] += 1
        by_rel = sorted(wsum.items(), key=lambda kv: (-kv[1], kv[0]))
        by_vote = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))

        single = answers[tid][best_m]
        n_single += int(single == gold)
        n_mode += int((by_vote[0][0] if by_vote else None) == gold)

        if single is not None and M2.mdp_seam1(task, single):
            pick = single
        else:
            pick = next((v for v, _w in by_rel if M2.mdp_seam1(task, v)), None)
            n_fired += 1
            if pick is None:
                n_inapp += 1
        n_arch += int(pick == gold)
        n_broke += int(single == gold and pick != gold)

    r_single, r_arch, r_mode = n_single / n, n_arch / n, n_mode / n
    delta = r_arch - r_single
    verdict = "CONFIRMED_OOS" if delta > 0 else "REJECTED"
    safe = "SAFE" if n_broke == 0 else "UNSAFE"

    print(f"\n=== M2-OOS (hop 1, TRAIN split, n={n}, {n*len(models)} calls) ===")
    print(f"  best single ({best_m[:22]})   = {r_single:.3f}")
    print(f"  mode over 6                     = {r_mode:.3f}")
    print(f"  ARCH single + MDP fallback      = {r_arch:.3f}")
    print(f"  delta ARCH - best single        = {delta:+.3f}   [{verdict}]")
    print(f"  MDP fired {n_fired}/{n}, broke {n_broke} [{safe}], INAPPLICABLE {n_inapp}")

    (METRICS / "m2_oos_result.json").write_text(json.dumps({
        "split": "TRAIN (never used)", "n": n, "calls": n * len(models),
        "reliability_source": "TEST split, frozen", "reliability": rel, "best_model": best_m,
        "r_best_single": r_single, "r_mode": r_mode, "r_arch": r_arch, "delta": delta,
        "mdp_fired": n_fired, "mdp_broke": n_broke, "inapplicable": n_inapp,
        "verdict": verdict, "safety": safe,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
