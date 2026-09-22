"""probe_specialize.py — два бесплатных зонда (0 новых вызовов GPU),
проверяющих предпосылки идеи «PTG + граф» ДО того, как её строить.

ЗОНД A (данные E0, whole_grid.json): division of labour.
  Голосование = избыточность (N единиц, ОДИН вопрос). Мозг = дифференциация
  (N единиц, РАЗНЫЕ вопросы). E0 сравнивала только схемы голосования и
  никогда -- маршрутизацию поля своему специалисту (argmax pi_{m,f},
  выбранный на train). Это прямой тест «дифференциация > избыточность».

ЗОНД B (данные E4/E5, filter_calls.jsonl): знает ли декодер, что не знает?
  margin = score(лучший класс) - score(второй). Если margin предсказывает
  правильность, то существует дешёвая политика «спрашивать только там, где
  неуверен» -- предпосылка любой активной/графовой схемы. Если margin
  неинформативен, активный опрос строить не на чем.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

import validate_esw0 as V0   # noqa: E402  (переиспользуем разбор полей E0)

N_SPLITS = 40
SPLIT_SEED = 7


# ---------------------------------------------------------------- ЗОНД A
def specialist_route(per, tasks, models, rel, test_ids) -> float:
    """Каждому полю -- СВОЯ модель (argmax pi по train). Без голосования."""
    spec = {f: max(models, key=lambda m: rel[(m, f)]) for f in V0.FIELDS}
    ok = 0
    for t in test_ids:
        if all(V0.eq(per[t][spec[f]][f], tasks[t]["steps"][i]["oracle"])
               for i, f in enumerate(V0.FIELDS)):
            ok += 1
    return ok / len(test_ids)


def specialist_plus_vote(per, tasks, models, rel, test_ids) -> float:
    """Гибрид: специалист решает, если он заметно лучше поля-консенсуса
    (pi_spec - pi_2nd >= 0.10), иначе -- кондорсе-мода. Проверяет, есть ли
    смысл СМЕШИВАТЬ дифференциацию и избыточность."""
    ok = 0
    for t in test_ids:
        good = True
        for i, f in enumerate(V0.FIELDS):
            ranked = sorted(models, key=lambda m: -rel[(m, f)])
            spec, second = ranked[0], ranked[1]
            if rel[(spec, f)] - rel[(second, f)] >= 0.10:
                val = per[t][spec][f]
            else:
                pool = [m for m in models if rel[(m, f)] >= 0.5] or [spec]
                tally: dict = defaultdict(float)
                for m in pool:
                    v = per[t][m][f]
                    if v is None:
                        continue
                    tally[round(v, 2) if isinstance(v, float) else v] += 1.0
                if not tally:
                    good = False
                    break
                val = max(tally.items(), key=lambda kv: kv[1])[0]
            if not V0.eq(val, tasks[t]["steps"][i]["oracle"]):
                good = False
                break
        if good:
            ok += 1
    return ok / len(test_ids)


def probe_a() -> dict:
    tasks, per, full6, models = V0.load_fields()
    res = defaultdict(list)
    spec_choices = defaultdict(lambda: defaultdict(int))
    rng = random.Random(SPLIT_SEED)
    for _ in range(N_SPLITS):
        ids = full6[:]
        rng.shuffle(ids)
        tr, te = ids[:50], ids[50:]
        rel = V0.reliabilities(per, tasks, models, tr)
        res["best_single"].append(V0.best_single_honest(per, tasks, models, tr, te))
        res["condorcet"].append(V0.consensus_score(per, tasks, models, rel, te, "condorcet"))
        res["specialist"].append(specialist_route(per, tasks, models, rel, te))
        res["spec_plus_vote"].append(specialist_plus_vote(per, tasks, models, rel, te))
        for f in V0.FIELDS:
            spec_choices[f][max(models, key=lambda m: rel[(m, f)])] += 1

    print("=== ZOND A: division of labour (E0 data, 0 GPU calls) ===\n")
    for k, lbl in (("best_single", "best single (one model, all fields)"),
                   ("condorcet", "condorcet vote per field"),
                   ("specialist", "SPECIALIST per field (no voting)"),
                   ("spec_plus_vote", "specialist if clear, else vote")):
        v = res[k]
        print(f"  {lbl:38s} {statistics.mean(v):.3f}  [{min(v):.2f}..{max(v):.2f}]")

    sw = sum(1 for a, b in zip(res["specialist"], res["condorcet"]) if a > b)
    sb = sum(1 for a, b in zip(res["specialist"], res["best_single"]) if a > b)
    print(f"\n  specialist > condorcet   : {sw}/{N_SPLITS}")
    print(f"  specialist > best_single : {sb}/{N_SPLITS}")
    print("\n  who owns each field (train-picked, across splits):")
    for f in V0.FIELDS:
        top = sorted(spec_choices[f].items(), key=lambda kv: -kv[1])
        print(f"    {f:14s} " + ", ".join(f"{m[:22]}x{c}" for m, c in top[:3]))
    n_distinct = len({max(spec_choices[f].items(), key=lambda kv: kv[1])[0] for f in V0.FIELDS})
    print(f"\n  distinct specialists across 4 fields: {n_distinct}"
          "   <- 1 means no real division of labour")
    return {k: statistics.mean(v) for k, v in res.items()} | {
        "specialist_beats_condorcet": sw, "specialist_beats_best_single": sb,
        "n_distinct_specialists": n_distinct}


# ---------------------------------------------------------------- ЗОНД B
def probe_b() -> dict:
    import task_generator_f2 as TF2
    import task_generator_hard as THARD
    import consensus_filter as CF
    import structure_decode as SD
    import model_registry_11 as MR

    model_ids = list(MR.MODEL_IDS)
    out = {}
    print("\n\n=== ZOND B: does the decoder know when it is unsure? ===\n")

    for fam, gen, split in (("E1-F2", TF2, "test"), ("E5-F2", TF2, "train"),
                            ("E2-Thard", THARD, "test"), ("E5-Thard", THARD, "train")):
        d = gen.build_tasks()
        tasks_by_id, ids = d["tasks"], d[split]
        votes = {t: {} for t in ids}
        for line in (HERE / "metrics" / "filter_calls.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["family"] == fam and r["task_id"] in votes:
                votes[r["task_id"]][r["model"]] = {"extracted": frozenset(r["extracted"])}
        if any(m not in votes[t] for t in ids for m in model_ids):
            continue

        rows = []
        for tid in ids:
            task = tasks_by_id[tid]
            cids = CF.candidate_ids(task)
            classes = SD.enumerate_classes(task)
            rel = CF.loo_reliabilities(tasks_by_id, ids, model_ids, votes, tid)
            admitted = sorted(m for m in model_ids if rel[m] > 0.60)
            sup = SD.swarm_support(task, votes[tid], admitted)
            scored = sorted(((SD.agreement(sup, cids, mem), k) for k, mem in classes.items()),
                            reverse=True)
            top, second = scored[0], scored[1]
            margin = top[0] - second[0]
            correct = classes[top[1]] == frozenset(task["matched_ids"])
            rows.append((margin, correct))

        rows.sort(key=lambda r: -r[0])
        n = len(rows)
        half = n // 2
        hi = statistics.mean(c for _, c in rows[:half])
        lo = statistics.mean(c for _, c in rows[half:])
        overall = statistics.mean(c for _, c in rows)
        # what a "answer only when confident" policy would buy
        print(f"  {fam:10s} {split:5s} n={n:3d}  overall={overall:.3f}   "
              f"top-half margin acc={hi:.3f}   bottom-half={lo:.3f}   spread={hi - lo:+.3f}")
        out[f"{fam}:{split}"] = {"n": n, "overall": overall, "hi": hi, "lo": lo,
                                "spread": hi - lo}
    print("\n  spread > 0 means margin IS an uncertainty signal -> selective querying is buildable")
    return out


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    a = probe_a()
    b = probe_b()
    (HERE / "metrics" / "probe_specialize.json").write_text(
        json.dumps({"zond_a_division_of_labour": a, "zond_b_decoder_margin": b},
                   ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
