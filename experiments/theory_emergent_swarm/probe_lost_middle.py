"""probe_lost_middle.py — 0 новых вызовов. Есть ли «потеря внимания в
середине» в уже собранных данных E1/E5?

У каждой задачи F2 ровно 18 записей в фиксированном порядке (тот же
порядок, что в db_text). По каждому вызову сохранён извлечённый набор id.
Значит для КАЖДОЙ позиции p=0..17 можно посчитать, верно ли модель
проголосовала за запись на этой позиции.

Если точность проседает в середине -- декомпозиция длинного контекста
получает измеренное обоснование, а не гипотетическое.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

import task_generator_f2 as TF2      # noqa: E402
import model_registry_11 as MR       # noqa: E402

FAMILIES = (("E1-F2", "test"), ("E5-F2", "train"))


def load(fam: str, ids: list) -> dict:
    votes = {t: {} for t in ids}
    for line in (HERE / "metrics" / "filter_calls.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["family"] == fam and r["task_id"] in votes:
            votes[r["task_id"]][r["model"]] = frozenset(r["extracted"])
    return votes


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    model_ids = list(MR.MODEL_IDS)
    d = TF2.build_tasks()

    pos_hit = defaultdict(lambda: [0, 0])          # position -> [correct, total]
    pos_hit_m = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    # separately: only the records that ARE golden matches (recall by position)
    pos_recall = defaultdict(lambda: [0, 0])

    for fam, split in FAMILIES:
        ids = d[split]
        votes = load(fam, ids)
        for tid in ids:
            if any(m not in votes[tid] for m in model_ids):
                continue
            task = d["tasks"][tid]
            gold = set(task["matched_ids"])
            for p, rec in enumerate(task["records"]):        # db_text order
                truth = rec["id"] in gold
                for m in model_ids:
                    said = rec["id"] in votes[tid][m]
                    ok = int(said == truth)
                    pos_hit[p][0] += ok
                    pos_hit[p][1] += 1
                    pos_hit_m[m][p][0] += ok
                    pos_hit_m[m][p][1] += 1
                    if truth:
                        pos_recall[p][0] += int(said)
                        pos_recall[p][1] += 1

    K = max(pos_hit) + 1
    print(f"=== per-position accuracy, F2 (K={K} records), E1+E5 pooled, 0 new calls ===\n")
    print("  pos  accuracy   recall(golden only)   bar")
    accs, recs = [], []
    for p in range(K):
        a = pos_hit[p][0] / pos_hit[p][1]
        r = pos_recall[p][0] / pos_recall[p][1] if pos_recall[p][1] else float("nan")
        accs.append(a)
        recs.append(r)
        bar = "#" * int(round((a - 0.5) * 100)) if a > 0.5 else ""
        print(f"  {p:3d}   {a:.3f}      {r:.3f}                {bar}")

    third = K // 3
    edge = accs[:third] + accs[-third:]
    mid = accs[third:K - third]
    edge_r = recs[:third] + recs[-third:]
    mid_r = recs[third:K - third]
    print(f"\n  edges (first {third} + last {third}) accuracy = {statistics.mean(edge):.3f}")
    print(f"  middle ({len(mid)} positions)          accuracy = {statistics.mean(mid):.3f}")
    print(f"  DROP IN THE MIDDLE                    = {statistics.mean(edge)-statistics.mean(mid):+.3f}")
    print(f"\n  edges  recall(golden) = {statistics.mean(edge_r):.3f}")
    print(f"  middle recall(golden) = {statistics.mean(mid_r):.3f}")
    print(f"  RECALL DROP           = {statistics.mean(edge_r)-statistics.mean(mid_r):+.3f}")

    print("\n  per-model middle-vs-edge recall drop:")
    for m in model_ids:
        a = [pos_hit_m[m][p][0] / pos_hit_m[m][p][1] for p in range(K)]
        e = statistics.mean(a[:third] + a[-third:])
        mi = statistics.mean(a[third:K - third])
        print(f"    {m:35s} edge={e:.3f} mid={mi:.3f} drop={e-mi:+.3f}")

    (HERE / "metrics" / "lost_middle.json").write_text(json.dumps({
        "K": K, "accuracy_by_position": accs, "recall_by_position": recs,
        "edge_acc": statistics.mean(edge), "mid_acc": statistics.mean(mid),
        "acc_drop": statistics.mean(edge) - statistics.mean(mid),
        "edge_recall": statistics.mean(edge_r), "mid_recall": statistics.mean(mid_r),
        "recall_drop": statistics.mean(edge_r) - statistics.mean(mid_r),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
