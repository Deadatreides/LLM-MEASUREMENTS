"""run_e6.py — E6: margin-gated targeted verification.
Thresholds pre-registered in PROTOCOL_E6.md SS3, fixed constants below.

Phase 1 (gate, 90 calls): measure single-record verification accuracy.
  If < 0.75 the premise "the atom is strictly easier" is FALSE -> abort,
  report, spend nothing further (gate-before-budget).
Phase 2 (main, ~360 calls): re-query only the LOW-margin half, only the
  R most-uncertain records, then re-decode.

F2 TEST only -- the same 40 tasks carrying DELTA-0's r_m*=0.525 and every
E1/E4 number, so comparisons need no transfer assumption.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import task_generator_f2 as TF2      # noqa: E402
import consensus_filter as CF        # noqa: E402
import structure_decode as SD        # noqa: E402
import oracles as OR                 # noqa: E402
import det_atoms as DA               # noqa: E402
import model_registry_11 as MR       # noqa: E402
import call_log                      # noqa: E402

AGENT_DIR = Path(__file__).resolve().parents[1]
METRICS_DIR = AGENT_DIR / "metrics"

PI_ADMIT = 0.60           # from E5, held out-of-sample
R_RECORDS = 3             # pre-registered, not tuned
SEED_BASE_VERIFY = 99000
MAX_TOKENS_VERIFY = 8
TEMPERATURE = 0.0

THR_GATE = 0.75           # premise falsified below this
THR_GATE_PREDICT = 0.90
BASE_E4 = 0.750           # E4 @ pi>0.60, same 40 tasks
THR_MAIN, THR_MAIN_FALSIFY = 0.85, 0.80
THR_LOW = 0.60

N_SMOKE_TASKS, N_SMOKE_RECORDS = 3, 5


def build_verify_prompt(task: dict, rec: dict) -> str:
    line = (f"{rec['id']} | account: {rec['account']} | category: {rec['category']} | "
            f"amount: {rec['amount']:.2f} | region: {rec['region']} | date: {rec['date']}")
    return (f"Запись: {line}\n\n"
            f"Вопрос: у этой записи category = \"{task['target_category']}\" "
            f"И region = \"{task['target_region']}\"?\n"
            "Ответь ровно одним словом: ДА или НЕТ.")


def draw_seed(task_id: str, rec_id: str, model_id: str) -> int:
    h = hashlib.sha256(f"{task_id}|{rec_id}|{model_id}".encode("utf-8")).hexdigest()
    return SEED_BASE_VERIFY + int(h, 16) % 10 ** 6


def ask(llm, model_id: str, task: dict, rec: dict):
    seed = draw_seed(task["task_id"], rec["id"], model_id)
    t0 = time.time()
    raw = MR.generate(llm, model_id, build_verify_prompt(task, rec), temperature=TEMPERATURE,
                      top_p=1.0, seed=seed, max_tokens=MAX_TOKENS_VERIFY)
    ms = (time.time() - t0) * 1000.0
    txt = raw.get("raw_text", "")
    call_log.log_filter_call(model=model_id, task_id=f"{task['task_id']}:{rec['id']}",
                             family="E6-verify", seed=seed, n_in=raw.get("input_tokens") or 0,
                             n_out=raw.get("output_tokens") or 0, ms=ms,
                             failed=bool(raw.get("generation_failed")),
                             extracted=[], raw_text=txt)
    return OR.extract_yes_no(txt)      # "ДА" / "НЕТ" / None


def load_votes(task_ids, model_ids) -> dict:
    votes = {t: {} for t in task_ids}
    for line in (METRICS_DIR / "filter_calls.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["family"] == "E1-F2" and r["task_id"] in votes:
            votes[r["task_id"]][r["model"]] = {"extracted": frozenset(r["extracted"])}
    if any(m not in votes[t] for t in task_ids for m in model_ids):
        raise RuntimeError("HARD_STOP: E1-F2 votes incomplete")
    return votes


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    model_ids = list(MR.MODEL_IDS)
    d = TF2.build_tasks()
    tasks_by_id, test_ids = d["tasks"], d["test"]
    votes = load_votes(test_ids, model_ids)

    # ---- baseline decode + margin (0 calls) ----
    base = {}
    for tid in test_ids:
        task = tasks_by_id[tid]
        cids = CF.candidate_ids(task)
        classes = SD.enumerate_classes(task)
        rel = CF.loo_reliabilities(tasks_by_id, test_ids, model_ids, votes, tid)
        admitted = sorted(m for m in model_ids if rel[m] > PI_ADMIT)
        sup = SD.swarm_support(task, votes[tid], admitted)
        _k, dec, _s = SD.decode(sup, cids, classes)
        scores = sorted((SD.agreement(sup, cids, mem) for mem in classes.values()), reverse=True)
        margin = scores[0] - scores[1] if len(scores) > 1 else scores[0]
        base[tid] = {"support": sup, "decoded": dec, "margin": margin, "classes": classes,
                    "cids": cids, "admitted": admitted,
                    "correct": dec == frozenset(task["matched_ids"])}
    base_rate = sum(b["correct"] for b in base.values()) / len(test_ids)
    print(f"baseline (E4 @ pi>{PI_ADMIT}) exact-set = {base_rate:.3f}   "
         f"(pre-registered reference {BASE_E4:.3f})\n", flush=True)

    # ================= PHASE 1: GATE =================
    print("=== PHASE 1: gate -- is single-record verification actually easier? ===", flush=True)
    smoke_ids = test_ids[:N_SMOKE_TASKS]
    per_model_hit = {m: [0, 0] for m in model_ids}
    for model_id in model_ids:
        llm, _ = MR.load_model(model_id)
        for tid in smoke_ids:
            task = tasks_by_id[tid]
            for rec in task["records"][:N_SMOKE_RECORDS]:
                got = ask(llm, model_id, task, rec)
                truth = "ДА" if rec["id"] in set(task["matched_ids"]) else "НЕТ"
                per_model_hit[model_id][1] += 1
                per_model_hit[model_id][0] += int(got == truth)
        del llm
    accs = {m: h / n for m, (h, n) in per_model_hit.items()}
    gate_acc = sum(h for h, _ in per_model_hit.values()) / sum(n for _, n in per_model_hit.values())
    for m in model_ids:
        print(f"  {m:35s} verify acc = {accs[m]:.3f}")
    print(f"\n  POOLED verification accuracy = {gate_acc:.3f}   "
         f"(predict >={THR_GATE_PREDICT}, falsify <{THR_GATE})", flush=True)

    if gate_acc < THR_GATE:
        res = {"phase": "GATE_FAILED", "gate_acc": gate_acc, "per_model": accs,
               "threshold": THR_GATE, "baseline": base_rate,
               "note": "premise 'the atom is strictly easier' FALSIFIED; main run not executed"}
        (METRICS_DIR / "e6_result.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\n  GATE FAILED -- premise false. Main run NOT executed, budget not spent.")
        return

    # ================= PHASE 2: MAIN =================
    ordered = sorted(test_ids, key=lambda t: base[t]["margin"])
    low_half = ordered[:len(ordered) // 2]
    print(f"\n=== PHASE 2: re-query {len(low_half)} low-margin tasks x {R_RECORDS} records "
         f"x {len(model_ids)} models = {len(low_half) * R_RECORDS * len(model_ids)} calls ===",
         flush=True)

    targets = {}
    for tid in low_half:
        b = base[tid]
        recs = sorted(tasks_by_id[tid]["records"],
                     key=lambda r: (abs(b["support"].get(r["id"], 0.0) - 0.5), r["id"]))
        targets[tid] = recs[:R_RECORDS]

    answers = {tid: {r["id"]: {} for r in targets[tid]} for tid in low_half}
    for model_id in model_ids:
        llm, _ = MR.load_model(model_id)
        for tid in low_half:
            for rec in targets[tid]:
                answers[tid][rec["id"]][model_id] = ask(llm, model_id, tasks_by_id[tid], rec)
        del llm

    # swarm-on-easy: per-model verification accuracy vs mode-of-6, on THESE calls
    vm_hit = {m: [0, 0] for m in model_ids}
    mode_hit = [0, 0]
    for tid in low_half:
        gold = set(tasks_by_id[tid]["matched_ids"])
        for rec in targets[tid]:
            truth = "ДА" if rec["id"] in gold else "НЕТ"
            got = answers[tid][rec["id"]]
            for m in model_ids:
                vm_hit[m][1] += 1
                vm_hit[m][0] += int(got[m] == truth)
            yes = sum(1 for m in model_ids if got[m] == "ДА")
            no = sum(1 for m in model_ids if got[m] == "НЕТ")
            mode_hit[1] += 1
            mode_hit[0] += int(("ДА" if yes > no else "НЕТ") == truth)
    vm_acc = {m: h / n for m, (h, n) in vm_hit.items()}
    best_verifier = max(model_ids, key=lambda m: vm_acc[m])
    mode_acc = mode_hit[0] / mode_hit[1]
    swarm_gain_easy = mode_acc - vm_acc[best_verifier]

    # ---- re-decode with updated support ----
    n_ok = 0
    per_task = {}
    for tid in test_ids:
        b = base[tid]
        if tid in low_half:
            sup = dict(b["support"])
            for rec in targets[tid]:
                got = answers[tid][rec["id"]]
                yes = sum(1 for m in model_ids if got[m] == "ДА")
                no = sum(1 for m in model_ids if got[m] == "НЕТ")
                if yes + no > 0:
                    sup[rec["id"]] = yes / (yes + no)
            _k, dec, _s = SD.decode(sup, b["cids"], b["classes"])
        else:
            dec = b["decoded"]
        ok = dec == frozenset(tasks_by_id[tid]["matched_ids"])
        n_ok += int(ok)
        per_task[tid] = {"before": sorted(b["decoded"]), "after": sorted(dec),
                        "golden": sorted(tasks_by_id[tid]["matched_ids"]),
                        "margin": b["margin"], "requeried": tid in low_half,
                        "was_correct": b["correct"], "now_correct": ok}

    after = n_ok / len(test_ids)
    low_before = sum(base[t]["correct"] for t in low_half) / len(low_half)
    low_after = sum(per_task[t]["now_correct"] for t in low_half) / len(low_half)

    # final yes/no chain, same as E4
    fin = 0
    for tid in test_ids:
        task = tasks_by_id[tid]
        ids = frozenset(per_task[tid]["after"])
        agg = DA.aggregate_det(ids, task["op"], task["id_to_amount"])
        fin += int(DA.derive_det(agg, task["threshold"], task["comparator"]) == task["final_oracle"])
    final_rate = fin / len(test_ids)

    v_main = ("CONFIRMED" if after >= THR_MAIN
              else "FALSIFIED" if after < THR_MAIN_FALSIFY else "ABOVE_FLOOR_BELOW_PREDICTION")
    v_low = "CONFIRMED" if low_after >= THR_LOW else "FALSIFIED"
    v_swarm = "SWARM_HELPS_ON_EASY" if swarm_gain_easy > 0 else "SWARM_DEAD_EVEN_ON_EASY"

    res = {
        "STRUCTURE_DET": True, "phase": "COMPLETE",
        "gate_acc": gate_acc, "gate_per_model": accs,
        "baseline_exact_set": base_rate, "after_exact_set": after,
        "final_yes_no": final_rate,
        "low_half_before": low_before, "low_half_after": low_after,
        "verify_per_model": vm_acc, "verify_mode_of_6": mode_acc,
        "best_verifier": best_verifier, "swarm_gain_on_easy": swarm_gain_easy,
        "calls_phase1": sum(n for _, n in per_model_hit.values()),
        "calls_phase2": len(low_half) * R_RECORDS * len(model_ids),
        "verdicts": {"E6_main": v_main, "E6_low": v_low, "E6_swarm": v_swarm},
        "thresholds": {"main": THR_MAIN, "main_falsify": THR_MAIN_FALSIFY, "low": THR_LOW,
                      "gate": THR_GATE},
        "per_task": per_task,
    }
    (METRICS_DIR / "e6_result.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== E6 RESULT (F2 test, n=40) ===")
    print(f"  exact-set  before -> after : {base_rate:.3f} -> {after:.3f}   [{v_main}]")
    print(f"  low-margin half            : {low_before:.3f} -> {low_after:.3f}   [{v_low}]")
    print(f"  final yes/no after         : {final_rate:.3f}   (whole-task r_m* = 0.525)")
    print(f"\n  verification accuracy, per model:")
    for m in model_ids:
        print(f"    {m:35s} {vm_acc[m]:.3f}")
    print(f"    {'MODE of 6':35s} {mode_acc:.3f}")
    print(f"    {'best single verifier':35s} {vm_acc[best_verifier]:.3f}  ({best_verifier})")
    print(f"  SWARM GAIN ON EASY QUESTION : {swarm_gain_easy:+.3f}   [{v_swarm}]")


if __name__ == "__main__":
    main()
