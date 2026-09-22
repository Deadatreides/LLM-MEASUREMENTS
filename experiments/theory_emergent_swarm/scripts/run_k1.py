"""run_k1.py — K1: Kahn office as swarm architecture.
Thresholds pre-registered in PROTOCOL_K1.md SS3.

Disciplines = context blocks (6 records each). Executors = 6 models per
block. VDP/normokontrol = deterministic filters. Assembly = composition
of per-block votes into one 18-record support vector. Then E4 decoding,
then E6-style consultation on low-margin records.

Baseline (whole-context) is read from disk -- 0 new calls for it.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import task_generator_f2 as TF2      # noqa: E402
import consensus_filter as CF        # noqa: E402
import structure_decode as SD        # noqa: E402
import det_atoms as DA               # noqa: E402
import model_registry_11 as MR       # noqa: E402
import call_log                      # noqa: E402

METRICS = Path(__file__).resolve().parents[1] / "metrics"
BLOCK = 6
PI_ADMIT = 0.60
MAX_TOKENS = 160
TEMPERATURE = 0.0
SEED_BASE = 77000
BASE_E4_SET = 0.750
THR_RECALL, THR_RECALL_FALSIFY = 0.45, 0.35
THR_SET, THR_SET_FALSIFY = 0.85, 0.78
THR_EMERGENCE = 3


def blocks_of(task: dict) -> list:
    recs = task["records"]
    return [recs[i:i + BLOCK] for i in range(0, len(recs), BLOCK)]


def render(recs: list) -> str:
    return "\n".join(
        f"{r['id']} | account: {r['account']} | category: {r['category']} | "
        f"amount: {r['amount']:.2f} | region: {r['region']} | date: {r['date']}" for r in recs)


def build_block_prompt(task: dict, recs: list) -> str:
    """Same wording as the whole-context FILTER prompt -- only the record
    list is shorter, so the comparison isolates context length."""
    return (
        f"База транзакций:\n{render(recs)}\n\n"
        f"Найди ID всех транзакций, у которых category = \"{task['target_category']}\" "
        f"И region = \"{task['target_region']}\".\n"
        "Выведи ТОЛЬКО список подходящих ID через запятую, без пояснений. "
        "Если подходящих нет, выведи слово НЕТ."
    )


def seed_for(tid: str, bi: int, m: str) -> int:
    return SEED_BASE + int(hashlib.sha256(f"{tid}|{bi}|{m}".encode()).hexdigest(), 16) % 10 ** 6


def load_whole(ids, model_ids) -> dict:
    votes = {t: {} for t in ids}
    for line in (METRICS / "filter_calls.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["family"] == "E1-F2" and r["task_id"] in votes:
            votes[r["task_id"]][r["model"]] = {"extracted": frozenset(r["extracted"])}
    return votes


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    model_ids = list(MR.MODEL_IDS)
    d = TF2.build_tasks()
    tasks_by_id, ids = d["tasks"], d["test"]
    whole = load_whole(ids, model_ids)
    nblocks = len(blocks_of(tasks_by_id[ids[0]]))
    print(f"K1: {len(ids)} tasks x {nblocks} blocks x {len(model_ids)} models = "
         f"{len(ids)*nblocks*len(model_ids)} calls\n", flush=True)

    # ---------------- executors ----------------
    raw_votes = {t: {m: set() for m in model_ids} for t in ids}
    vdp_dropped = 0
    t0 = time.time()
    for model_id in model_ids:
        llm, _ = MR.load_model(model_id)
        for tid in ids:
            task = tasks_by_id[tid]
            for bi, recs in enumerate(blocks_of(task)):
                allowed = {r["id"] for r in recs}
                r = MR.generate(llm, model_id, build_block_prompt(task, recs),
                                temperature=TEMPERATURE, top_p=1.0, seed=seed_for(tid, bi, model_id),
                                max_tokens=MAX_TOKENS)
                txt = r.get("raw_text", "")
                got = CF.OR.extract_ids(txt)
                # VDP: an id outside this block is a hallucination -> drop
                kept = got & allowed
                vdp_dropped += len(got - allowed)
                raw_votes[tid][model_id] |= kept
                call_log.log_filter_call(model=model_id, task_id=f"{tid}#b{bi}", family="K1-block",
                                        seed=seed_for(tid, bi, model_id),
                                        n_in=r.get("input_tokens") or 0,
                                        n_out=r.get("output_tokens") or 0, ms=0.0,
                                        failed=bool(r.get("generation_failed")),
                                        extracted=sorted(kept), raw_text=txt)
        del llm
        print(f"  [{model_id}] done  ({time.time()-t0:.0f}s elapsed)", flush=True)
    print(f"\n  VDP dropped {vdp_dropped} out-of-block ids (hallucinations)\n", flush=True)

    votes = {t: {m: {"extracted": frozenset(raw_votes[t][m])} for m in model_ids} for t in ids}

    # ---------------- recall by original position ----------------
    def recall_by_pos(v):
        pos = defaultdict(lambda: [0, 0])
        for tid in ids:
            task = tasks_by_id[tid]
            gold = set(task["matched_ids"])
            for p, rec in enumerate(task["records"]):
                if rec["id"] not in gold:
                    continue
                for m in model_ids:
                    pos[p][1] += 1
                    pos[p][0] += int(rec["id"] in v[tid][m]["extracted"])
        return {p: c / n for p, (c, n) in sorted(pos.items()) if n}

    rw, rk = recall_by_pos(whole), recall_by_pos(votes)
    K = len(tasks_by_id[ids[0]]["records"])
    tail = list(range(12, K))
    head = list(range(0, 6))
    tail_w = statistics.mean(rw[p] for p in tail if p in rw)
    tail_k = statistics.mean(rk[p] for p in tail if p in rk)
    head_w = statistics.mean(rw[p] for p in head if p in rw)
    head_k = statistics.mean(rk[p] for p in head if p in rk)
    print("--- recall by original position: whole-context -> Kahn blocks ---")
    for p in range(K):
        if p in rw and p in rk:
            print(f"  pos {p:2d}   {rw[p]:.3f} -> {rk[p]:.3f}   {rk[p]-rw[p]:+.3f}")
    print(f"\n  head 0-5   {head_w:.3f} -> {head_k:.3f}")
    print(f"  tail 12-17 {tail_w:.3f} -> {tail_k:.3f}   [K1-recall bar {THR_RECALL}]")
    print(f"  gap head-tail {head_w-tail_w:.3f} -> {head_k-tail_k:.3f}   [K1-flat bar 0.12]")

    # ---------------- assembly + decode ----------------
    def decode_all(v):
        n_ok = 0
        per = {}
        for tid in ids:
            task = tasks_by_id[tid]
            cids = CF.candidate_ids(task)
            classes = SD.enumerate_classes(task)
            rel = CF.loo_reliabilities(tasks_by_id, ids, model_ids, v, tid)
            adm = sorted(m for m in model_ids if rel[m] > PI_ADMIT)
            sup = SD.swarm_support(task, v[tid], adm)
            _k, dec, _s = SD.decode(sup, cids, classes)
            ok = dec == frozenset(task["matched_ids"])
            n_ok += int(ok)
            per[tid] = {"decoded": sorted(dec), "ok": ok, "admitted": adm}
        return n_ok / len(ids), per

    set_k, per_k = decode_all(votes)
    print(f"\n--- exact-set after assembly + decoding: {set_k:.3f}   "
         f"(E4 whole-context baseline {BASE_E4_SET:.3f}, bar {THR_SET}) ---")

    # ---------------- emergence ----------------
    emergent = []
    for tid in ids:
        gold = frozenset(tasks_by_id[tid]["matched_ids"])
        any_model_whole = any(whole[tid][m]["extracted"] == gold for m in model_ids)
        if per_k[tid]["ok"] and not any_model_whole:
            emergent.append(tid)
    print(f"\n--- EMERGENCE: {len(emergent)} tasks correct after assembly where NO model "
         f"got the full set on whole context (bar >= {THR_EMERGENCE}) ---")
    print(f"    {emergent[:12]}")

    # ---------------- final chain ----------------
    fin = 0
    for tid in ids:
        task = tasks_by_id[tid]
        agg = DA.aggregate_det(frozenset(per_k[tid]["decoded"]), task["op"], task["id_to_amount"])
        fin += int(DA.derive_det(agg, task["threshold"], task["comparator"]) == task["final_oracle"])
    final_rate = fin / len(ids)

    v_recall = ("CONFIRMED" if tail_k >= THR_RECALL else
                "FALSIFIED" if tail_k < THR_RECALL_FALSIFY else "ABOVE_FLOOR_BELOW_PREDICTION")
    v_set = ("CONFIRMED" if set_k >= THR_SET else
             "FALSIFIED" if set_k < THR_SET_FALSIFY else "ABOVE_FLOOR_BELOW_PREDICTION")
    v_emg = "EMERGENCE_CONFIRMED" if len(emergent) >= THR_EMERGENCE else (
        "NO_EMERGENCE" if not emergent else "WEAK_EMERGENCE")

    out = {
        "STRUCTURE_DET": True, "n": len(ids), "block": BLOCK, "calls": len(ids)*nblocks*len(model_ids),
        "vdp_dropped": vdp_dropped,
        "recall_whole": rw, "recall_kahn": rk,
        "tail_whole": tail_w, "tail_kahn": tail_k, "head_whole": head_w, "head_kahn": head_k,
        "gap_whole": head_w - tail_w, "gap_kahn": head_k - tail_k,
        "exact_set_kahn": set_k, "exact_set_e4_whole": BASE_E4_SET,
        "final_yes_no": final_rate,
        "emergent_tasks": emergent, "n_emergent": len(emergent),
        "verdicts": {"K1_recall": v_recall, "K1_set": v_set, "K1_emergence": v_emg},
        "elapsed_sec": time.time() - t0,
    }
    METRICS.mkdir(parents=True, exist_ok=True)
    (METRICS / "k1_result.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    print(f"\n=== K1 ===")
    print(f"  recall tail   {tail_w:.3f} -> {tail_k:.3f}   [{v_recall}]")
    print(f"  exact-set     {BASE_E4_SET:.3f} -> {set_k:.3f}   [{v_set}]")
    print(f"  final yes/no  {final_rate:.3f}   (whole-task r_m* = 0.525)")
    print(f"  emergence     {len(emergent)} tasks   [{v_emg}]")
    print(f"  elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
