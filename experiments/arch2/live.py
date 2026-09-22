"""live.py — живой backend и кампания E1: переносится ли отбор с replay на реальность.

E0 отбирает комплексы по СОХРАНЁННЫМ измерениям. Это опирается на допущение
**обменимости**: что исход плеча, измеренного как одиночный retry, тот же самый, когда
плечо стоит внутри более длинного комплекса. Данными это не проверяется — только
живым прогоном. Если знак и порядок величины не переносятся, replay как метод отбора
ОТВЕРГНУТ, и это результат, а не сбой (SPEC.md §12).

Отличия от replay, зафиксированные до прогона:
  * seed'ы 2000+slot — не пересекаются ни с фазой 0 (1-4), ни с retry (1000):
    E1 обязан быть НОВЫМ измерением, а не переигрыванием старого;
  * состояния берутся из held-out среза, не участвовавшего в E0;
  * ранняя остановка генерации (stop_predicate эксп. 13) НЕ используется — протокол
    генерации сохраняется тем же, что в эксп. 12, ради сопоставимости.

Промпты и загрузка моделей переиспользуются как есть: experiment11/configs/model_registry.py,
experiment12/context_builder.py, experiment12/tasks/*.generation_prompt.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

import fitness as F
import genotype as G
import registry as REG
import replay
import runner as R

ROOT = Path(__file__).resolve().parents[1]
ARCH2 = Path(__file__).resolve().parent
RUNS = ARCH2 / "runs"
METRICS = ARCH2 / "metrics"

LIVE_SEED_BASE = 2000
TOP_P = 1.0
# Компромисс между VRAM (6 ГБ на этой машине) и числом перезагрузок. Четыре модели
# класса 1-2B в q4/q5 занимают ~4 ГБ; на пилоте 48 вызовов дали 14 загрузок при кэше 3.
MODEL_CACHE_SIZE = 4


class LiveBackend:
    """Настоящие вызовы GGUF-моделей. Кэш моделей — LRU, число загрузок пишется в отчёт:
    если оно велико, это стоимость протокола, а не свойство комплексов."""

    def __init__(self, dataset, model_registry, context_builder):
        self.ds = dataset
        self.mr = model_registry
        self.ctx = context_builder
        self._cache: dict = {}
        self._order: list = []
        self.n_loads = 0
        self.n_calls = 0
        self.load_seconds = 0.0

    def _llm(self, model_id: str):
        if model_id in self._cache:
            self._order.remove(model_id)
            self._order.append(model_id)
            return self._cache[model_id]
        while len(self._order) >= MODEL_CACHE_SIZE:
            evict = self._order.pop(0)
            del self._cache[evict]
        llm, load_t = self.mr.load_model(model_id)
        self.load_seconds += load_t
        self.n_loads += 1
        self._cache[model_id] = llm
        self._order.append(model_id)
        return llm

    def _prompt(self, prompt_variant: str, state) -> Optional[str]:
        task = self.ds.raw_tasks[state.task_id]
        if prompt_variant == "primary":
            mod = self.ds.arith_module if state.task_family == "arithmetic" else self.ds.code_module
            return mod.generation_prompt(task)

        seam_result = None
        for ev in reversed(state.evidence):
            if ev.get("status") is not None:
                seam_result = ev
                break
        seam_result = seam_result or state.initial_seam_result
        if seam_result is None or not state.artifact:
            return None
        if prompt_variant == "promptB":
            return self.ctx.build_prompt_promptB(state.task_family, task, state.artifact, seam_result)
        return self.ctx.build_prompt(prompt_variant, state.task_family, task,
                                     state.artifact, seam_result)

    def estimate(self, **kw):
        return None            # живьём точная стоимость заранее неизвестна

    def call(self, *, model_id, prompt_variant, temperature, seed_slot, max_tokens, state, seed_vector):
        prompt = self._prompt(prompt_variant, state)
        if prompt is None:
            return {"available": False, "reason": f"prompt {prompt_variant} needs a prior artifact and evidence"}
        llm = self._llm(model_id)
        self.n_calls += 1
        raw = self.mr.generate(llm, model_id, prompt, temperature=temperature, top_p=TOP_P,
                               seed=LIVE_SEED_BASE + int(seed_slot), max_tokens=int(max_tokens))
        return {
            "raw_text": raw.get("raw_text", ""),
            "input_tokens": raw.get("input_tokens") or 0,
            "output_tokens": raw.get("output_tokens") or 0,
            "latency": raw.get("generation_time_sec", 0.0),
            "available": True,
            "generation_failed": bool(raw.get("generation_failed")),
            "source": f"live:{prompt_variant}",
        }

    def feasible_params(self, family: str) -> dict:
        return {"prompt": ("primary", "K0", "K1", "K2", "promptB"),
                "temperature": (0.0, 0.5, 1.0), "seed_slot": (0, 1, 2, 3),
                "max_tokens": (replay.REPLAY_MAX_TOKENS[family],)}


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    tmp.replace(path)


def main(args) -> int:
    import evolve as E
    import library

    source = args.source
    summary_path = METRICS / f"{source}_evolution.json"
    archive_path = RUNS / source / "archive.json"
    if not summary_path.exists() or not archive_path.exists():
        print(f"[e1] нет данных кампании {source} — сначала прогоните --phase e0")
        return 1
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)
    with open(archive_path, encoding="utf-8") as f:
        archive = {g["complex_id"]: g for g in json.load(f)["genotypes"]}

    ds = replay.default_dataset()
    reg = REG.default_registry()
    names = summary.get("names", {})
    last = summary["generations"][-1]

    # кого проверяем: топ по r из финальной элиты + все три эталона
    ranked = sorted(last["scores"].items(), key=lambda kv: (-kv[1]["r"], kv[1]["c"]))
    picked = [cid for cid, _ in ranked if cid in archive][:args.top]
    ref_ids = [cid for cid in last.get("references", {})]
    subjects = picked + ref_ids

    # held-out: состояния, не участвовавшие в E0
    import random

    pools = E.split_pools(ds.collision_ids(), random.Random(20260818 + 1))
    holdout = pools["holdout"][:args.n_tasks]

    print(f"[e1] живой перенос: {len(picked)} комплексов + {len(ref_ids)} эталонов "
          f"на {len(holdout)} held-out состояниях")
    print(f"[e1] seed'ы {LIVE_SEED_BASE}+slot (не пересекаются ни с фазой 0, ни с retry эксп.12)")

    mr = replay._load_module("arch2_live_model_registry",
                             ROOT / "experiment11" / "configs" / "model_registry.py")
    ctx = REG._ctx12
    be = LiveBackend(ds, mr, ctx)

    experiment_id = "e1"
    out_dir = RUNS / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    for cid in subjects:
        g = archive[cid]
        path = out_dir / f"{cid}.jsonl"
        done = set()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    done.add(json.loads(line)["task_id"])
        with open(path, "a", encoding="utf-8") as f:
            for tid in holdout:
                if tid in done:
                    continue
                res = R.run(g, ds.collision_state(tid), reg, be,
                            budget_tokens=E.BUDGET_PER_TASK)
                f.write(json.dumps({
                    "experiment_id": experiment_id, "phase": "e1:holdout",
                    "complex_id": cid, "task_id": tid, "outcome": res.outcome,
                    "cost": res.cost, "n_calls": res.n_calls,
                    "final_status": res.final_status, "stop_reason": res.stop_reason,
                    "exhausted": res.exhausted, "trace": res.trace,
                }, ensure_ascii=False) + "\n")
        print(f"  [{names.get(cid, cid)}] готово ({be.n_calls} вызовов, "
              f"{be.n_loads} загрузок моделей, {time.time() - t0:.0f} с)")

    live = F.load_results(RUNS, experiment_id)
    offline = F.load_results(RUNS, source)
    gate = last.get("gate_reference_id")

    rows = []
    for cid in subjects:
        m_live = F.metrics(live.get(cid, {}), holdout)
        d_live = F.paired_delta_r(live.get(cid, {}), live.get(gate, {}), holdout, n_boot=E.N_BOOT)
        e_live = F.bootstrap_efficiency(live.get(cid, {}), live.get(gate, {}), holdout, n_boot=E.N_BOOT)
        off = last["scores"].get(cid) or last.get("references", {}).get(cid) or {}
        vg = (last.get("vs_gate") or {}).get(cid, {})
        rows.append({
            "complex_id": cid, "name": names.get(cid, cid),
            "offline_r": off.get("r"), "offline_c": off.get("c"),
            "offline_delta_r_vs_gate": (vg.get("delta_r_vs_gate") or {}).get("point"),
            "live_r": m_live["r"], "live_c": m_live["c"],
            "live_delta_r_vs_gate": d_live["point"], "live_delta_ci": d_live["ci_95"],
            "live_efficiency_vs_gate": e_live.get("point"), "live_eff_ci": e_live.get("ci_95"),
        })

    report = {
        "experiment_id": experiment_id, "source": source, "n_holdout": len(holdout),
        "gate_reference_id": gate, "seed_base": LIVE_SEED_BASE,
        "n_live_calls": be.n_calls, "n_model_loads": be.n_loads,
        "load_seconds": be.load_seconds, "wall_seconds": time.time() - t0,
        "rows": rows,
    }
    _write(METRICS / "e1_transfer.json", report)

    print(f"\n[e1] перенос replay -> живой прогон ({len(holdout)} held-out состояний):")
    print("  offline_r  live_r   offline dR(B2)  live dR(B2)  CI95            комплекс")
    for r in sorted(rows, key=lambda x: -(x["live_r"] or 0)):
        od = r["offline_delta_r_vs_gate"]
        od_s = f"{od:+.3f}" if od is not None else "  -  "
        ci = r["live_delta_ci"]
        print(f"   {r['offline_r']:.3f}     {r['live_r']:.3f}      {od_s}       "
              f"{r['live_delta_r_vs_gate']:+.3f}   [{ci[0]:+.3f},{ci[1]:+.3f}]  {r['name']}")

    signs_kept = sum(1 for r in rows
                     if r["offline_delta_r_vs_gate"] is not None
                     and (r["offline_delta_r_vs_gate"] > 0) == (r["live_delta_r_vs_gate"] > 0))
    n_cmp = sum(1 for r in rows if r["offline_delta_r_vs_gate"] is not None)
    print(f"\n  знак ΔR относительно B2 сохранился у {signs_kept} из {n_cmp} комплексов")
    print(f"  живых вызовов {be.n_calls}, загрузок моделей {be.n_loads} "
          f"({be.load_seconds:.0f} с), всего {time.time() - t0:.0f} с")
    return 0
