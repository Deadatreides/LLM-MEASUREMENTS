"""replay.py — offline-backend: CALL резолвится из УЖЕ СОБРАННОГО сырья.

Ноль вызовов модели. Это делает кампанию E0 бесплатной и повторяемой, а заодно
превращает валидатор в реальный экономический фильтр: генотип, требующий неизмеренной
ячейки, отбраковывается ДО любого расхода.

Что доступно (и, значит, что вообще выразимо в E0):

  primary CALL  -- первичная генерация с нуля: experiment12/runs12/phase0_full_exp12-v1.json,
                   `initial_records`, ключ (model, task_id, seed). 6 моделей x 55 задач x
                   4 seed = 1320 записей. seed_slot 0..3 -> seed 1..4.
  retry CALL    -- ремонт состояния коллизии: .../phase1_full_exp12-v2.json, `arm_results`,
                   ключ (collision_id, arm), arm = уровень контекста + режим оператора
                   (K0O0/K0O1/K1O0/K1O1/K2O0/K2O1/K0O0_promptB). 318 состояний x 7 плеч.

Три честных ограничения, которые НЕЛЬЗЯ обойти данными:

  1. Глубина ремонта = 1. Плечи фазы 1 стартуют от ИСХОДНОГО артефакта; данных о
     «ремонте ремонта» не существует. Второй retry подряд объявляется недоступным.
  2. Режим O1 привязан к КОНКРЕТНОЙ альтернативной модели, которую харнесс эксп. 12
     выбрал случайно на состояние (experiment12/harness.py::build_arm_configs).
     Поэтому retry возможен либо `gen.same_operator`, либо `gen.other_operator`;
     запрос конкретной третьей модели недоступен.
  3. temperature = 0.5 и max_tokens = 150/320 (арифметика/код) — единственные
     измеренные точки сетки. Остальные недоступны.

Ограничение, которое данными НЕ проверяется и потому остаётся допущением (SPEC.md §12):
**обменимость** — что исход плеча, измеренного как одиночный retry, тот же самый, когда
это плечо стоит внутри более длинного комплекса. Проверяет E1 на живой модели.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Optional

import runner as R

ROOT = Path(__file__).resolve().parents[1]

PHASE0 = ROOT / "experiment12" / "runs12" / "phase0_full_exp12-v1.json"
PHASE1 = ROOT / "experiment12" / "runs12" / "phase1_full_exp12-v2.json"

REPLAY_TEMPERATURE = 0.5                      # experiment12/harness.py::BASE_TEMPERATURE
REPLAY_MAX_TOKENS = {"arithmetic": 150, "code": 320}   # MAX_TOKENS_ARITH / MAX_TOKENS_CODE
SEED_BY_SLOT = (1, 2, 3, 4)                   # INITIAL_SEEDS
RETRY_PROMPTS = ("K0", "K1", "K2", "promptB")


def _load_module(name: str, path: Path, shim: Optional[tuple] = None):
    if shim:
        saved = sys.modules.get(shim[0])
        sys.modules[shim[0]] = shim[1]
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        if shim:
            if saved is None:
                sys.modules.pop(shim[0], None)
            else:
                sys.modules[shim[0]] = saved


def _load_tasks() -> dict:
    """Задачи эксп. 12: шаги арифметики и requirements кода — вход для швов."""
    exp12 = ROOT / "experiment12"
    pkg_path = str(exp12)
    if pkg_path not in sys.path:
        sys.path.insert(0, pkg_path)
    arith = _load_module("arch2_arith_tasks", exp12 / "tasks" / "arithmetic_multistep_tasks.py")
    code = _load_module("arch2_code_tasks", exp12 / "tasks" / "code_tasks.py")

    tasks, raw = {}, {}
    for tid, t in arith.TASKS.items():
        tasks[tid] = {"task_id": tid, "family": "arithmetic", "steps": t["steps"],
                      "answer": t["steps"][-1]["value"]}
        raw[tid] = t
    for tid, t in code.CODE_TASKS.items():
        tasks[tid] = {"task_id": tid, "family": "code", "function_name": t["function_name"],
                      "requirements": t["requirements"]}
        raw[tid] = t
    # `raw` нужен только живому бэкенду: `generation_prompt`/`build_prompt` эксп. 12
    # принимают ИСХОДНЫЙ словарь задачи, а не сокращённый вход шва.
    return tasks, raw, arith, code


class Dataset:
    """Индекс сохранённого сырья. Читается один раз, дальше — только поиск по ключу."""

    def __init__(self, phase0_path: Path = PHASE0, phase1_path: Path = PHASE1):
        with open(phase0_path, encoding="utf-8") as f:
            p0 = json.load(f)
        with open(phase1_path, encoding="utf-8") as f:
            p1 = json.load(f)

        self.tasks, self.raw_tasks, self.arith_module, self.code_module = _load_tasks()

        # первичные генерации: (model, task_id, seed) -> запись
        self.primary: dict = {}
        for r in p0["initial_records"]:
            self.primary[(r["model"], r["task_id"], r["seed"])] = r

        # состояния коллизии (только FAIL первичных): collision_id -> состояние
        self.states: dict = {s["collision_id"]: s for s in p0["states"]}

        # плечи ремонта: (collision_id, arm) -> запись
        self.arms: dict = {}
        self.arm_model: dict = {}
        for r in p1["arm_results"]:
            self.arms[(r["collision_id"], r["arm"])] = r
            self.arm_model[(r["collision_id"], r["arm"])] = r["config"]["model_id"]

        self.models = sorted({r["model"] for r in p0["initial_records"]})
        self.task_ids = sorted({r["task_id"] for r in p0["initial_records"]})

    # -- построение стартовых состояний --

    def collision_state(self, collision_id: str) -> R.State:
        s = self.states[collision_id]
        task = self.tasks[s["task_id"]]
        return R.State(
            task_id=s["task_id"], task_family=s["task_family"], task=task,
            origin_model=s["model_id"], artifact=s["initial_artifact"],
            evidence=[], initial_seam_result=s["initial_seam_result"],
            meta={"collision_id": collision_id, "origin_seed": s["seed"],
                  "provenance": "initial", "retry_used": False},
        )

    def initial_state(self, collision_id: str) -> R.State:
        """Алиас `collision_state` под именем, которое `evolve.Evolution.evaluate`
        теперь использует НЕЗАВИСИМО от датасета (пятый цикл, SPEC.md §33) --
        `heterostep.Dataset` не имеет понятия "состояние коллизии". `collision_state`
        сама НЕ переименована (используется по имени в 4 контрольных случаях
        tests/), так что этот метод существует ТОЛЬКО ради общего интерфейса."""
        return self.collision_state(collision_id)

    def fresh_state(self, task_id: str, origin_model: str) -> R.State:
        """Состояние ДО первой генерации: артефакт пуст, коллизии ещё нет."""
        task = self.tasks[task_id]
        return R.State(
            task_id=task_id, task_family=task["family"], task=task,
            origin_model=origin_model, artifact="", evidence=[], initial_seam_result=None,
            meta={"collision_id": None, "provenance": "empty", "retry_used": False},
        )

    def collision_ids(self, family: Optional[str] = None) -> list:
        return sorted(cid for cid, s in self.states.items()
                      if family is None or s["task_family"] == family)


class ReplayBackend:
    """Backend для Runner'а. `estimate()` точен — токены известны заранее, значит
    BUDGET-рамки в offline-режиме считаются по факту, не по верхней границе."""

    def __init__(self, dataset: Dataset):
        self.ds = dataset
        self.n_served = 0
        self.n_unavailable = 0
        self.unavailable_reasons: dict = {}

    # -- поиск записи --

    def _primary_seed(self, model_id: str, seed_slot: int, state) -> int:
        """`seed_slot` — это «какой ПО СЧЁТУ независимый розыгрыш», а не абсолютный seed.

        Розыгрыш, породивший текущее состояние коллизии, из выборки исключается: иначе
        `seed_slot=0` на состоянии с origin seed=1 переиграл бы сам себя и «слепой
        повтор» механически всегда давал бы ту же ошибку. Исключается ровно пара
        (origin_model, origin_seed) — у другой модели все четыре seed независимы.
        """
        origin_seed = state.meta.get("origin_seed")
        pool = list(SEED_BY_SLOT)
        if origin_seed is not None and model_id == state.origin_model:
            pool = [s for s in pool if s != origin_seed]
        return pool[seed_slot % len(pool)]

    def _lookup(self, *, model_id, prompt_variant, temperature, seed_slot, max_tokens, state):
        family = state.task_family
        if temperature != REPLAY_TEMPERATURE:
            return None, f"temperature {temperature} not measured (only {REPLAY_TEMPERATURE})"
        if max_tokens != REPLAY_MAX_TOKENS[family]:
            return None, f"max_tokens {max_tokens} not measured for {family}"

        if prompt_variant == "primary":
            seed = self._primary_seed(model_id, seed_slot, state)
            rec = self.ds.primary.get((model_id, state.task_id, seed))
            if rec is None:
                return None, f"no primary record for ({model_id}, {state.task_id}, seed{seed})"
            return {"kind": "primary", "record": rec, "seed": seed}, None

        if prompt_variant not in RETRY_PROMPTS:
            return None, f"unknown prompt variant {prompt_variant}"

        # retry возможен только из состояния коллизии, зафиксированного фазой 0
        cid = state.meta.get("collision_id")
        if cid is None:
            return None, "retry requires a recorded collision state (current artifact is not one)"
        if state.meta.get("retry_used"):
            return None, "repair depth > 1 was never measured (no retry-of-retry data)"

        origin_model = self.ds.states[cid]["model_id"]
        if model_id == origin_model:
            operator = "O0"
        else:
            expected = self.ds.arm_model.get((cid, "K0O1"))
            if expected is None or model_id != expected:
                return None, ("O1 arm was measured only for the alternative model recorded by "
                              f"experiment 12 ({expected}), not for {model_id}")
            operator = "O1"

        if prompt_variant == "promptB":
            if operator != "O0":
                return None, "promptB was measured only with the same operator (O0)"
            arm = "K0O0_promptB"
        else:
            arm = f"{prompt_variant}{operator}"

        rec = self.ds.arms.get((cid, arm))
        if rec is None:
            return None, f"no arm record for ({cid}, {arm})"
        return {"kind": "retry", "record": rec, "arm": arm, "collision_id": cid}, None

    # -- интерфейс backend'а --

    def estimate(self, **kw) -> Optional[int]:
        found, _ = self._lookup(**{k: v for k, v in kw.items() if k != "seed_vector"})
        if found is None:
            return 0                    # недоступная ячейка ничего не стоит
        rec = found["record"]
        if found["kind"] == "primary":
            return int(rec.get("total_tokens") or 0)
        return int(rec.get("tokens") or 0)

    def call(self, *, model_id, prompt_variant, temperature, seed_slot, max_tokens, state, seed_vector):
        found, reason = self._lookup(model_id=model_id, prompt_variant=prompt_variant,
                                     temperature=temperature, seed_slot=seed_slot,
                                     max_tokens=max_tokens, state=state)
        if found is None:
            self.n_unavailable += 1
            self.unavailable_reasons[reason] = self.unavailable_reasons.get(reason, 0) + 1
            return {"available": False, "reason": reason}

        self.n_served += 1
        rec = found["record"]
        if found["kind"] == "primary":
            text = rec["result"]
            in_tok, out_tok = int(rec.get("input_tokens") or 0), int(rec.get("output_tokens") or 0)
            cid = f"col:{model_id}:{state.task_family}:{state.task_id}:seed{found['seed']}"
            state.meta["collision_id"] = cid if cid in self.ds.states else None
            state.meta["provenance"] = "primary"
            state.meta["retry_used"] = False
        else:
            inner = rec.get("record") or {}
            text = inner.get("result", "")
            in_tok = int(inner.get("input_tokens") or 0)
            out_tok = int(inner.get("output_tokens") or 0)
            if in_tok + out_tok == 0:
                in_tok, out_tok = int(rec.get("tokens") or 0), 0
            state.meta["provenance"] = f"retry:{found['arm']}"
            state.meta["retry_used"] = True

        return {
            "raw_text": text, "input_tokens": in_tok, "output_tokens": out_tok,
            "latency": rec.get("latency", 0.0), "available": True,
            "generation_failed": bool((rec.get("record") or rec).get("generation_failed")),
            "source": found.get("arm", "phase0"),
        }

    # -- для mutate.py: какие точки сетки вообще осмысленны в offline-режиме --

    def feasible_params(self, family: str) -> dict:
        return {
            "prompt": ("primary",) + RETRY_PROMPTS,
            "temperature": (REPLAY_TEMPERATURE,),
            "seed_slot": tuple(range(len(SEED_BY_SLOT))),
            "max_tokens": (REPLAY_MAX_TOKENS[family],),
        }


_DATASET: Optional[Dataset] = None


def default_dataset() -> Dataset:
    global _DATASET
    if _DATASET is None:
        _DATASET = Dataset()
    return _DATASET


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ds = default_dataset()
    print(f"primary records : {len(ds.primary)}")
    print(f"collision states: {len(ds.states)}")
    print(f"retry arms      : {len(ds.arms)}")
    print(f"tasks           : {len(ds.tasks)}  models: {len(ds.models)}")
    fam = {}
    for s in ds.states.values():
        fam[s["task_family"]] = fam.get(s["task_family"], 0) + 1
    print(f"states by family: {fam}")
