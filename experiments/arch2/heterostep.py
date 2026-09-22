"""heterostep.py — Complex Layer, пятый цикл: датасет/backend/реестр HETEROSTEP.

Offline-replay поверх УЖЕ СОБРАННОГО сырья эксп. 14: полная факториальная сетка
train (`experiment14/runs14/train_grid.json`, 6 моделей x 4 рода x 100 задач =
2400 ячеек, покрытие 100% -- проверено `Dataset.coverage()`) и, после фазы
`testgrid` в `experiment14/run_experiment14.py`, такая же сетка на held-out.

Зачем отдельный модуль, а не расширение `replay.py`: полигон другой (HETEROSTEP,
не MSARITH/код), швы другие (`experiment14/seams.py`, не experiment12/13), молекулы
другие (6 моделей эксп. 14, БЕЗ ролей same/other -- маршрутизация здесь называет
КОНКРЕТНУЮ модель на слот, это и есть весь смысл ASSEMBLE, SPEC.md §31/§33).

Что переиспользуется без копирования: `experiment14/seams.py::check_step` (тот же
приём `importlib` по явному пути, что уже применён в `registry.py`/`replay.py`),
`experiment14/tasks/heterostep.py` (`STEP_KINDS`, `build_tasks`, `split` -- ТЕ ЖЕ
генератор и разбиение train/test, что и в эксп. 14 -- иначе воспроизводимость
между полигоном эксп. 14 и этой кампанией разошлась бы).

Единственная измеренная точка сетки: prompt="step" (условное имя, не параметр
эксп. 14 -- там нет вариантов промпта), temperature=0.5, seed_slot=0,
max_tokens=60 (`experiment14/harness.py::MAX_TOKENS_STEP`). `train_grid.json`
хранит РОВНО один вызов на (task_id, kind, model), без развёртки по seed --
как и в `replay.py`, недоступная ячейка отбраковывается ДО единого вызова
модели, а не подделывается.

СИЛЬНОЕ ДОПУЩЕНИЕ, унаследованное из эксп. 14 (TASK_EXPERIMENT14.md §4): каждая
ячейка сетки записана с upstream, ПОДТВЕРЖДЁННЫМ оракулом (не собственным ответом
модели на предыдущий шаг). Replay здесь поэтому точен ТОЛЬКО в ветке A проекта
(декомпозиция и проверка заданы) -- то же ограничение, что уже зафиксировано в
`REPORT_EXPERIMENT14.md` §6 для G3'/OQ-4, не новое, не ослабленное и не усиленное
этим модулем.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Optional

import runner as R
from replay import _load_module

ROOT = Path(__file__).resolve().parents[1]
EXP14 = ROOT / "experiment14"

TRAIN_GRID = EXP14 / "runs14" / "train_grid.json"
TEST_GRID = EXP14 / "runs14" / "test_grid.json"          # фаза testgrid (GPU), см. §5 плана

TEMPERATURE = 0.5
MAX_TOKENS_STEP = 60              # experiment14/harness.py::MAX_TOKENS_STEP
PROMPT_VARIANT = "step"           # единственная измеренная точка; условное имя

_seams14 = _load_module("arch2_seams14", EXP14 / "seams.py")
_tasks14 = _load_module("arch2_tasks14", EXP14 / "tasks" / "heterostep.py")
_mr14 = _load_module("arch2_model_registry14", EXP14 / "configs" / "model_registry.py")

MODEL_IDS = tuple(_mr14.MODEL_IDS)
STEP_KINDS = tuple(_tasks14.STEP_KINDS)      # ("READ", "FORMAT", "LOOKUP", "COMPUTE")

# Верхняя оценка длины промпта для BUDGET, когда бэкенд не знает точной длины
# (`estimate()` здесь ТОЧЕН всегда, см. ниже -- эта константа используется только
# валидатором genotype.py как консервативная граница до первого вызова). Реальные
# input_tokens в train_grid.json лежат в диапазоне до нескольких сотен (запись
# HETEROSTEP короче полного prompt эксп. 12); 900 -- заведомый запас.
MAX_INPUT_TOKENS = 900

_GENERATOR_GRID = {
    "prompt": (PROMPT_VARIANT,), "temperature": (TEMPERATURE,),
    "seed_slot": (0,), "max_tokens": (MAX_TOKENS_STEP,),
}

SEAM_ID = "seam.heterostep"


# -- датасет ---------------------------------------------------------------------------


class Dataset:
    """Индекс сырья эксп. 14: (task_id, kind, model) -> запись ячейки сетки."""

    def __init__(self, train_path: Path = TRAIN_GRID, test_path: Path = TEST_GRID):
        self.tasks = _tasks14.build_tasks()
        self.split = _tasks14.split(list(self.tasks))

        self.cells: dict = {}
        self._load_grid(train_path)

        self.has_test = test_path.exists()
        if self.has_test:
            self._load_grid(test_path)

        self.models = MODEL_IDS
        self.kinds = STEP_KINDS

    def _load_grid(self, path: Path) -> None:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for r in data["rows"]:
            self.cells[(r["task_id"], r["kind"], r["model"])] = r

    def initial_state(self, task_id: str) -> R.State:
        """Начальное состояние для ЦЕЛОЙ задачи. Все 4 шага видны ASSEMBLE через
        `task["steps"]`; заужение на конкретный шаг делает сам `_exec_assemble`
        (SPEC.md §31) -- этот датасет не готовит узкие состояния заранее. Имя
        метода -- ОБЩИЙ интерфейс `evolve.Evolution.evaluate` (SPEC.md §33): нет
        понятия "состояние коллизии" здесь, задача просто ещё не начата."""
        task = self.tasks[task_id]
        return R.State(task_id=task_id, task_family="heterostep", task=task,
                       origin_model=None, artifact="", evidence=[])

    state_for = initial_state    # человекочитаемый алиас для прямых вызовов вне evolve.py

    def coverage(self, task_ids, kinds=None, models=None) -> dict:
        """Доля покрытых ячеек. Гейт a0 (план §6): кампания не начинается при
        покрытии < 100% -- иначе offline-результат был бы недостоверен молча."""
        kinds = kinds or self.kinds
        models = models or self.models
        total = len(task_ids) * len(kinds) * len(models)
        found = sum(1 for t in task_ids for k in kinds for m in models
                   if (t, k, m) in self.cells)
        return {"total": total, "found": found,
                "coverage": found / total if total else 0.0}


# -- backend ----------------------------------------------------------------------------


class HeterostepBackend:
    """Backend для Runner'а. `estimate()` ТОЧЕН (train/test-сетка хранит фактические
    usage-токены), как и в `replay.ReplayBackend` -- BUDGET-рамки считаются по факту."""

    def __init__(self, dataset: Dataset):
        self.ds = dataset
        self.n_served = 0
        self.n_unavailable = 0
        self.unavailable_reasons: dict = {}

    def _lookup(self, *, model_id, prompt_variant, temperature, seed_slot, max_tokens, state):
        if prompt_variant != PROMPT_VARIANT:
            return None, f"prompt {prompt_variant!r} not measured (only {PROMPT_VARIANT!r})"
        if temperature != TEMPERATURE:
            return None, f"temperature {temperature} not measured (only {TEMPERATURE})"
        if seed_slot != 0:
            return None, "only seed_slot=0 was measured (grid has no seed sweep)"
        if max_tokens != MAX_TOKENS_STEP:
            return None, f"max_tokens {max_tokens} not measured (only {MAX_TOKENS_STEP})"
        step = (state.task or {}).get("step")
        if step is None:
            return None, "CALL outside an ASSEMBLE slot has no step view -- nothing to look up"
        kind = step.get("kind")
        rec = self.ds.cells.get((state.task_id, kind, model_id))
        if rec is None:
            return None, f"no recorded cell for ({state.task_id}, {kind}, {model_id})"
        return rec, None

    def estimate(self, **kw) -> Optional[int]:
        rec, _ = self._lookup(**{k: v for k, v in kw.items() if k != "seed_vector"})
        return int(rec["tokens"]) if rec else 0

    def call(self, *, model_id, prompt_variant, temperature, seed_slot, max_tokens, state, seed_vector):
        rec, reason = self._lookup(model_id=model_id, prompt_variant=prompt_variant,
                                   temperature=temperature, seed_slot=seed_slot,
                                   max_tokens=max_tokens, state=state)
        if rec is None:
            self.n_unavailable += 1
            self.unavailable_reasons[reason] = self.unavailable_reasons.get(reason, 0) + 1
            return {"available": False, "reason": reason}
        self.n_served += 1
        return {
            "raw_text": rec.get("raw_output", ""),
            # эксп. 14 не делит вход/выход в сетке -- `tokens` уже сумма (harness.py).
            # Записано целиком в input_tokens, чтобы `spent = in+out` не удвоился.
            "input_tokens": rec.get("tokens", 0), "output_tokens": 0,
            "latency": rec.get("latency", 0.0), "available": True,
            "generation_failed": False, "source": "heterostep_grid",
        }

    def feasible_params(self, family: str) -> dict:
        return dict(_GENERATOR_GRID)


# -- шов ---------------------------------------------------------------------------------


def _seam_heterostep(text, task):
    """Обёртка над `experiment14.seams.check_step` -- ПЕРЕИСПОЛЬЗУЕТ, не повторяет,
    ту же логику numeric/token-шва и то же правило двух ярусов извлечения, что и
    в эксп. 14 (`seams.py` докстринг). `task["step"]` -- узкий вид, подставленный
    `_exec_assemble` (SPEC.md §31); отсутствие -- CHECK вне слота ASSEMBLE, не
    предусмотренное использование, не крах."""
    step = (task or {}).get("step")
    if step is None:
        return {"status": R.INAPPLICABLE, "collision_type": None,
                "details": {"reason": "no step view: CHECK used outside an ASSEMBLE slot"}}
    result = _seams14.check_step(step, text)
    return {"status": result["status"], "collision_type": None,
            "details": {"extracted": result.get("extracted"), "oracle": result.get("oracle"),
                        "reason": result.get("reason")}}


# Контрольные случаи ОБЁРТКИ (известный ответ по контракту), а не переоткрытие
# правильности numeric_seam/token_seam -- та уже покрыта 20/20 случаями в
# experiment14/seams.py::self_test(), переиспользуется, не повторяется здесь
# (тот же принцип, что registry.py применяет к швам experiment12/13).
_CONTROL_STEP_NUMERIC = {"name": "x", "kind": "READ", "seam": "numeric", "oracle": 42.0}
_CONTROL_STEP_TOKEN = {"name": "d", "kind": "FORMAT", "seam": "token", "oracle": "2026-03-07"}

_CONTROL_CASES = [
    ("PASS через numeric -> PASS", {"step": _CONTROL_STEP_NUMERIC}, "42", R.PASS),
    ("FAIL через numeric -> FAIL", {"step": _CONTROL_STEP_NUMERIC}, "41", R.FAIL),
    ("неоднозначный numeric -> INAPPLICABLE",
     {"step": _CONTROL_STEP_NUMERIC}, "40\n41", R.INAPPLICABLE),
    ("PASS через token -> PASS", {"step": _CONTROL_STEP_TOKEN}, "2026-03-07", R.PASS),
    ("неверный формат token -> INAPPLICABLE",
     {"step": _CONTROL_STEP_TOKEN}, "07 March 2026", R.INAPPLICABLE),
    ("нет узкого вида шага -> INAPPLICABLE, не крах", {}, "42", R.INAPPLICABLE),
]


def self_test_seam() -> dict:
    failures = []
    for label, task, text, expected in _CONTROL_CASES:
        try:
            got = _seam_heterostep(text, task)["status"]
        except Exception as exc:                                    # noqa: BLE001
            failures.append(f"{label}: raised {type(exc).__name__}: {exc}")
            continue
        if got != expected:
            failures.append(f"{label}: expected {expected}, got {got}")
    return {"passed": not failures, "n_cases": len(_CONTROL_CASES), "failures": failures}


# -- реестр ------------------------------------------------------------------------------


class Registry:
    """Duck-typed реестр для HETEROSTEP: 6 конкретных моделей + один шов. БЕЗ ролей
    same/other (в отличие от `registry.Registry`) -- маршрутизация здесь называет
    модель НАПРЯМУЮ, это и есть содержание тезиса эксп. 14/G1."""

    def __init__(self, *, run_self_tests: bool = True):
        self._molecules: dict = {}
        for mid in MODEL_IDS:
            self._molecules[f"gen.{mid}"] = {
                "id": f"gen.{mid}", "kind": "generator", "model_id": mid,
                "family": mid, "iface_in": "state", "iface_out": "state",
                "params_grid": _GENERATOR_GRID,
            }
        self._molecules[SEAM_ID] = {
            "id": SEAM_ID, "kind": "seam", "iface_in": "state", "iface_out": "evidence",
            "params_grid": None, "impl": _seam_heterostep,
        }
        self._self_test = self_test_seam() if run_self_tests else {"passed": False, "failures": ["not run"]}

    # -- интерфейс для валидатора --
    def has_molecule(self, mid) -> bool:
        return mid in self._molecules

    def kind_of(self, mid):
        m = self._molecules.get(mid)
        return m["kind"] if m else None

    def params_grid(self, mid):
        m = self._molecules.get(mid)
        return m.get("params_grid") if m else None

    def max_input_tokens(self, mid, prompt: str = PROMPT_VARIANT) -> int:
        return MAX_INPUT_TOKENS

    # -- интерфейс для Runner --
    def get(self, mid) -> dict:
        return self._molecules[mid]

    def is_trusted(self, seam_id: str) -> bool:
        return bool(self._self_test.get("passed"))

    def self_test_report(self) -> dict:
        return {SEAM_ID: dict(self._self_test)}

    def observe(self, obs_name: str, state):
        # HETEROSTEP-генотипы не используют SWITCH (SPEC.md §33: маршрут выражается
        # структурой ASSEMBLE+CALL, не диспетчером по наблюдаемой) -- намеренный отказ,
        # не пропуск: наблюдаемые из OBSERVABLES (task_family и т.д.) не осмысленны
        # для этого полигона (нет коллизий/ретраев, чей контекст они кодируют).
        raise NotImplementedError(f"SWITCH({obs_name}) не предусмотрен для HETEROSTEP")

    def observable_names(self) -> tuple:
        """Пустой список -- `evolve.Evolution.reproduce` (SPEC.md §33) читает его
        через `getattr(registry, "observable_names", None)` вместо старого хардкода
        `list(genotype.OBSERVABLES)`, и тогда M4 (SWITCH-мутации) и ветка "switch" в
        `mutate._random_node` перестают производить SWITCH-узлы для этого реестра --
        не тихо игнорируются, а честно не могут возникнуть: `observe()` выше в любом
        случае упал бы, если бы такой узел дожил до исполнения."""
        return ()

    def resolve_model(self, molecule_id: str, state, seed_slot: int) -> Optional[str]:
        return self._molecules[molecule_id]["model_id"]

    def is_composite(self, mid: str) -> bool:
        m = self._molecules.get(mid) or {}
        return m.get("role") == "composite"

    # -- автокатализ (SPEC.md §10/§17): не витрина цикла, но нужен, чтобы
    # `evolve.Evolution.reproduce`/`_try_register_molecule` не падали -- вся
    # машинерия эволюции переиспользуется ЦЕЛИКОМ, не наполовину.
    def register_composite(self, complex_id: str, genotype: dict, cost_profile: dict) -> str:
        mid = f"cx.{complex_id}"
        if mid in self._molecules:
            return mid
        self._molecules[mid] = {
            "id": mid, "kind": "generator", "role": "composite", "model_id": None,
            "family": f"composite:{complex_id}", "iface_in": "state", "iface_out": "state",
            "params_grid": {}, "genotype": copy.deepcopy(genotype),
            "cost_profile": dict(cost_profile), "registered_from": complex_id,
        }
        return mid

    def composite_ids(self) -> tuple:
        return tuple(mid for mid, m in self._molecules.items() if m.get("role") == "composite")

    def generator_ids(self) -> tuple:
        # Динамически, как `registry.Registry.generator_ids` -- включает и
        # зарегистрированные композиты (kind="generator", role="composite"), не
        # только 6 исходных моделей, иначе новый композит был бы невидим M1.
        return tuple(mid for mid, m in self._molecules.items() if m["kind"] == "generator")

    def seam_ids(self) -> tuple:
        return (SEAM_ID,)

    def family_of(self, molecule_id: str):
        m = self._molecules.get(molecule_id) or {}
        return m.get("family")


_DATASET: Optional[Dataset] = None
_REGISTRY: Optional[Registry] = None


def default_dataset() -> Dataset:
    global _DATASET
    if _DATASET is None:
        _DATASET = Dataset()
    return _DATASET


def default_registry() -> Registry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = Registry()
    return _REGISTRY


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ds = default_dataset()
    reg = default_registry()
    print(f"задач        : {len(ds.tasks)} (train {len(ds.split['train'])}, "
          f"test {len(ds.split['test'])})")
    print(f"ячеек сетки  : {len(ds.cells)}")
    print(f"train покрытие: {ds.coverage(ds.split['train'])}")
    if ds.has_test:
        print(f"test  покрытие: {ds.coverage(ds.split['test'])}")
    else:
        print("test-сетка ещё не сгенерирована (experiment14 --phase testgrid)")
    print(f"шов HETEROSTEP: {reg.self_test_report()}")
