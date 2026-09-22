"""registry.py — Complex Layer, уровень 0: Molecular Layer.

Молекулы — то, что УЖЕ измерено экспериментами 1-13. Здесь они только
объявляются, снабжаются конечной сеткой параметров, эмпирическим профилем
стоимости и проходят ГЕЙТ ДОВЕРИЯ. Ничего нового не изобретается: реализации
загружаются из experiment11/12/13 тем же приёмом importlib по явному пути,
который уже применён в проекте четырежды (коллизия имён модулей `seams`/`harness`).

Молекула не знает о существовании комплексов (SPEC.md §7.1).

ГЕЙТ ДОВЕРИЯ (I-15). Шов без пройденных контрольных случаев не получает
`trusted=True`, а значит не может: закрыть ветвь PAR, дать значение наблюдаемой,
объявить RESOLVED. Контрольные случаи написаны по ДОКУМЕНТИРОВАННОМУ контракту
шва (и по форме уже найденных в проекте дефектов извлечения), а не подогнаны под
фактический вывод — провал такого случая означает дефект шва, а не повод
поправить ожидание (SEAM_MODEL §5.3).

Отдельно: контрольные случаи покрывают и ИЗВЛЕЧЕНИЕ входов шва, а не только его
`check()` — прямое исполнение вывода этапа 8 arch1 («дыра в самой спеке»).
"""

from __future__ import annotations

import copy
import importlib.util
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Optional


def copy_genotype(g: dict) -> dict:
    return copy.deepcopy(g)

_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_with_seams_shim(name: str, path: Path, seams_mod):
    """`experiment12/outcome_classification.py` делает голый `from seams import ...`
    и рассчитывает, что нужный `seams` лежит на sys.path (так это работает внутри
    experiment12/13). Мы не хотим класть каталоги экспериментов на sys.path — там
    три разных модуля с именем `seams`. Поэтому нужный подставляется в sys.modules
    ровно на время загрузки и сразу снимается."""
    saved = sys.modules.get("seams")
    sys.modules["seams"] = seams_mod
    try:
        return _load(name, path)
    finally:
        if saved is None:
            sys.modules.pop("seams", None)
        else:
            sys.modules["seams"] = saved


_seams11 = _load("arch2_seams11", _ROOT / "experiment11" / "seams.py")
_seams12 = _load("arch2_seams12", _ROOT / "experiment12" / "seams.py")
_seams13 = _load("arch2_seams13", _ROOT / "experiment13" / "seams.py")
_outcome = _load_with_seams_shim("arch2_outcome", _ROOT / "experiment12" / "outcome_classification.py", _seams12)
_ctx12 = _load_with_seams_shim("arch2_ctx12", _ROOT / "experiment12" / "context_builder.py", _seams12)

PASS = _seams11.PASS
FAIL = _seams11.FAIL
INAPPLICABLE = _seams11.INAPPLICABLE
ERROR = _seams11.ERROR
UNKNOWN = _seams11.UNKNOWN

collision_subtype = _seams12.collision_subtype
classify_outcome = _outcome.classify_outcome
failure_signature = _outcome._failure_signature       # noqa: SLF001 -- единственная точка доступа к сигнатуре ошибки

# -- сетки параметров (SPEC.md §1.1: конечны и объявлены) -----------------------------

MODEL_IDS = (
    "llama-3.2-1b-instruct-q4_0",
    "qwen2.5-coder-1.5b-instruct-q4_0",
    "qwen3-1.7b-q4_0-unsloth",
    "internvl3-2b-q4_k_m",
    "gemma-3-it-1b-q5_k_s",
    "smollm2-1.7b-instruct-q4_k_m",
)
MODEL_FAMILY = {
    "llama-3.2-1b-instruct-q4_0": "llama-3.2",
    "qwen2.5-coder-1.5b-instruct-q4_0": "qwen2.5-coder",
    "qwen3-1.7b-q4_0-unsloth": "qwen3",
    "internvl3-2b-q4_k_m": "internvl3",
    "gemma-3-it-1b-q5_k_s": "gemma-3",
    "smollm2-1.7b-instruct-q4_k_m": "smollm2",
}

PROMPT_VARIANTS = ("primary", "K0", "K1", "K2", "promptB")
TEMPERATURES = (0.0, 0.5, 1.0)
SEED_SLOTS = (0, 1, 2, 3)
MAX_TOKENS_GRID = (150, 320, 400, 500)

_GENERATOR_GRID = {
    "prompt": PROMPT_VARIANTS,
    "temperature": TEMPERATURES,
    "seed_slot": SEED_SLOTS,
    "max_tokens": MAX_TOKENS_GRID,
}

# Эмпирические профили длины промпта, ИЗМЕРЕНЫ (не оценены) по
# experiment12/runs12/phase0_full_exp12-v1.json (initial_records, n=1320) и
# .../phase1_full_exp12-v2.json (arm_results, n=2226). Используются только для
# КОНСЕРВАТИВНОЙ верхней оценки перед BUDGET-проверкой; учёт стоимости всегда
# идёт по фактическим usage-токенам.
MAX_INPUT_TOKENS_BY_PROMPT = {
    "primary": 158,   # max по 6 моделям: 111-158
    "K0": 350,        # K0O0/K0O1 max 328/350
    "promptB": 333,
    "K1": 435,
    "K2": 435,
}
_SAFETY = 1.25  # запас на задачи вне измеренного датасета; консервативен намеренно

# Роли оператора: то, что РЕАЛЬНО измерялось в эксп. 12 (O0/O1), в отличие от
# «конкретная другая модель» — альтернативная модель там выбиралась случайно на
# состояние и записана в config.model_id. Роль честно отражает измеренную ось.
ROLE_SAME = "gen.same_operator"
ROLE_OTHER = "gen.other_operator"


# -- контрольные случаи швов (известный ответ ПО КОНТРАКТУ, не по прогону) ------------

_ARITH_STEPS = [
    {"name": "subtotal", "formula": "2 * 20", "value": 40.0},
    {"name": "total", "formula": "subtotal + 105", "value": 145.0},
]

_CODE_REQS = [
    {"name": "returns_sum", "description": "sums two ints", "check": "f(1, 2) == 3"},
    {"name": "handles_zero", "description": "zero is neutral", "check": "f(0, 5) == 5"},
]

_CONTROL_CASES = {
    "seam.multistep_arith": [
        # (описание, вход, ожидаемый статус)
        ("all steps correct -> PASS",
         "subtotal = 2 * 20 = 40\ntotal = 40 + 105 = 145", PASS),
        ("final step numerically wrong -> FAIL",
         "subtotal = 2 * 20 = 40\ntotal = 40 + 105 = 150", FAIL),
        ("final step line missing -> INAPPLICABLE (not FAIL)",
         "subtotal = 2 * 20 = 40", INAPPLICABLE),
        ("duplicate final step name -> INAPPLICABLE",
         "subtotal = 2 * 20 = 40\ntotal = 145\ntotal = 145", INAPPLICABLE),
        # Форма дефекта, найденного на этапе 8 arch1: несколько чисел без равенства.
        ("two numbers, no equation result -> INAPPLICABLE (not guessed)",
         "subtotal = 2 * 20 = 40\ntotal = 40 + 105", INAPPLICABLE),
    ],
    "seam.final_answer": [
        ("single correct FINAL ANSWER -> PASS", "FINAL ANSWER = 145", PASS),
        ("single wrong FINAL ANSWER -> FAIL", "FINAL ANSWER = 150", FAIL),
        ("no FINAL ANSWER line -> INAPPLICABLE", "the answer is 145", INAPPLICABLE),
        ("two FINAL ANSWER lines -> INAPPLICABLE",
         "FINAL ANSWER = 145\nFINAL ANSWER = 145", INAPPLICABLE),
    ],
    "seam.enriched_code": [
        ("correct implementation -> PASS",
         "```python\ndef f(a, b):\n    return a + b\n```", PASS),
        ("wrong implementation -> FAIL",
         "```python\ndef f(a, b):\n    return a - b\n```", FAIL),
        ("syntax error -> FAIL (not ERROR)",
         "```python\ndef f(a, b)\n    return a + b\n```", FAIL),
        ("wrong function name -> FAIL/structural",
         "```python\ndef g(a, b):\n    return a + b\n```", FAIL),
        ("no code at all -> INAPPLICABLE",
         "I think the answer is to add them.", INAPPLICABLE),
        ("raises at import time -> FAIL, never ERROR",
         "```python\nraise RuntimeError('boom')\ndef f(a, b):\n    return a + b\n```", FAIL),
    ],
}


def _seam_multistep_arith(text, task):
    return _seams12.multistep_arithmetic_seam(text, task["steps"])


def _seam_final_answer(text, task):
    return _seams13.final_answer_seam(text, task["answer"])


def _seam_enriched_code(text, task):
    return _seams12.enriched_code_seam(text, task["function_name"], task["requirements"])


_CONTROL_TASK = {
    "seam.multistep_arith": {"steps": _ARITH_STEPS},
    "seam.final_answer": {"answer": 145.0},
    "seam.enriched_code": {"function_name": "f", "requirements": _CODE_REQS},
}

SEAM_IMPL = {
    "seam.multistep_arith": _seam_multistep_arith,
    "seam.final_answer": _seam_final_answer,
    "seam.enriched_code": _seam_enriched_code,
}

# Какой шов применим к какому семейству задач и к какому контракту вывода.
SEAM_FOR = {
    ("arithmetic", "steps"): "seam.multistep_arith",
    ("arithmetic", "final"): "seam.final_answer",
    ("code", "any"): "seam.enriched_code",
}


def self_test_seam(seam_id: str) -> dict:
    """Прогоняет контрольные случаи. -> {'passed': bool, 'failures': [...]}.

    Исключение внутри шва — это ПРОВАЛ контрольного случая, а не ERROR наружу:
    иначе сломанный шов мог бы «пройти» гейт, ни разу не ответив.
    """
    impl = SEAM_IMPL[seam_id]
    task = _CONTROL_TASK[seam_id]
    failures = []
    for label, text, expected in _CONTROL_CASES[seam_id]:
        try:
            got = impl(text, task).get("status")
        except Exception as exc:                                   # noqa: BLE001
            failures.append(f"{label}: raised {type(exc).__name__}: {exc}")
            continue
        if got != expected:
            failures.append(f"{label}: expected {expected}, got {got}")
    return {"passed": not failures, "n_cases": len(_CONTROL_CASES[seam_id]), "failures": failures}


# -- наблюдаемые (SPEC.md §1.2) ---------------------------------------------------------

# Границы бакетов — произвольные константы, зафиксированы ДО прогона (SPEC.md §12).
_TOKENS_LOW = 250
_TOKENS_MID = 600
_DOMINANT_HIGH = 0.75        # порог эксп. 6 §2, не подбирался


def _last_seam(state) -> Optional[dict]:
    for ev in reversed(state.evidence):
        if ev.get("status") is not None:
            return ev
    return None


def obs_last_status(state) -> str:
    ev = _last_seam(state)
    return ev["status"] if ev else "NONE"


def obs_collision_subtype(state) -> str:
    ev = _last_seam(state)
    if ev is None:
        return "other"
    steps = state.task.get("steps") if state.task_family == "arithmetic" else None
    try:
        return collision_subtype(state.task_family, ev, steps) or "other"
    except Exception:                                              # noqa: BLE001
        return "other"


def obs_same_failure(state) -> str:
    ev = _last_seam(state)
    if ev is None or state.initial_seam_result is None:
        return "NA"
    try:
        cls = classify_outcome(state.initial_seam_result, ev)
    except Exception:                                              # noqa: BLE001
        return "NA"
    return {"SAME_FAILURE": "SAME", "DIFFERENT_FAILURE": "DIFFERENT"}.get(cls, "NA")


def obs_attempts_used(state) -> str:
    n = state.n_calls
    return str(n) if n < 3 else "3+"


def obs_tokens_bucket(state) -> str:
    if state.cost == 0:
        return "0"
    if state.cost < _TOKENS_LOW:
        return "low"
    if state.cost < _TOKENS_MID:
        return "mid"
    return "high"


def obs_task_family(state) -> str:
    return state.task_family


def _signatures(state) -> list:
    sigs = []
    for ev in state.evidence:
        if ev.get("status") in (PASS, FAIL, INAPPLICABLE):
            try:
                sigs.append(failure_signature(ev) if ev["status"] != PASS else ("PASS", None))
            except Exception:                                      # noqa: BLE001
                pass
    return sigs


def obs_dominant_share_bucket(state) -> str:
    sigs = _signatures(state)
    if len(sigs) < 2:
        return "NA"
    share = Counter(sigs).most_common(1)[0][1] / len(sigs)
    return "high" if share >= _DOMINANT_HIGH else "low"


def obs_n_unique_signatures(state) -> str:
    sigs = _signatures(state)
    if len(sigs) < 2:
        return "NA"
    n = len(set(sigs))
    return str(n) if n < 3 else "3+"


OBSERVABLE_IMPL = {
    "last_status": obs_last_status,
    "collision_subtype": obs_collision_subtype,
    "same_failure": obs_same_failure,
    "attempts_used": obs_attempts_used,
    "tokens_bucket": obs_tokens_bucket,
    "task_family": obs_task_family,
    "dominant_share_bucket": obs_dominant_share_bucket,
    "n_unique_signatures": obs_n_unique_signatures,
}


# -- реестр ------------------------------------------------------------------------------


class Registry:
    """Duck-typed объект, который получают genotype.validate() и runner.run()."""

    def __init__(self, *, run_self_tests: bool = True):
        self._molecules: dict = {}

        for mid in MODEL_IDS:
            self._molecules[f"gen.{mid}"] = {
                "id": f"gen.{mid}", "kind": "generator", "model_id": mid,
                "family": MODEL_FAMILY[mid], "role": "concrete",
                "iface_in": "state", "iface_out": "state", "params_grid": _GENERATOR_GRID,
            }
        for rid, role in ((ROLE_SAME, "same"), (ROLE_OTHER, "other")):
            self._molecules[rid] = {
                "id": rid, "kind": "generator", "model_id": None, "family": None,
                "role": role, "iface_in": "state", "iface_out": "state",
                "params_grid": _GENERATOR_GRID,
            }
        for sid in SEAM_IMPL:
            self._molecules[sid] = {
                "id": sid, "kind": "seam", "iface_in": "state", "iface_out": "evidence",
                "params_grid": None, "impl": SEAM_IMPL[sid],
            }

        self._self_tests = {}
        for sid in SEAM_IMPL:
            self._self_tests[sid] = self_test_seam(sid) if run_self_tests else {"passed": False, "failures": ["not run"]}

    # -- интерфейс для валидатора --
    def has_molecule(self, mid) -> bool:
        return mid in self._molecules

    def kind_of(self, mid):
        m = self._molecules.get(mid)
        return m["kind"] if m else None

    def params_grid(self, mid):
        m = self._molecules.get(mid)
        return m.get("params_grid") if m else None

    def max_input_tokens(self, mid, prompt: str = "K1") -> int:
        return int(MAX_INPUT_TOKENS_BY_PROMPT.get(prompt, max(MAX_INPUT_TOKENS_BY_PROMPT.values())) * _SAFETY)

    # -- интерфейс для Runner --
    def get(self, mid) -> dict:
        return self._molecules[mid]

    def is_trusted(self, seam_id: str) -> bool:
        """I-15: шов без пройденных контрольных случаев не даёт HARD-evidence."""
        return bool(self._self_tests.get(seam_id, {}).get("passed"))

    def self_test_report(self) -> dict:
        return dict(self._self_tests)

    def observe(self, obs_name: str, state):
        return OBSERVABLE_IMPL[obs_name](state)

    def resolve_model(self, molecule_id: str, state, seed_slot: int) -> Optional[str]:
        """Роль -> конкретная модель. `other_operator` выбирается детерминированно
        по (origin_model, task_id, seed_slot) — тем же принципом «одна альтернатива
        на состояние», что в experiment12/harness.py::build_arm_configs."""
        mol = self._molecules[molecule_id]
        if mol["role"] == "concrete":
            return mol["model_id"]
        origin = state.origin_model
        if mol["role"] == "same":
            return origin
        others = [m for m in MODEL_IDS if m != origin]
        if not others:
            return origin
        rng = random.Random(f"{origin}|{state.task_id}|{seed_slot}")
        return rng.choice(others)

    def seam_for(self, task_family: str, contract: str = "steps") -> str:
        if task_family == "code":
            return SEAM_FOR[("code", "any")]
        return SEAM_FOR[("arithmetic", contract)]

    # -- автокатализ: комплекс -> молекула (SPEC.md §10) --

    def register_composite(self, complex_id: str, genotype: dict, cost_profile: dict) -> str:
        """Замораживает прошедший гейт комплекс как молекулу-генератор.

        Никакого нового узла в языке не появляется: композит вызывается тем же CALL,
        а Runner, увидев role='composite', исполняет его корень в ТОМ ЖЕ контексте —
        общий бюджет, общая трасса. Чёрных ящиков нет (SPEC.md §10).
        """
        mid = f"cx.{complex_id}"
        if mid in self._molecules:
            return mid
        self._molecules[mid] = {
            "id": mid, "kind": "generator", "role": "composite", "model_id": None,
            "family": f"composite:{complex_id}", "iface_in": "state", "iface_out": "state",
            "params_grid": {},                      # у композита нет свободных параметров
            "genotype": copy_genotype(genotype), "cost_profile": dict(cost_profile),
            "registered_from": complex_id,
        }
        return mid

    def composite_ids(self) -> tuple:
        return tuple(mid for mid, m in self._molecules.items() if m.get("role") == "composite")

    def is_composite(self, mid: str) -> bool:
        m = self._molecules.get(mid) or {}
        return m.get("role") == "composite"

    def generator_ids(self) -> tuple:
        return tuple(mid for mid, m in self._molecules.items() if m["kind"] == "generator")

    def seam_ids(self) -> tuple:
        return tuple(SEAM_IMPL)

    def family_of(self, molecule_id: str):
        m = self._molecules.get(molecule_id) or {}
        return m.get("family") or m.get("role")


_DEFAULT: Optional[Registry] = None


def default_registry() -> Registry:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Registry()
    return _DEFAULT


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    reg = default_registry()
    print("seam self-tests (I-15 gate):")
    for sid, rep in reg.self_test_report().items():
        mark = "OK " if rep["passed"] else "FAIL"
        print(f"  [{mark}] {sid}: {rep['n_cases']} control cases")
        for f in rep["failures"]:
            print(f"         - {f}")
