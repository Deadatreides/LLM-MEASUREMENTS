"""fakes.py — игрушечный реестр и backend для контрольных случаев.

Существуют отдельно от `registry.py` намеренно: если Runner можно прогнать на
чужом реестре, значит он действительно не зависит от конкретных молекул
(I-18 CORE_IS_MODEL_AGNOSTIC в редакции этого слоя).
"""

from __future__ import annotations

PASS = "PASS"
FAIL = "FAIL"
INAPPLICABLE = "INAPPLICABLE"

GRID = {
    "prompt": ("primary", "K0"),
    "temperature": (0.0, 0.5),
    "seed_slot": (0, 1, 2, 3),
    "max_tokens": (150, 320),
}


def seam_says_pass(text, task):
    return {"status": PASS, "collision_type": None, "details": {"text": text}}


def seam_by_marker(text, task):
    """PASS, если в тексте есть маркер OK; иначе FAIL. Известный ответ по построению."""
    return ({"status": PASS, "collision_type": None, "details": {}} if "OK" in (text or "")
            else {"status": FAIL, "collision_type": "numeric", "details": {"text": text}})


class FakeRegistry:
    """Минимальный duck-typed реестр. `trusted` задаётся явно, чтобы проверить гейт I-15."""

    def __init__(self, *, trusted=True, max_input_tokens=100, seam_impl=seam_by_marker):
        self._trusted = trusted
        self._max_in = max_input_tokens
        self._seam_impl = seam_impl
        self.observed = []

    def has_molecule(self, mid):
        return mid.startswith("gen.") or mid.startswith("seam.")

    def kind_of(self, mid):
        return "generator" if mid.startswith("gen.") else "seam"

    def params_grid(self, mid):
        return GRID if mid.startswith("gen.") else None

    def max_input_tokens(self, mid, prompt="K0"):
        return self._max_in

    def get(self, mid):
        return {"id": mid, "kind": self.kind_of(mid), "impl": self._seam_impl}

    def is_trusted(self, seam_id):
        return self._trusted

    def resolve_model(self, molecule_id, state, seed_slot):
        if molecule_id == "gen.same_operator":
            return state.origin_model
        if molecule_id == "gen.other_operator":
            return "other-model"
        return molecule_id[len("gen."):]

    def observe(self, obs_name, state):
        self.observed.append(obs_name)
        if obs_name == "task_family":
            return state.task_family
        if obs_name == "attempts_used":
            return str(state.n_calls) if state.n_calls < 3 else "3+"
        if obs_name == "last_status":
            return state.evidence[-1]["status"] if state.evidence else "NONE"
        return "other"


class ScriptedBackend:
    """Отдаёт заранее заданные ответы по порядку вызовов. Полностью детерминирован —
    позволяет проверять гарантию 3 (детерминизм структуры) без модели."""

    def __init__(self, script, tokens=(50, 50)):
        self.script = list(script)
        self.tokens = tokens
        self.calls = []

    def call(self, *, model_id, prompt_variant, temperature, seed_slot, max_tokens, state, seed_vector):
        self.calls.append({"model_id": model_id, "prompt": prompt_variant, "seed_slot": seed_slot})
        idx = len(self.calls) - 1
        text = self.script[idx] if idx < len(self.script) else self.script[-1]
        return {
            "raw_text": text, "input_tokens": self.tokens[0], "output_tokens": self.tokens[1],
            "latency": 0.0, "available": True, "generation_failed": False, "source": "scripted",
        }
