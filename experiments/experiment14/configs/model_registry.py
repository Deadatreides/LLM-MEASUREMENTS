"""model_registry.py — тонкая обёртка над experiment11/configs/model_registry.py
(тот же приём, что experiment12/13: реестр экспериментов не дублируется)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "experiment11" / "configs" / "model_registry.py"
_spec = importlib.util.spec_from_file_location("experiment11_model_registry_for_exp14", _PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

MODEL_REGISTRY = _mod.MODEL_REGISTRY
MODEL_IDS = _mod.MODEL_IDS
N_CTX = _mod.N_CTX
load_model = _mod.load_model
generate = _mod.generate
