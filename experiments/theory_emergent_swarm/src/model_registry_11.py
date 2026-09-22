"""model_registry_11.py — thin loader onto `experiment11/configs/
model_registry.py`, by DIRECT FILE PATH (never a bare `import
model_registry`) — sys.path-sibling-collision lesson from LIFE-9/DELTA-0/
FORK-1, applied from the start."""

from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parents[1]
ROOT = AGENT_DIR.parent
_PATH = ROOT / "experiment11" / "configs" / "model_registry.py"


def _load_module(name: str, path: Path):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mr11 = _load_module("esw_model_registry11", _PATH)

MODEL_IDS = mr11.MODEL_IDS
load_model = mr11.load_model
generate = mr11.generate
