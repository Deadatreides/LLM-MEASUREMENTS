"""model_registry_11.py — thin loader onto `experiment11/configs/
model_registry.py`, by DIRECT FILE PATH (never a bare `import
model_registry`) — same lesson applied from the start as DELTA-0/LIFE-9
(both hit the sys.path-sibling-collision bug class this session)."""

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


mr11 = _load_module("fork1_model_registry11", _PATH)

MODEL_IDS = mr11.MODEL_IDS
load_model = mr11.load_model
generate = mr11.generate
