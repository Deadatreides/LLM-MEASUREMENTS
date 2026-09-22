"""model_registry_11.py — thin loader onto `experiment11/configs/
model_registry.py`, by DIRECT FILE PATH (never a bare `import
model_registry`) — this session already hit the sys.path-sibling-
collision bug class twice (LIFE-8's `ceiling.py`, LIFE-9's `run_all.py`
re-hitting it); loading by path from the very first use avoids the whole
class of bug rather than fixing it after the fact.
"""

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


mr11 = _load_module("delta0_model_registry11", _PATH)

MODEL_IDS = mr11.MODEL_IDS
load_model = mr11.load_model
generate = mr11.generate
