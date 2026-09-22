"""unfold_metrics.py — Imp core, копия (не импорт) `agent_life4_block_fix/
src/metrics_lib.py`'s unfold-aware functions (`slot_models` recursively
разворачивает `cx.`-composite-CALL через `registry.get(mid)["genotype"]
["root"]`, семантика разворота = `arch2/runner.py`'s исполнение
composite-CALL: весь `root`, не один слот). Формула Imp/A/AB САМА не
менялась с LIFE-1 (PROTOCOL.md §2).

Не скопированы (LIFE-3/4-специфичны, не нужны для карты слотов LIFE-5):
`compute_block_a1_a5`, `audit_cell`, `unfolded_block_signatures` (A1-A5 —
про блоки/автокатализ, LIFE-5 §1 явно не строит A1-A5).

`build_audit_registry`/`_AuditRegistry` — read-only резолвер `.get(mid)->
{"genotype":{"root":...}}` поверх уже СОХРАНЁННЫХ `archive.json`+
`blocks.jsonl` одной клетки (реальный in-memory registry никогда не
сериализуется — подтверждено LIFE-4). `slot_map.py` строит один такой
резолвер НА КАЖДЫЙ из 24 прогонов (blocks.jsonl пуст для всех, кроме
life3 seed 601 и life4 seed 706 — см. LIFE-3/4 BLOCKERS.md), не один общий
на весь пул (носители/carrier_cid уникальны только внутри своего прогона).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import live_dataset as LD   # noqa: E402

HSTEP = LD.HSTEP
H = LD.H

STEP_KINDS = HSTEP.STEP_KINDS

MAX_UNFOLD_DEPTH = 6


def slot_models(slot_node: dict, registry, _depth: int = 0, _seen: frozenset = frozenset()) -> list:
    """`CALL(cx.*)` разворачивается в `registry.get(mid)["genotype"]
    ["root"]` и обходится рекурсивно (весь `root`, как исполнитель в
    `arch2/runner.py`). Возвращает СПИСОК (может содержать дубликаты,
    если одна модель встречается в нескольких ветках) -- дедуп, если
    нужен для сигнатуры, делает вызывающий код (`slot_map.py`), не эта
    функция (см. модульный docstring и LIFE-4 BLOCKERS.md "Находка 1")."""
    out: list = []

    def walk(n, depth, seen):
        if not isinstance(n, dict):
            return
        op = n.get("op")
        if op == "CALL":
            mol = n.get("molecule", "")
            if mol.startswith("gen."):
                out.append(mol[4:])
            elif mol.startswith("cx.") and depth < MAX_UNFOLD_DEPTH and mol not in seen:
                entry = registry.get(mol) if registry is not None else None
                inner_root = (entry or {}).get("genotype", {}).get("root")
                if inner_root is not None:
                    walk(inner_root, depth + 1, seen | {mol})
        elif op in ("SEQ", "PAR", "ASSEMBLE"):
            for c in n.get("children") or []:
                walk(c, depth, seen)
        elif op == "SWITCH":
            for c in (n.get("cases") or {}).values():
                walk(c, depth, seen)
            if "default" in n:
                walk(n["default"], depth, seen)
        elif op == "BUDGET":
            if "child" in n:
                walk(n["child"], depth, seen)

    walk(slot_node, _depth, _seen)
    return out


def best_single_per_slot(ds, panel: list) -> dict:
    out = {}
    for s, kind in enumerate(STEP_KINDS):
        best = 0
        for m in ds.models:
            n = sum(1 for t in panel if ds.cells[(t, kind, m)]["status"] == "PASS")
            best = max(best, n)
        out[s] = best
    return out


def covered_set(ds, panel: list, s: int, models: list) -> set:
    kind = STEP_KINDS[s]
    out: set = set()
    for m in models:
        out |= {t for t in panel if ds.cells[(t, kind, m)]["status"] == "PASS"}
    return out


def imp_set(ds, panel: list, best_single: dict, genotype: dict, registry) -> frozenset:
    root = genotype.get("root", {})
    children = root.get("children") or []
    out = set()
    for s in range(min(len(children), len(STEP_KINDS))):
        models = slot_models(children[s], registry)
        if not models:
            continue
        cov = covered_set(ds, panel, s, models)
        if len(cov) > best_single[s]:
            out.add(s)
    return frozenset(out)


def load_cell_artifacts(cell_dir: Path) -> dict:
    with open(cell_dir / "archive.json", encoding="utf-8") as f:
        archive = json.load(f)["genotypes"]
    genotypes = {g["complex_id"]: g for g in archive}

    with open(cell_dir / "summary.json", encoding="utf-8") as f:
        summary = json.load(f)

    def _read_jsonl(path):
        out = []
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    out.append(json.loads(line))
        return out

    births = _read_jsonl(cell_dir / "births.jsonl")
    blocks = _read_jsonl(cell_dir / "blocks.jsonl")
    edges = H.EdgeLog.from_jsonl(cell_dir / "heredity.jsonl")

    return {"genotypes": genotypes, "summary": summary, "births": births,
           "blocks": blocks, "edges": edges}


class _AuditRegistry:
    """Read-only резолвер поверх archive.json+blocks.jsonl ОДНОЙ клетки --
    см. модульный docstring. Копия `agent_life4_block_fix/src/
    metrics_lib.py::_AuditRegistry`, без изменений."""

    def __init__(self, genotypes: dict, blocks: list):
        self._molecules: dict = {}
        for cid, g in genotypes.items():
            self._molecules[f"cx.{cid}"] = {"genotype": g}
        for b in blocks:
            carrier = genotypes.get(b.get("carrier_cid"))
            if carrier is None:
                continue
            children = (carrier.get("root") or {}).get("children") or []
            slot = b.get("slot")
            if slot is None or slot >= len(children):
                continue
            self._molecules[b["molecule_id"]] = {"genotype": {"root": children[slot]}}

    def get(self, mid: str):
        return self._molecules.get(mid)


def build_audit_registry(genotypes: dict, blocks: list) -> _AuditRegistry:
    return _AuditRegistry(genotypes, blocks)


def death_gen_index(summary: dict) -> dict:
    out = {}
    for rep in summary.get("generations", []):
        gen = rep["generation"]
        for d in rep.get("deaths", []):
            cid = d["complex_id"]
            if cid not in out:
                out[cid] = gen
    return out
