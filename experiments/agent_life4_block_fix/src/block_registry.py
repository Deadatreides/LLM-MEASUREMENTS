"""block_registry.py — PROTOCOL.md §4.2: правила регистрации блока R1-R4
(N_MIN_USE=3, median L>=2, <=1 регистрация/поколение, MAX_BLOCKS=12) --
пороги НЕ изменены в LIFE-4 (не Change 1/2, см. PROTOCOL.md §0) -- копия
(не импорт) `agent_life3_block_live/src/block_registry.py`. Сам акт
регистрации переиспользует `arch2.heterostep.Registry.register_composite`
БЕЗ ИЗМЕНЕНИЙ -- блок оборачивается в самостоятельный генотип,
регистрируется под СОБСТВЕННЫМ complex_id.

LIFE-4 Change 1 (единственное изменение в этом файле): `block_signature`
теперь принимает `registry` и передаёт его в `metrics_lib.slot_models`,
чтобы вложенные `cx.`-композиты разворачивались при вычислении сигнатуры
блока -- иначе R1 (группировка по сигнатуре) продолжает молча
недооценивать содержимое слота, как это уже подтверждено в LIFE-3
(BLOCKERS.md "Находка 2").
"""

from __future__ import annotations

import copy
import json
import statistics
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import live_dataset as LD          # noqa: E402
import metrics_lib as ML           # noqa: E402

G = LD.G
H = LD.H
F = LD.F

N_MIN_USE = 3
MEDIAN_L_MIN = 2
MAX_BLOCKS = 12


def block_signature(slot_node: dict, registry) -> tuple:
    """PROTOCOL.md §4.2: (op, sorted(models)) -- та же `signature_s`, что
    LIFE-1/2/3, здесь используется как РЕГИСТРИРУЕМЫЙ ключ, не только для
    Imp. LIFE-4: `registry` обязателен -- разворачивает cx.-CALL внутри
    слота (Change 1)."""
    op = slot_node.get("op")
    models = tuple(sorted(ML.slot_models(slot_node, registry)))
    return (op, models)


def find_registrable_block(ev, already_registered: set, deaths_at: dict, last_gen: int) -> dict:
    """R1 (использование >=3) + R2 (медиана L >=2). Возвращает лучшего
    кандидата (R3: ровно один вызов -- ровно одна регистрация за раз) или
    None. Тай-брейк: наибольшее число носителей, затем меньший slot,
    затем алфавитно по сигнатуре. `ev.registry` передаётся в
    `block_signature` (Change 1) -- та же живая registry, что
    `run_generation`/`transfer_slot` уже используют."""
    reference_cids = set(ev.references)
    groups: dict = {}
    for cid, g in ev.archive.items():
        if cid in reference_cids:
            continue
        children = (g.get("root") or {}).get("children") or []
        for s, child in enumerate(children):
            sig = block_signature(child, ev.registry)
            if not sig[1]:
                continue
            key = (s, sig)
            if key in already_registered:
                continue
            groups.setdefault(key, []).append(cid)

    candidates = []
    for (slot, sig), cids in groups.items():
        cids = sorted(set(cids))
        if len(cids) < N_MIN_USE:
            continue
        lifespans = []
        for cid in cids:
            birth_gen = ev.genotypes[cid].get("gen", 0)
            death_gen = deaths_at.get(cid, last_gen)
            lifespans.append(max(0, death_gen - birth_gen))
        median_l = statistics.median(lifespans)
        if median_l < MEDIAN_L_MIN:
            continue
        candidates.append({"slot": slot, "signature": sig, "carriers": cids,
                           "n_use": len(cids), "median_L": median_l})

    if not candidates:
        return None
    candidates.sort(key=lambda c: (-c["n_use"], c["slot"], c["signature"]))
    return candidates[0]


def register_block(ev, candidate: dict, gen: int, block_registry: dict, blocks_path: Path) -> str:
    """Акт регистрации: оборачивает слот-поддерево в самостоятельный
    генотип, регистрирует как композит (`registry.register_composite`,
    БЕЗ ИЗМЕНЕНИЙ), дописывает `blocks.jsonl`."""
    carrier_cid = min(candidate["carriers"])   # детерминированно; сам блок идентичен у всех носителей
    carrier_g = ev.genotypes[carrier_cid]
    slot_subtree = copy.deepcopy(carrier_g["root"]["children"][candidate["slot"]])
    wrapped = G.genotype(slot_subtree, gen=gen, origin=f"block:slot{candidate['slot']}")

    results = F.load_results(ev.runs_dir, ev.experiment_id)
    per_task = results.get(carrier_cid, {})
    profile = H.cost_profile_from_traces(per_task)

    mid = ev.registry.register_composite(wrapped["complex_id"], wrapped, profile)

    op, models = candidate["signature"]
    entry = {"molecule_id": mid, "slot": candidate["slot"], "op": op, "models": list(models),
            "registered_gen": gen, "carrier_cid": carrier_cid,
            "n_use_at_registration": candidate["n_use"], "median_L_at_registration": candidate["median_L"]}
    block_registry[mid] = entry

    blocks_path.parent.mkdir(parents=True, exist_ok=True)
    with open(blocks_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return mid


def maybe_register_one_block(ev, block_registry: dict, blocks_path: Path, gen: int,
                             deaths_at: dict, last_gen: int) -> None:
    """R3 (<=1/поколение) + R4 (<=MAX_BLOCKS=12 всего). Вызывается один раз
    за поколение, ПОСЛЕ `run_generation` (использует уже посчитанные
    deaths/elite этого поколения), ДО `custom_reproduce` следующего --
    новый блок доступен `BLOCK_INSERT` начиная со следующего размножения
    (тот же порядок, что у arch2's собственной регистрации молекул)."""
    if len(block_registry) >= MAX_BLOCKS:
        return
    already = {(e["slot"], (e["op"], tuple(e["models"]))) for e in block_registry.values()}
    candidate = find_registrable_block(ev, already, deaths_at, last_gen)
    if candidate is None:
        return
    register_block(ev, candidate, gen, block_registry, blocks_path)
