"""STORAGE — уровень 0 (API_BOUNDARIES.md §11).

Append-only хранилище. Единственный метод записи — append. Методов
update и delete нет вообще и не будет: это жёсткое правило проекта
(arch1/CLAUDE.md), а не временное упущение.

Знает: как сохранить и прочитать запись, как построить снимок графа
по graph_version.
Не знает: ничего о структуре графа, о LLM, о политике, о стоимости.

Здесь же — общие типы уровня 0 (DATA_MODEL.md §0), нужные всем
вышележащим слоям: Provenance, Sampling, Confidence, генератор ID.

Гарантии:
- нет методов update/delete/set/remove;
- append с уже существующим record_id — ошибка (нельзя даже
  перезаписать тем же id);
- любое состояние графа восстановимо по graph_version (snapshot_at).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional


def new_id(kind: str) -> str:
    """ID := "<kind>:<uuid>" (DATA_MODEL.md §0)."""
    return f"{kind}:{uuid.uuid4()}"


def now_iso() -> str:
    """TIMESTAMP := ISO-8601 UTC (DATA_MODEL.md §0)."""
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Sampling:
    """SAMPLING (DATA_MODEL.md §0) — immutable целиком."""

    temperature: float
    top_p: float
    top_k: int
    seed: int
    max_tokens: int


@dataclass(frozen=True)
class Confidence:
    """CONFIDENCE (DATA_MODEL.md §0).

    basis="MODEL_REPORTED" никогда не участвует в решениях STATE/EVIDENCE
    (F9-F14) — хранится только как диагностика. Это правило соблюдают
    вышележащие слои; сам тип его не навязывает.
    """

    value: Optional[float] = None
    basis: str = "UNSET"  # MECHANICAL | STATISTICAL | MODEL_REPORTED | UNSET


@dataclass(frozen=True)
class Cost:
    """COST (DATA_MODEL.md §15).

    Живёт на уровне 0 рядом с остальными общими типами, а не в
    budget_manager.py (уровень 5): иначе уровень 5 импортировал бы
    соседа по своему же уровню, что запрещает API_BOUNDARIES.md §1.
    Внутренние вычисления (graph-анализ) заполняют time_sec при
    нулевых tokens — они тоже имеют стоимость (F41).
    """

    tokens: int = 0
    time_sec: float = 0.0
    calls: int = 0
    model_id: Optional[str] = None

    def __add__(self, other: "Cost") -> "Cost":
        return Cost(
            tokens=self.tokens + other.tokens,
            time_sec=self.time_sec + other.time_sec,
            calls=self.calls + other.calls,
            model_id=self.model_id if self.model_id == other.model_id else None,
        )

    def fits_in(self, budget: "Cost") -> bool:
        return (
            self.tokens <= budget.tokens
            and self.time_sec <= budget.time_sec
            and self.calls <= budget.calls
        )


@dataclass(frozen=True)
class Provenance:
    """PROVENANCE (DATA_MODEL.md §0) — все поля I."""

    created_by: str  # SYSTEM | MODEL | HUMAN | EXTERNAL
    action_id: Optional[str] = None
    run_id: Optional[str] = None
    model_id: Optional[str] = None
    prompt_profile_id: Optional[str] = None
    sampling_config: Optional[Sampling] = None
    input_refs: tuple[str, ...] = field(default_factory=tuple)
    parent_version: Optional[str] = None
    timestamp: str = field(default_factory=now_iso)


@dataclass(frozen=True)
class Record:
    """Универсальный конверт для любой append-only записи."""

    record_id: str
    kind: str
    seq: int
    payload: Any
    timestamp: str


class Storage:
    """Append-only хранилище (API_BOUNDARIES.md §11)."""

    def __init__(self) -> None:
        self._records: dict[str, Record] = {}
        self._order: list[str] = []
        self._seq = 0

    def append(self, record_id: str, kind: str, payload: Any) -> str:
        """ЕДИНСТВЕННЫЙ метод записи. Дубликат record_id — ошибка."""
        if record_id in self._records:
            raise ValueError(
                f"storage is append-only: record_id already exists: {record_id!r}"
            )
        self._seq += 1
        self._records[record_id] = Record(
            record_id=record_id,
            kind=kind,
            seq=self._seq,
            payload=payload,
            timestamp=now_iso(),
        )
        self._order.append(record_id)
        return record_id

    def get(self, record_id: str) -> Record:
        return self._records[record_id]

    def has(self, record_id: str) -> bool:
        return record_id in self._records

    def query(self, predicate: Callable[[Record], bool]) -> list[Record]:
        return [self._records[rid] for rid in self._order if predicate(self._records[rid])]

    def current_seq(self) -> int:
        return self._seq

    def snapshot_at(self, graph_version: int) -> "StorageView":
        """graph_version — счётчик записей (seq). Снимок = всё с seq <= graph_version."""
        return StorageView(self, graph_version)


class StorageView:
    """Снимок Storage на заданный graph_version. Только чтение."""

    def __init__(self, storage: Storage, graph_version: int) -> None:
        self._storage = storage
        self._graph_version = graph_version

    def get(self, record_id: str) -> Record:
        record = self._storage.get(record_id)
        if record.seq > self._graph_version:
            raise KeyError(f"{record_id!r} not visible at graph_version={self._graph_version}")
        return record

    def has(self, record_id: str) -> bool:
        return self._storage.has(record_id) and self._storage.get(record_id).seq <= self._graph_version

    def query(self, predicate: Callable[[Record], bool]) -> list[Record]:
        return self._storage.query(lambda r: r.seq <= self._graph_version and predicate(r))

    @property
    def graph_version(self) -> int:
        return self._graph_version
