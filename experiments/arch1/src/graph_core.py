"""GRAPH CORE — уровень 1 (API_BOUNDARIES.md §2).

Знает: структуру. Артефакты, claims, зависимости, версии, обходы.
Не знает: ничего об LLM, о промптах, о стоимости, о политике.

Не импортирует ничего, связанного с моделями (проверка L аудита
ARCH-0 — здесь нечего менять при замене модели).

Поля state/verification_status/confidence/evidence из DATA_MODEL.md
(claim/artifact) здесь сознательно отсутствуют — они принадлежат
STATE ENGINE и EVIDENCE ENGINE (этап 2), а не GRAPH CORE
(API_BOUNDARIES.md §1: «GRAPH CORE... Не знает... о политике»).
parents/children claim'а и claims артефакта — не хранимые списки,
а обход по DEPENDENCY-рёбрам (direct_children и т.п. — запрос,
не поле). superseded_by не хранится — вычисляется по supersedes
следующей версии (append-only без исключений).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from .storage import Provenance, Storage, new_id, now_iso

# --- константы (DATA_MODEL.md §3, §4) ------------------------------------

ATOMIC = "ATOMIC"
COMPOSITE = "COMPOSITE"
INDIVISIBLE_BLOCK = "INDIVISIBLE_BLOCK"

STATUS_HYPOTHESIS = "HYPOTHESIS"
STATUS_CONFIRMED = "CONFIRMED"
STATUS_REFUTED = "REFUTED"
STATUS_DISPUTED = "DISPUTED"
STATUS_REFUTED_BY_ACYCLICITY = "REFUTED_BY_ACYCLICITY"

_INACTIVE_STATUSES = (STATUS_REFUTED, STATUS_REFUTED_BY_ACYCLICITY)

CYCLE_DETECTED = "CYCLE_DETECTED"

# Виды записей, которые реально принадлежат GraphCore в Storage. С этапа 6
# один и тот же Storage делят StateEngine/SeamEngine/EvidenceEngine/
# ActionExecutor/orchestrator (kind="state_record"/"seam_result"/"evidence"/
# "run"/"action"/"task") -- graph_snapshot/graph_hash обязаны видеть только
# СВОИ записи, а не реплеить/хешировать вообще всё хранилище.
_GRAPH_RECORD_KINDS = frozenset(
    {"artifact_version", "claim_version", "dependency", "collapse", "cycle_observation"}
)


# --- сущности (DATA_MODEL.md §2-4) ----------------------------------------


@dataclass(frozen=True)
class Artifact:
    artifact_version_id: str
    artifact_id: str
    artifact_version: int
    artifact_type: str
    task_id: str
    content: str
    content_hash: str
    parents: tuple[str, ...]
    producer: str
    run_id: Optional[str]
    supersedes: Optional[str]
    provenance: Provenance
    creation_time: str


@dataclass(frozen=True)
class Claim:
    claim_version_id: str
    claim_id: str
    claim_version: int
    artifact_id: str
    content: str
    claim_type: str
    atomicity: str
    supersedes: Optional[str]
    provenance: Provenance


@dataclass(frozen=True)
class Dependency:
    dependency_id: str
    source_claim: str
    target_claim: str
    dependency_type: str
    status: str
    created_by: str
    provenance: Provenance


@dataclass(frozen=True)
class CollapseRecord:
    """Схлопывание SCC (DEPENDENCY_MODEL.md §5.2b)."""

    collapse_id: str
    composite_claim_id: str
    member_claim_ids: tuple[str, ...]
    provenance: Provenance


@dataclass(frozen=True)
class CycleObservation:
    """Отказ в добавлении ребра из-за цикла (DEPENDENCY_MODEL.md §5.2)."""

    observation_id: str
    source_claim: str
    target_claim: str
    path: tuple[str, ...]
    dependency_status: str
    timestamp: str = field(default_factory=now_iso)


class GraphCore:
    """GRAPH CORE (API_BOUNDARIES.md §2)."""

    def __init__(self, storage: Storage) -> None:
        self._storage = storage
        self._latest_artifact_version: dict[str, str] = {}
        self._artifact_versions_by_id_version: dict[tuple[str, int], str] = {}
        self._latest_claim_version: dict[str, str] = {}
        self._claim_versions_by_id_version: dict[tuple[str, int], str] = {}
        self._children: dict[str, list[str]] = {}
        self._parents: dict[str, list[str]] = {}
        self._collapsed_of: dict[str, str] = {}
        self._collapse_members: dict[str, tuple[str, ...]] = {}

    # -- запись через единственный внутренний путь: storage + индексы ----

    def _ingest(self, kind: str, payload: Any) -> str:
        if kind == "artifact_version":
            record_id = payload.artifact_version_id
            self._storage.append(record_id, kind, payload)
            self._latest_artifact_version[payload.artifact_id] = record_id
            self._artifact_versions_by_id_version[
                (payload.artifact_id, payload.artifact_version)
            ] = record_id
            return record_id

        if kind == "claim_version":
            record_id = payload.claim_version_id
            self._storage.append(record_id, kind, payload)
            self._latest_claim_version[payload.claim_id] = record_id
            self._claim_versions_by_id_version[
                (payload.claim_id, payload.claim_version)
            ] = record_id
            self._children.setdefault(payload.claim_id, [])
            self._parents.setdefault(payload.claim_id, [])
            return record_id

        if kind == "dependency":
            record_id = payload.dependency_id
            self._storage.append(record_id, kind, payload)
            self._children.setdefault(payload.source_claim, []).append(record_id)
            self._parents.setdefault(payload.target_claim, []).append(record_id)
            return record_id

        if kind == "collapse":
            record_id = payload.collapse_id
            self._storage.append(record_id, kind, payload)
            self._collapse_members[payload.composite_claim_id] = payload.member_claim_ids
            for member in payload.member_claim_ids:
                self._collapsed_of[member] = payload.composite_claim_id
            return record_id

        if kind == "cycle_observation":
            record_id = payload.observation_id
            self._storage.append(record_id, kind, payload)
            return record_id

        raise ValueError(f"unknown record kind: {kind!r}")

    # -- запись: публичный интерфейс (API_BOUNDARIES.md §2) --------------

    def add_artifact_version(
        self,
        *,
        artifact_type: str,
        task_id: str,
        content: str,
        artifact_id: Optional[str] = None,
        parents: Iterable[str] = (),
        producer: str = "SYSTEM",
        run_id: Optional[str] = None,
        provenance: Optional[Provenance] = None,
        content_hash: Optional[str] = None,
    ) -> str:
        if artifact_id is None:
            artifact_id = new_id("art")
            version = 1
            supersedes = None
        else:
            prev_id = self._latest_artifact_version.get(artifact_id)
            if prev_id is None:
                version, supersedes = 1, None
            else:
                prev = self._storage.get(prev_id).payload
                version, supersedes = prev.artifact_version + 1, prev_id

        artifact = Artifact(
            artifact_version_id=new_id("art"),
            artifact_id=artifact_id,
            artifact_version=version,
            artifact_type=artifact_type,
            task_id=task_id,
            content=content,
            content_hash=content_hash or hashlib.sha256(content.encode("utf-8")).hexdigest(),
            parents=tuple(parents),
            producer=producer,
            run_id=run_id,
            supersedes=supersedes,
            provenance=provenance or Provenance(created_by=producer),
            creation_time=now_iso(),
        )
        return self._ingest("artifact_version", artifact)

    def add_claim_version(
        self,
        *,
        artifact_id: str,
        content: str,
        claim_type: str,
        atomicity: str = ATOMIC,
        claim_id: Optional[str] = None,
        provenance: Optional[Provenance] = None,
    ) -> str:
        if claim_id is None:
            claim_id = new_id("claim")
            version, supersedes = 1, None
        else:
            prev_id = self._latest_claim_version.get(claim_id)
            if prev_id is None:
                version, supersedes = 1, None
            else:
                prev = self._storage.get(prev_id).payload
                version, supersedes = prev.claim_version + 1, prev_id

        claim = Claim(
            claim_version_id=new_id("claim"),
            claim_id=claim_id,
            claim_version=version,
            artifact_id=artifact_id,
            content=content,
            claim_type=claim_type,
            atomicity=atomicity,
            supersedes=supersedes,
            provenance=provenance or Provenance(created_by="SYSTEM"),
        )
        return self._ingest("claim_version", claim)

    def add_dependency(
        self,
        *,
        source_claim: str,
        target_claim: str,
        dependency_type: str,
        status: str = STATUS_CONFIRMED,
        created_by: str = "HUMAN",
        provenance: Optional[Provenance] = None,
    ) -> str:
        """DEPENDENCY_MODEL.md §5.2.

        Цикл: source -> target замыкает цикл тогда и только тогда, когда
        target уже может достичь source по существующим рёбрам (source
        принадлежит текущему downstream_closure(target)) — эквивалентно
        псевдокоду спеки, но записано через однозначное условие
        достижимости, а не дословную (неоднозначную) формулировку.

        При цикле ребро НЕ добавляется в основной граф (граф не
        меняется) в любом случае; для status=HYPOTHESIS ребро
        дополнительно сохраняется отдельно с status=REFUTED_BY_ACYCLICITY
        (видимо в диагностике, исключено из обхода) — §5.2.a. Для
        status=CONFIRMED решение о схлопывании SCC принимает вызывающий
        код через отдельный явный collapse_scc() — §5.2.b.
        """
        if source_claim not in self._children or target_claim not in self._children:
            raise KeyError(
                "add_dependency: unknown claim_id "
                "(call add_claim_version first for both endpoints)"
            )

        src = self._resolve(source_claim)
        tgt = self._resolve(target_claim)
        is_cycle = src == tgt or src in self.downstream_closure(target_claim)

        if is_cycle:
            observation = CycleObservation(
                observation_id=new_id("cycobs"),
                source_claim=source_claim,
                target_claim=target_claim,
                path=tuple(sorted(self.downstream_closure(target_claim) | {tgt})),
                dependency_status=status,
            )
            self._ingest("cycle_observation", observation)

            if status == STATUS_HYPOTHESIS:
                dep = Dependency(
                    dependency_id=new_id("dep"),
                    source_claim=source_claim,
                    target_claim=target_claim,
                    dependency_type=dependency_type,
                    status=STATUS_REFUTED_BY_ACYCLICITY,
                    created_by=created_by,
                    provenance=provenance or Provenance(created_by=created_by),
                )
                self._ingest("dependency", dep)

            return CYCLE_DETECTED

        dep = Dependency(
            dependency_id=new_id("dep"),
            source_claim=source_claim,
            target_claim=target_claim,
            dependency_type=dependency_type,
            status=status,
            created_by=created_by,
            provenance=provenance or Provenance(created_by=created_by),
        )
        return self._ingest("dependency", dep)

    def collapse_scc(
        self, claim_ids: Iterable[str], *, provenance: Optional[Provenance] = None
    ) -> str:
        """DEPENDENCY_MODEL.md §5.2.b.

        Явный, отдельно вызываемый шаг (не автоматический побочный
        эффект add_dependency — см. её docstring). Исходные claims не
        удаляются: get_claim(member_id) продолжает возвращать их как
        есть; обход (direct_children/direct_parents/closures) прозрачно
        резолвит их в composite через _collapsed_of.
        """
        members = tuple(sorted(set(claim_ids)))
        if not members:
            raise ValueError("collapse_scc: claim_ids must be non-empty")
        member_claims = [self.get_claim(m) for m in members]  # валидирует существование

        composite_claim_id = new_id("claim")
        content = "INDIVISIBLE_BLOCK(" + ", ".join(
            f"{c.claim_id}: {c.content}" for c in member_claims
        ) + ")"
        composite = Claim(
            claim_version_id=new_id("claim"),
            claim_id=composite_claim_id,
            claim_version=1,
            artifact_id=member_claims[0].artifact_id,
            content=content,
            claim_type="OTHER",
            atomicity=INDIVISIBLE_BLOCK,
            supersedes=None,
            provenance=provenance or Provenance(created_by="SYSTEM", input_refs=members),
        )
        self._ingest("claim_version", composite)

        collapse_record = CollapseRecord(
            collapse_id=new_id("collapse"),
            composite_claim_id=composite_claim_id,
            member_claim_ids=members,
            provenance=Provenance(created_by="SYSTEM", input_refs=members),
        )
        self._ingest("collapse", collapse_record)

        return composite_claim_id

    # -- чтение: публичный интерфейс (API_BOUNDARIES.md §2) --------------

    def get_claim(self, claim_id: str, version: Optional[int] = None) -> Claim:
        if version is None:
            version_id = self._latest_claim_version.get(claim_id)
        else:
            version_id = self._claim_versions_by_id_version.get((claim_id, version))
        if version_id is None:
            raise KeyError(f"unknown claim: {claim_id!r} version={version!r}")
        return self._storage.get(version_id).payload

    def get_artifact(self, artifact_id: str, version: Optional[int] = None) -> Artifact:
        if version is None:
            version_id = self._latest_artifact_version.get(artifact_id)
        else:
            version_id = self._artifact_versions_by_id_version.get((artifact_id, version))
        if version_id is None:
            raise KeyError(f"unknown artifact: {artifact_id!r} version={version!r}")
        return self._storage.get(version_id).payload

    def direct_children(self, claim_id: str) -> list[str]:
        resolved = self._resolve(claim_id)
        result: set[str] = set()
        members = self._collapse_members.get(resolved, (resolved,))
        for member in members:
            for dep_id in self._children.get(member, []):
                dep = self._storage.get(dep_id).payload
                if dep.status in _INACTIVE_STATUSES:
                    continue
                target = self._resolve(dep.target_claim)
                if target == resolved:
                    continue  # внутреннее ребро SCC
                result.add(target)
        return sorted(result)

    def direct_parents(self, claim_id: str) -> list[str]:
        resolved = self._resolve(claim_id)
        result: set[str] = set()
        members = self._collapse_members.get(resolved, (resolved,))
        for member in members:
            for dep_id in self._parents.get(member, []):
                dep = self._storage.get(dep_id).payload
                if dep.status in _INACTIVE_STATUSES:
                    continue
                source = self._resolve(dep.source_claim)
                if source == resolved:
                    continue
                result.add(source)
        return sorted(result)

    def direct_parent_edges(self, claim_id: str) -> list[tuple[str, str]]:
        """Как direct_parents(), но пары (parent_claim_id, dependency_type).

        Добавлено на этапе 4: REPAIR_MODEL.md §3, шаг 1 сужения нуждается
        именно в типе связи (DATA/COMPUTATIONAL), а не только в факте её
        существования — этого direct_parents() принципиально не отдаёт.
        Не расширяет границы уровня 1 (по-прежнему только чтение из
        собственных структур GraphCore).
        """
        resolved = self._resolve(claim_id)
        result: list[tuple[str, str]] = []
        members = self._collapse_members.get(resolved, (resolved,))
        for member in members:
            for dep_id in self._parents.get(member, []):
                dep = self._storage.get(dep_id).payload
                if dep.status in _INACTIVE_STATUSES:
                    continue
                source = self._resolve(dep.source_claim)
                if source == resolved:
                    continue
                result.append((source, dep.dependency_type))
        return result

    def downstream_closure(self, claim_id: str) -> set[str]:
        """DEPENDENCY_MODEL.md §6.1. O(V+E), точное (F34)."""
        start = self._resolve(claim_id)
        seen: set[str] = set()
        frontier = deque([start])
        while frontier:
            cur = frontier.popleft()
            for child in self.direct_children(cur):
                if child not in seen and child != start:
                    seen.add(child)
                    frontier.append(child)
        return seen

    def upstream_closure(self, claim_id: str) -> set[str]:
        start = self._resolve(claim_id)
        seen: set[str] = set()
        frontier = deque([start])
        while frontier:
            cur = frontier.popleft()
            for parent in self.direct_parents(cur):
                if parent not in seen and parent != start:
                    seen.add(parent)
                    frontier.append(parent)
        return seen

    def topological_order(self, claim_set: Iterable[str]) -> list[str]:
        nodes = {self._resolve(c) for c in claim_set}
        adjacency = {n: [c for c in self.direct_children(n) if c in nodes] for n in nodes}
        indegree = {n: 0 for n in nodes}
        for n in nodes:
            for child in adjacency[n]:
                indegree[child] += 1

        queue = deque(sorted(n for n in nodes if indegree[n] == 0))
        order: list[str] = []
        while queue:
            cur = queue.popleft()
            order.append(cur)
            for child in sorted(adjacency[cur]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)

        if len(order) != len(nodes):
            raise ValueError("topological_order: cycle detected within given claim_set")
        return order

    def root_origins(self, defective_claims: Iterable[str]) -> list[str]:
        """DEPENDENCY_MODEL.md §6.2 — через upstream_closure, а не
        через собственный статус claim'а (наивная версия ошибается, F31)."""
        defective = {self._resolve(c) for c in defective_claims}
        roots = [c for c in defective if not (self.upstream_closure(c) & defective)]
        return sorted(roots)

    # -- версионирование ---------------------------------------------------

    def graph_snapshot(self, graph_version: int) -> "GraphCore":
        snapshot = GraphCore(Storage())
        records = sorted(
            self._storage.query(lambda r: r.seq <= graph_version and r.kind in _GRAPH_RECORD_KINDS),
            key=lambda r: r.seq,
        )
        for record in records:
            snapshot._ingest(record.kind, record.payload)
        return snapshot

    def graph_hash(self, graph_version: int) -> str:
        records = sorted(
            self._storage.query(lambda r: r.seq <= graph_version and r.kind in _GRAPH_RECORD_KINDS),
            key=lambda r: r.seq,
        )
        canonical = [
            {"record_id": r.record_id, "kind": r.kind, "payload": dataclasses.asdict(r.payload)}
            for r in records
        ]
        blob = json.dumps(canonical, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def current_graph_version(self) -> int:
        return self._storage.current_seq()

    # -- внутреннее --------------------------------------------------------

    def _resolve(self, claim_id: str) -> str:
        seen: set[str] = set()
        current = claim_id
        while current in self._collapsed_of and current not in seen:
            seen.add(current)
            current = self._collapsed_of[current]
        return current
