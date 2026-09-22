"""CONTEXT BUILDER — вспомогательный модуль уровня 5, используется
ACTION EXECUTOR (CONTEXT_MODEL.md).

Собирает контекст под конкретное действие — «модель никогда не получает
весь граф» (§1). ARCH-1 не реализует DECOMPOSE/PATCH/SOFT-VERIFY
(WORK_PLAN.md §8), поэтому из таблицы слоёв §2 нужны только те, что
использует GENERATE/REGENERATE: TASK_CONTEXT, DEPENDENCY_CONTEXT (только
прямые upstream, CONTEXT_MODEL §4.2/OQ-CTX-2), REPAIR_CONTEXT,
EVIDENCE_CONTEXT (только HARD), output_contract.

Зависит только от graph_core (direct_parents) — уровень 1, без
model_adapter/state_engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

GENERATE = "GENERATE"
REGENERATE = "REGENERATE"

_LAYER_ORDER = ("TASK_CONTEXT", "DEPENDENCY_CONTEXT", "REPAIR_CONTEXT", "EVIDENCE_CONTEXT")
# §6: порядок сброса при переполнении — от наименее к наиболее ценному.
# TASK_CONTEXT, output_contract, прямые upstream и REPAIR_CONTEXT никогда
# не сбрасываются — из нашего урезанного набора слоёв это оставляет
# только EVIDENCE_CONTEXT как единственный кандидат на сброс.
_DROPPABLE_ON_OVERFLOW = ("EVIDENCE_CONTEXT",)


class ContextOverflow(Exception):
    """§6: «Молчаливое обрезание запрещено» — переполнение после всех
    разрешённых сокращений возвращается вызывающему явно, не тихо режется."""


@dataclass(frozen=True)
class Context:
    action_type: str
    target_claim_id: str
    layers: dict = field(default_factory=dict)  # layer_name -> str
    output_contract: Optional[str] = None

    def render(self) -> str:
        parts = []
        for name in _LAYER_ORDER:
            if name in self.layers and self.layers[name]:
                parts.append(f"[{name}]\n{self.layers[name]}")
        if self.output_contract:
            parts.append(f"[OUTPUT_CONTRACT]\n{self.output_contract}")
        return "\n\n".join(parts)


def _dependency_context(graph: Any, parent_ids: tuple) -> str:
    lines = []
    for parent_id in parent_ids:
        parent = graph.get_claim(parent_id)
        lines.append(f"{parent_id}: {parent.content}")
    return "\n".join(lines)


def build_context(
    action_type: str,
    target_claim_id: str,
    graph: Any,
    task_statement: str,
    output_contract: Optional[str] = None,
    do_not_touch: tuple = (),
    evidence_summary: Optional[str] = None,
    max_chars: Optional[int] = None,
    dependency_claim_ids: Optional[tuple] = None,
) -> Context:
    """CONTEXT_MODEL.md §3.

    evidence_summary передаётся, только если основание HARD (§3: «ТОЛЬКО
    если HARD») — вызывающий код (ACTION EXECUTOR) обязан не передавать
    его для SOFT/недоверенных результатов; сам build_context этого не
    проверяет (не его уровень ответственности — он просто размещает то,
    что дано).

    do_not_touch — claims с necessity MAY_REUSE/DO_NOT_TOUCH (REPAIR
    PLANNER, этап 4) — то, что регенерировать запрещено.

    dependency_claim_ids — явный список прямых upstream, в обход
    graph.direct_parents(target_claim_id). Нужно для ПЕРВОГО GENERATE
    нового claim'а (этап 6, ORCHESTRATOR): ребро source->target нельзя
    добавить в граф раньше, чем появится сам target, а значит на момент
    сборки контекста graph.direct_parents(target) закономерно пуст —
    структура зависимостей в этом случае известна заранее (не выводится
    моделью) и передаётся напрямую. Для REGENERATE (claim и рёбра уже в
    графе) явную передачу можно не делать — умолчание не меняет
    поведение.
    """
    if action_type not in (GENERATE, REGENERATE):
        raise ValueError(f"build_context: unsupported action_type {action_type!r}")

    parent_ids = (
        tuple(dependency_claim_ids)
        if dependency_claim_ids is not None
        else tuple(graph.direct_parents(target_claim_id))
    )

    layers: dict[str, str] = {
        "TASK_CONTEXT": task_statement,
        "DEPENDENCY_CONTEXT": _dependency_context(graph, parent_ids),
    }

    if action_type == REGENERATE:
        repair_lines = [f"изменить: {target_claim_id}"]
        if do_not_touch:
            repair_lines.append("НЕ менять: " + ", ".join(sorted(do_not_touch)))
        layers["REPAIR_CONTEXT"] = "\n".join(repair_lines)
        if evidence_summary:
            layers["EVIDENCE_CONTEXT"] = evidence_summary
        # НЕ включаем предыдущую неверную версию (CONTEXT_MODEL §4.1) --
        # она просто нигде здесь не запрашивается и не подставляется.

    context = Context(
        action_type=action_type,
        target_claim_id=target_claim_id,
        layers=layers,
        output_contract=output_contract,
    )

    if max_chars is not None and len(context.render()) > max_chars:
        for droppable in _DROPPABLE_ON_OVERFLOW:
            if droppable in context.layers:
                reduced_layers = dict(context.layers)
                del reduced_layers[droppable]
                context = Context(
                    action_type=action_type,
                    target_claim_id=target_claim_id,
                    layers=reduced_layers,
                    output_contract=output_contract,
                )
                if len(context.render()) <= max_chars:
                    return context
        raise ContextOverflow(
            f"context for {target_claim_id!r} exceeds max_chars={max_chars} "
            "after all permitted reductions"
        )

    return context
