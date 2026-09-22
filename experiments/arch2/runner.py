"""runner.py — Complex Layer, уровень 1: интерпретатор генотипа.

Исполняет восемь узлов (SPEC.md §1.2, §31) над состоянием σ = (артефакт, evidence,
история, стоимость, задача). Стохастика — только внутри CALL (её даёт backend);
сам интерпретатор детерминирован.

Пять гарантий (SPEC.md §7.2), каждая покрыта контрольным случаем в tests/:
 1. фактическая стоимость никогда не превышает бюджет — проверка ПЕРЕД каждым
    CALL по консервативной верхней оценке и факт-проверка ПОСЛЕ;
 2. трасса полна: каждый исполненный узел + сырьё модели до парсинга (I-16);
 3. детерминизм по структуре при одинаковых выходах молекул;
 4. Runner не пишет ничего, кроме трассы;
 5. RESOLVED не выставляется от недоверенного шва (I-15) — структурно.

Решение по семантике STOP, зафиксировано здесь и в SPEC.md: **STOP завершает весь
прогон**, а не только свою ветвь. Иначе «мы решили не тратить дальше» в одной ветви
PAR противоречило бы продолжению соседних. Валидатор дополнительно запрещает узлы
после STOP внутри SEQ (недостижимый код). ASSEMBLE (§31) наследует это решение без
изменений: STOP в любом слоте завершает весь прогон, тем же путём, что и в PAR.
"""

from __future__ import annotations

import contextlib
import copy
import io
import time
from dataclasses import dataclass, field
from typing import Any, Optional

RESOLVED = "RESOLVED"
UNRESOLVED = "UNRESOLVED"
INAPPLICABLE_OUT = "INAPPLICABLE"
EXHAUSTED = "EXHAUSTED"
ERROR_OUT = "ERROR"

PASS = "PASS"
FAIL = "FAIL"
INAPPLICABLE = "INAPPLICABLE"
ERROR = "ERROR"


class RunnerInvariantError(RuntimeError):
    """Нарушена гарантия Runner'а. Падать громко лучше, чем тихо превысить бюджет."""


@dataclass
class State:
    """σ. `cost` — токены, потраченные НА ЭТОМ пути (ветви PAR не делят историю);
    глобальный расход живёт в ExecContext.spent, потому что токены тратятся
    по-настоящему, независимо от того, в какой ветви."""

    task_id: str
    task_family: str
    task: dict
    origin_model: Optional[str]
    artifact: str = ""
    evidence: list = field(default_factory=list)
    initial_seam_result: Optional[dict] = None
    history: list = field(default_factory=list)
    cost: int = 0
    n_calls: int = 0
    terminal: bool = False
    stop_reason: Optional[str] = None
    exhausted: bool = False
    errored: bool = False
    meta: dict = field(default_factory=dict)
    """Непрозрачный для интерпретатора карман backend'а. Runner его только ПЕРЕНОСИТ,
    никогда не читает и не интерпретирует. Нужен offline-backend'у, чтобы помнить, из
    какой сохранённой записи получен текущий артефакт (без этого retry-плечо нельзя
    сопоставить с состоянием коллизии). Гарантия 4 не нарушена: пишет backend, не Runner."""

    def branch(self) -> "State":
        """Копия для независимой ветви PAR: артефакт/evidence/история не текут между ветвями."""
        return State(
            task_id=self.task_id, task_family=self.task_family, task=self.task,
            origin_model=self.origin_model, artifact=self.artifact,
            evidence=list(self.evidence), initial_seam_result=self.initial_seam_result,
            history=list(self.history), cost=self.cost, n_calls=self.n_calls,
            terminal=self.terminal, stop_reason=self.stop_reason,
            exhausted=self.exhausted, errored=self.errored, meta=dict(self.meta),
        )


@dataclass
class RunResult:
    complex_id: str
    task_id: str
    outcome: str
    cost: int
    n_calls: int
    trace: list
    final_status: Optional[str]
    stop_reason: Optional[str]
    exhausted: bool


@dataclass
class _Ctx:
    registry: Any
    backend: Any
    budget: int
    seed_vector: list
    spent: int = 0
    trace: list = field(default_factory=list)
    frames: list = field(default_factory=list)   # [(limit, spent_at_entry), ...]


# -- вспомогательное -------------------------------------------------------------------


def _last_seam(state: State) -> Optional[dict]:
    for ev in reversed(state.evidence):
        if ev.get("status") is not None:
            return ev
    return None


def _trusted_pass(state: State) -> bool:
    ev = _last_seam(state)
    return bool(ev and ev.get("status") == PASS and ev.get("trusted"))


def _headroom(ctx: _Ctx) -> int:
    """Сколько токенов ещё можно потратить с учётом глобального бюджета и всех
    активных BUDGET-рамок. Минимум по всем ограничениям."""
    room = ctx.budget - ctx.spent
    for limit, spent_at_entry in ctx.frames:
        room = min(room, limit - (ctx.spent - spent_at_entry))
    return room


# -- исполнение узлов -------------------------------------------------------------------


def _exec(node: dict, state: State, ctx: _Ctx, path: str) -> State:
    if state.terminal:
        return state
    op = node["op"]
    handler = _HANDLERS[op]
    return handler(node, state, ctx, path)


def _exec_call(node: dict, state: State, ctx: _Ctx, path: str) -> State:
    # Автокатализ: молекула может быть ЗАМОРОЖЕННЫМ комплексом предыдущего поколения.
    # Исполняется в ТОМ ЖЕ контексте — общий бюджет, общая трасса, никаких чёрных
    # ящиков (SPEC.md §10). Отдельного узла в языке для этого не нужно.
    if getattr(ctx.registry, "is_composite", None) and ctx.registry.is_composite(node["molecule"]):
        inner = ctx.registry.get(node["molecule"])["genotype"]
        ctx.trace.append({"node_path": path, "op": "CALL_COMPOSITE", "molecule": node["molecule"],
                          "inner_complex_id": inner.get("complex_id")})
        return _exec(inner["root"], state, ctx, f"{path}/composite")

    params = node["params"]
    prompt = params["prompt"]
    max_tokens = int(params["max_tokens"])
    seed_slot = int(params["seed_slot"])

    model_id = ctx.registry.resolve_model(node["molecule"], state, seed_slot)

    # Оценка стоимости ДО вызова. Backend может знать её точно (offline replay читает
    # уже записанные usage-токены) — тогда берём точную; иначе консервативная
    # статическая граница из реестра. Инвариант один и тот же: est >= факт.
    est = None
    if hasattr(ctx.backend, "estimate"):
        est = ctx.backend.estimate(
            model_id=model_id, prompt_variant=prompt, temperature=params["temperature"],
            seed_slot=seed_slot, max_tokens=max_tokens, state=state, seed_vector=ctx.seed_vector,
        )
    if est is None:
        est = ctx.registry.max_input_tokens(node["molecule"], prompt) + max_tokens

    # ГАРАНТИЯ 1, часть «до»: консервативная оценка не должна пробить ни глобальный
    # бюджет, ни ни одну активную BUDGET-рамку.
    if est > _headroom(ctx):
        state.exhausted = True
        ctx.trace.append({
            "node_path": path, "op": "CALL", "molecule": node["molecule"], "params": params,
            "model_id": model_id, "skipped": "budget_exhausted",
            "estimate": est, "headroom": _headroom(ctx),
        })
        return state

    t0 = time.time()
    out = ctx.backend.call(
        model_id=model_id, prompt_variant=prompt, temperature=params["temperature"],
        seed_slot=seed_slot, max_tokens=max_tokens, state=state, seed_vector=ctx.seed_vector,
    )
    latency = out.get("latency", time.time() - t0)

    if not out.get("available", True):
        state.errored = True
        ctx.trace.append({
            "node_path": path, "op": "CALL", "molecule": node["molecule"], "params": params,
            "model_id": model_id, "unavailable": out.get("reason"), "latency": latency,
        })
        return state

    in_tok = int(out.get("input_tokens") or 0)
    out_tok = int(out.get("output_tokens") or 0)
    spent = in_tok + out_tok

    # ГАРАНТИЯ 1, часть «после».
    if spent > est:
        raise RunnerInvariantError(
            f"{path}: actual {spent} tokens exceeds static estimate {est} "
            f"(molecule={node['molecule']}, prompt={prompt}, max_tokens={max_tokens})"
        )
    ctx.spent += spent
    if ctx.spent > ctx.budget:
        raise RunnerInvariantError(f"{path}: global budget {ctx.budget} exceeded ({ctx.spent})")

    state.artifact = out.get("raw_text", "")
    state.cost += spent
    state.n_calls += 1
    state.history.append({"node_path": path, "kind": "CALL", "model_id": model_id,
                          "prompt": prompt, "tokens": spent})
    if out.get("generation_failed"):
        state.errored = True

    ctx.trace.append({
        "node_path": path, "op": "CALL", "molecule": node["molecule"], "params": params,
        "model_id": model_id,
        "raw_output": state.artifact,               # I-16: сырьё ДО парсинга, всегда
        "input_tokens": in_tok, "output_tokens": out_tok, "latency": latency,
        "generation_failed": bool(out.get("generation_failed")),
        "source": out.get("source"),
    })
    return state


def _exec_check(node: dict, state: State, ctx: _Ctx, path: str) -> State:
    seam_id = node["seam"]
    trusted = ctx.registry.is_trusted(seam_id)          # ГАРАНТИЯ 5 (I-15)
    impl = ctx.registry.get(seam_id)["impl"]
    try:
        # Кодовый шов ИСПОЛНЯЕТ кандидатный код (унаследовано от experiment11/12),
        # а тот иногда печатает. Вывод глушится, чтобы журнал кампании оставался
        # читаемым; сам факт исполнения недоверенного кода в процессе — известное
        # свойство стенда, зафиксированное в STATUS.md, а не введённое здесь.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = dict(impl(state.artifact, state.task))
    except Exception as exc:                            # noqa: BLE001
        result = {"status": ERROR, "collision_type": None,
                  "details": {"reason": f"{type(exc).__name__}: {exc}"}}
        state.errored = True
    result["trusted"] = trusted
    result["seam_id"] = seam_id
    state.evidence.append(result)
    state.history.append({"node_path": path, "kind": "CHECK", "seam": seam_id,
                          "status": result.get("status"), "trusted": trusted})
    ctx.trace.append({"node_path": path, "op": "CHECK", "seam": seam_id,
                      "status": result.get("status"), "trusted": trusted,
                      "collision_type": result.get("collision_type"),
                      "details": result.get("details")})
    return state


def _exec_seq(node: dict, state: State, ctx: _Ctx, path: str) -> State:
    ctx.trace.append({"node_path": path, "op": "SEQ", "n_children": len(node["children"])})
    for i, child in enumerate(node["children"]):
        state = _exec(child, state, ctx, f"{path}/seq[{i}]")
        if state.terminal:
            break
    return state


def _exec_par(node: dict, state: State, ctx: _Ctx, path: str) -> State:
    """Ленивое исполнение ветвей от КОПИИ входного состояния; выход по первому
    PASS от доверенного шва. Токены всех фактически исполненных ветвей учтены."""
    ctx.trace.append({"node_path": path, "op": "PAR", "n_children": len(node["children"])})
    merged = state.branch()
    n_started = 0
    for i, child in enumerate(node["children"]):
        if _headroom(ctx) <= 0:
            merged.exhausted = True
            break
        branch_state = state.branch()
        n_started += 1
        branch_state = _exec(child, branch_state, ctx, f"{path}/par[{i}]")
        if _trusted_pass(branch_state):
            ctx.trace.append({"node_path": path, "op": "PAR_RESULT", "winner": i,
                              "branches_started": n_started})
            return branch_state
        # ветвь не дала PASS: её evidence/история присоединяются к общему исходу
        merged.evidence.extend(branch_state.evidence[len(state.evidence):])
        merged.history.extend(branch_state.history[len(state.history):])
        merged.cost += branch_state.cost - state.cost
        merged.n_calls += branch_state.n_calls - state.n_calls
        merged.exhausted = merged.exhausted or branch_state.exhausted
        merged.errored = merged.errored or branch_state.errored
        if branch_state.terminal:                       # STOP завершает весь прогон
            merged.terminal = True
            merged.stop_reason = branch_state.stop_reason
            merged.artifact = branch_state.artifact
            break
        merged.artifact = branch_state.artifact
    ctx.trace.append({"node_path": path, "op": "PAR_RESULT", "winner": None,
                      "branches_started": n_started})
    return merged


def _exec_assemble(node: dict, state: State, ctx: _Ctx, path: str) -> State:
    """SPEC.md §31. Слот `i` получает состояние, ЗАУЖЕННОЕ на шаг `i` задачи: `task`
    подменяется на `{**task, "step_index": i, "step": steps[i]}` (нет новых полей
    `State` — шаг виден через уже существующее поле `task`, как решено в SPEC.md §31);
    artifact/evidence слота стартуют пустыми, по образцу `replay.Dataset.fresh_state`
    (шаг — независимый вопрос, а не продолжение чужого артефакта, в отличие от PAR,
    где ветви — альтернативные попытки той же цели и потому делят входной артефакт).

    Слот подтверждён ⟺ его ветвь закончилась PASS от ДОВЕРЕННОГО шва — тот же критерий,
    что `_trusted_pass` уже применяет в PAR. Итог всего узла попадает в
    `merged.evidence` ОДНОЙ синтетической записью, дописанной ПОСЛЕДНЕЙ: так
    `_last_seam`/`run()` видят агрегат по ВСЕМ слотам, а не статус последнего по
    порядку слота (без этого RESOLVED мог бы зависеть от секретной ветви, где ⟨последний
    слот прошёл, но предыдущий провалился⟩). ГАРАНТИЯ 5 не нарушена: агрегат несёт
    `trusted=True`, только если построен из уже доверенных PASS каждого слота — сам он
    не занимается независимой оценкой доверия.

    STOP в любом слоте завершает весь прогон (SPEC.md §1.2, как в PAR) — агрегат в этом
    случае НЕ дописывается: вычисление оборвано, а не завершено, и исход должен решаться
    тем, что уже накоплено (обычно UNRESOLVED/EXHAUSTED), а не подделанным PASS/FAIL.

    ВАЖНО (найдено на кампании a1, до публикации любого результата): per-слотовое
    evidence копится в ЛОКАЛЬНОМ `slot_evidence`, НЕ в `merged.evidence`, до самого
    конца. Первая редакция дописывала evidence каждого слота в `merged` СРАЗУ внутри
    цикла — и если слот i (i>0) прерывал прогон через STOP, честный PASS слота 0
    оставался ПОСЛЕДНЕЙ записью `merged.evidence`, и `_last_seam`/`run()` объявляли
    RESOLVED всему ASSEMBLE, хотя вычисление даже не дошло до решения. `slot_evidence`
    коммитится в `merged.evidence` ТОЛЬКО при нормальном завершении цикла (вместе с
    агрегатом); на пути STOP `merged.evidence` остаётся РОВНО тем, чем было на входе.
    """
    children = node["children"]
    ctx.trace.append({"node_path": path, "op": "ASSEMBLE", "n_children": len(children)})
    steps = (state.task or {}).get("steps") or []
    merged = state.branch()
    slot_evidence: list = []
    n_confirmed = 0
    for i, child in enumerate(children):
        if _headroom(ctx) <= 0:
            merged.exhausted = True
            break
        slot_state = state.branch()
        slot_state.artifact = ""
        slot_state.evidence = []
        if i < len(steps):
            slot_state.task = {**state.task, "step_index": i, "step": steps[i]}
        slot_state = _exec(child, slot_state, ctx, f"{path}/assemble[{i}]")

        slot_evidence.extend(slot_state.evidence)
        merged.history.extend(slot_state.history[len(state.history):])
        merged.cost += slot_state.cost - state.cost
        merged.n_calls += slot_state.n_calls - state.n_calls
        merged.exhausted = merged.exhausted or slot_state.exhausted
        merged.errored = merged.errored or slot_state.errored
        merged.artifact = slot_state.artifact
        if slot_state.terminal:                      # STOP завершает весь прогон
            merged.terminal = True
            merged.stop_reason = slot_state.stop_reason
            ctx.trace.append({"node_path": path, "op": "ASSEMBLE_RESULT",
                              "n_children": len(children), "confirmed": n_confirmed,
                              "stopped_at_slot": i})
            return merged
        if _trusted_pass(slot_state):
            n_confirmed += 1

    all_confirmed = n_confirmed == len(children)
    merged.evidence.extend(slot_evidence)      # полный след слотов -- ПОСЛЕ него агрегат
    merged.evidence.append({
        "status": PASS if all_confirmed else FAIL, "trusted": True, "collision_type": None,
        "details": {"kind": "assemble", "confirmed": n_confirmed, "n_children": len(children)},
        "seam_id": None,
    })
    ctx.trace.append({"node_path": path, "op": "ASSEMBLE_RESULT", "n_children": len(children),
                      "confirmed": n_confirmed, "all_confirmed": all_confirmed})
    return merged


def _exec_switch(node: dict, state: State, ctx: _Ctx, path: str) -> State:
    obs_name = node["obs"]
    value = ctx.registry.observe(obs_name, state)
    branch = node["cases"].get(value)
    chosen = str(value) if branch is not None else "default"
    if branch is None:
        branch = node["default"]
    ctx.trace.append({"node_path": path, "op": "SWITCH", "obs": obs_name,
                      "value": value, "chosen": chosen})
    sub = f"{path}/case[{value}]" if chosen != "default" else f"{path}/default"
    return _exec(branch, state, ctx, sub)


def _exec_budget(node: dict, state: State, ctx: _Ctx, path: str) -> State:
    limit = int(node["limit_tokens"])
    ctx.frames.append((limit, ctx.spent))
    ctx.trace.append({"node_path": path, "op": "BUDGET", "limit_tokens": limit,
                      "spent_at_entry": ctx.spent})
    try:
        state = _exec(node["child"], state, ctx, f"{path}/budget")
    finally:
        _, entry = ctx.frames.pop()
        ctx.trace.append({"node_path": path, "op": "BUDGET_END",
                          "spent_inside": ctx.spent - entry, "limit_tokens": limit})
    return state


def _exec_stop(node: dict, state: State, ctx: _Ctx, path: str) -> State:
    state.terminal = True
    state.stop_reason = node["reason"]
    ctx.trace.append({"node_path": path, "op": "STOP", "reason": node["reason"]})
    return state


_HANDLERS = {
    "CALL": _exec_call, "CHECK": _exec_check, "SEQ": _exec_seq, "PAR": _exec_par,
    "SWITCH": _exec_switch, "BUDGET": _exec_budget, "STOP": _exec_stop,
    "ASSEMBLE": _exec_assemble,
}


# -- точка входа -------------------------------------------------------------------------


def run(genotype: dict, state: State, registry, backend, budget_tokens: int,
        seed_vector: Optional[list] = None) -> RunResult:
    ctx = _Ctx(registry=registry, backend=backend, budget=int(budget_tokens),
               seed_vector=list(seed_vector or [0, 1, 2, 3]))
    start = state.branch()
    final = _exec(genotype["root"], start, ctx, "root")

    last = _last_seam(final)
    status = last.get("status") if last else None

    if last is not None and status == PASS and last.get("trusted"):
        outcome = RESOLVED                       # ГАРАНТИЯ 5
    elif final.errored:
        outcome = ERROR_OUT
    elif final.exhausted:
        outcome = EXHAUSTED
    elif status == INAPPLICABLE:
        outcome = INAPPLICABLE_OUT
    else:
        outcome = UNRESOLVED

    return RunResult(
        complex_id=genotype.get("complex_id"), task_id=final.task_id, outcome=outcome,
        cost=ctx.spent, n_calls=final.n_calls, trace=ctx.trace, final_status=status,
        stop_reason=final.stop_reason, exhausted=final.exhausted,
    )
