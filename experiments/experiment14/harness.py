"""harness.py — четыре плеча эксперимента 14.

    A  WHOLE          1 вызов  вся задача одним откликом (контроль, текущий протокол)
    B  STEP_SAME      3 вызова по шагу, одна модель, upstream ПОДТВЕРЖДЁН швом
    C  STEP_ROUTED    3 вызова по шагу, модель выбрана ПОД РОД ШАГА, upstream подтверждён
    D  STEP_NOORACLE  3 вызова по шагу, одна модель, upstream — СОБСТВЕННЫЕ ответы

Читаемые разности: B−A — стоит ли декомпозиция вообще; **C−B — вклад разнообразия
моделей при равном числе вызовов (главный тезис)**; B−D — цена оракула.

РЕШЁННОЕ = финальный шаг (COMPUTE) прошёл шов. Промежуточные шаги — промежуточные;
именно значение последнего шага является ответом задачи. Подстановка подтверждённого
upstream НЕ засчитывает шаг как пройденный: она влияет только на то, что ВИДИТ
следующий шаг. Иначе плечо B само себе ставило бы оценку.

Сильное допущение плеч B/C: подтверждённый upstream берётся из оракула шага. Это ветка A
проекта (декомпозиция и проверка заданы), но это СИЛЬНЕЕ, чем нужно текущей системе,
поэтому плечо D существует как обязательный контроль и его разность с B публикуется
(гейт G3, TASK_EXPERIMENT14.md §5).
"""

from __future__ import annotations

import hashlib
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from configs.model_registry import MODEL_IDS, generate, load_model  # noqa: E402
from seams import INAPPLICABLE, PASS, check_step, whole_seam  # noqa: E402
from tasks.heterostep import STEP_KINDS, step_prompt, whole_prompt  # noqa: E402

EXPERIMENT_ID = "exp14-v1"

TEMPERATURE = 0.5
TOP_P = 1.0
LIVE_SEED_BASE = 3000          # не пересекается с фазой 0 эксп.12 (1-4), retry (1000), E1 (2000+)
MAX_TOKENS_WHOLE = 200
MAX_TOKENS_STEP = 60

ARMS = ("A_WHOLE", "B_STEP_SAME", "C_STEP_ROUTED", "D_STEP_NOORACLE")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _prompt_id(p: str) -> str:
    return hashlib.sha256(p.encode("utf-8")).hexdigest()[:16]


def _gen(llm, model_id: str, prompt: str, max_tokens: int, slot: int = 0) -> dict:
    return generate(llm, model_id, prompt, temperature=TEMPERATURE, top_p=TOP_P,
                    seed=LIVE_SEED_BASE + slot, max_tokens=max_tokens)


# -- плечо A: целостный отклик ------------------------------------------------------------


def run_whole(task: dict, model_id: str, llm, slot: int = 0) -> dict:
    prompt = whole_prompt(task)
    raw = _gen(llm, model_id, prompt, MAX_TOKENS_WHOLE, slot)
    seam = whole_seam(raw.get("raw_text", ""), task["steps"])
    final = seam["step_results"][-1] if seam["step_results"] else {"status": INAPPLICABLE}
    tok = (raw.get("input_tokens") or 0) + (raw.get("output_tokens") or 0)
    return {
        "experiment_id": EXPERIMENT_ID, "arm": "A_WHOLE", "task_id": task["task_id"],
        "model": model_id, "prompt_id": _prompt_id(prompt), "prompt": prompt,
        "raw_output": raw.get("raw_text", ""),
        "step_results": seam["step_results"], "n_calls": 1, "tokens": tok,
        "latency": raw.get("generation_time_sec"),
        "final_status": final["status"], "resolved": final["status"] == PASS,
        "resolved_all_steps": all(s["status"] == PASS for s in seam["step_results"]),
        "generation_failed": bool(raw.get("generation_failed")),
        "timestamp": _now(),
    }


# -- плечи B/C/D: пошагово ------------------------------------------------------------------


def run_steps(task: dict, model_by_step: list, llm_by_model: dict, arm: str,
              use_oracle_upstream: bool, slot: int = 0) -> dict:
    """`model_by_step[i]` — модель для шага i. `llm_by_model` — уже загруженные модели.

    `use_oracle_upstream=True` (B, C): следующий шаг видит ПОДТВЕРЖДЁННОЕ значение.
    `use_oracle_upstream=False` (D): следующий шаг видит то, что модель сама выдала;
    если значение не извлеклось, upstream просто отсутствует в промпте.
    """
    results, prompts, raws = [], [], []
    tok = 0
    latency = 0.0
    upstream: dict = {}

    for i, step in enumerate(task["steps"]):
        mid = model_by_step[i]
        prompt = step_prompt(task, i, upstream)
        raw = _gen(llm_by_model[mid], mid, prompt, MAX_TOKENS_STEP, slot)
        res = check_step(step, raw.get("raw_text", ""))
        res["model"] = mid

        results.append(res)
        prompts.append(prompt)
        raws.append(raw.get("raw_text", ""))
        tok += (raw.get("input_tokens") or 0) + (raw.get("output_tokens") or 0)
        latency += raw.get("generation_time_sec") or 0.0

        # что увидит следующий шаг
        if use_oracle_upstream:
            upstream[step["name"]] = step["oracle"]
        elif res["extracted"] is not None:
            upstream[step["name"]] = res["extracted"]

    final = results[-1]
    return {
        "experiment_id": EXPERIMENT_ID, "arm": arm, "task_id": task["task_id"],
        "model": model_by_step[0] if len(set(model_by_step)) == 1 else "ROUTED",
        "model_by_step": list(model_by_step),
        "prompt_ids": [_prompt_id(p) for p in prompts], "prompts": prompts,
        "raw_outputs": raws, "step_results": results,
        "n_calls": len(task["steps"]), "tokens": tok, "latency": latency,
        "final_status": final["status"], "resolved": final["status"] == PASS,
        "resolved_all_steps": all(s["status"] == PASS for s in results),
        "upstream_mode": "oracle_confirmed" if use_oracle_upstream else "own_output",
        "timestamp": _now(),
    }


# -- фаза train: по шагам, все модели -------------------------------------------------------


def run_train_grid(tasks: dict, train_ids: list, on_checkpoint=None) -> list:
    """Для каждой модели — все три шага на всех train-задачах с подтверждённым upstream.

    Даёт сразу и таблицу маршрутизации (род шага -> модель), и лучшую одиночную модель
    для плеча B. Порядок модель-мажорный: загрузка модели дорога, вызов дёшев.

    С подтверждённым upstream шаги НЕЗАВИСИМЫ (значение предка известно заранее),
    поэтому их можно гонять в любом порядке — это и позволяет модель-мажорный обход.
    """
    out = []
    for mid in MODEL_IDS:
        llm, load_t = load_model(mid)
        print(f"[train] {mid} загружена за {load_t:.1f}с — "
              f"{len(train_ids) * 4} вызовов в очереди", flush=True)
        t0 = time.time()
        for tid in train_ids:
            task = tasks[tid]
            upstream = {}
            for i, step in enumerate(task["steps"]):
                prompt = step_prompt(task, i, upstream)
                raw = _gen(llm, mid, prompt, MAX_TOKENS_STEP)
                res = check_step(step, raw.get("raw_text", ""))
                out.append({
                    "experiment_id": EXPERIMENT_ID, "phase": "train", "task_id": tid,
                    "model": mid, "step_index": i, "step_name": step["name"],
                    "kind": step["kind"], "status": res["status"],
                    "extracted": res["extracted"], "oracle": res["oracle"],
                    "reason": res.get("reason"), "raw_output": raw.get("raw_text", ""),
                    "tokens": (raw.get("input_tokens") or 0) + (raw.get("output_tokens") or 0),
                    "latency": raw.get("generation_time_sec"), "timestamp": _now(),
                })
                upstream[step["name"]] = step["oracle"]      # подтверждённый upstream
        del llm
        print(f"[train] {mid} готова за {time.time() - t0:.0f}с", flush=True)
        if on_checkpoint:
            on_checkpoint(out)
    return out


# -- разбор train: таблица маршрутизации и лучшая одиночная ---------------------------------


def per_kind_rates(train_rows: list) -> dict:
    """{kind: {model: доля PASS}} — основа и для G0, и для таблицы маршрутизации."""
    acc: dict = {k: {m: [0, 0] for m in MODEL_IDS} for k in STEP_KINDS}
    for r in train_rows:
        cell = acc[r["kind"]][r["model"]]
        cell[0] += 1
        cell[1] += r["status"] == PASS
    return {k: {m: (v[1] / v[0] if v[0] else 0.0) for m, v in kv.items()}
            for k, kv in acc.items()}


def routing_table(rates: dict) -> dict:
    """Род шага -> модель с наибольшей долей PASS на train.

    Перечисление, не эволюция: 6^3 = 216 вариантов, оптимум находится точно. Это прямое
    следствие поправки к No Free Lunch (TASK_EXPERIMENT14.md §2) — там, где пространство
    перечислимо, перечисление доминирует поиск.
    """
    return {k: max(v, key=v.get) for k, v in rates.items()}


def best_single_model(train_rows: list) -> str:
    """Лучшая ОДИНОЧНАЯ модель по ВСЕЙ ЦЕПОЧКЕ (все шаги задачи верны), эмпирически.

    ДЕФЕКТ, НАЙДЕННЫЙ НА ГЕЙТЕ G0 И ИСПРАВЛЕННЫЙ ДО ИЗМЕРЕНИЯ. Раньше и «решено», и
    выбор одиночного конкурента считались по ФИНАЛЬНОМУ шагу. В плечах B и C upstream
    подтверждён оракулом, поэтому финальный шаг не зависит от предыдущих вовсе, и
    разность C-B оказалась бы нулём ПО ПОСТРОЕНИЮ: обе модели финального шага — одна и
    та же (лучшая по COMPUTE). Предрегистрированная метрика вырождена для плеч B/C.

    Исправление: основная метрика — ВСЕ шаги задачи верны. Это и есть решение задачи о
    четырёх полях записи; получить верным только последнее поле, когда три предыдущих
    подставил оракул, решением не является. Прежняя метрика сохраняется и публикуется
    рядом как вторичная (`resolved`), чтобы правка была видна, а не спрятана.
    """
    by_model: dict = {}
    for r in train_rows:
        by_model.setdefault(r["model"], {}).setdefault(r["task_id"], []).append(r["status"] == PASS)
    rates = {m: sum(all(v) for v in tasks.values()) / len(tasks)
             for m, tasks in by_model.items()}
    return max(rates, key=rates.get)


def all_steps_rates(train_rows: list) -> dict:
    by_model: dict = {}
    for r in train_rows:
        by_model.setdefault(r["model"], {}).setdefault(r["task_id"], []).append(r["status"] == PASS)
    return {m: sum(all(v) for v in tasks.values()) / len(tasks) for m, tasks in by_model.items()}


def gate_g0(rates: dict) -> dict:
    """G0 — гейт полигона: если одна модель лучшая для ВСЕХ родов шагов,
    гетерогенности негде взяться и тезис на этом пуле опровергнут."""
    table = routing_table(rates)
    winners = set(table.values())
    return {"table": table, "distinct_winners": len(winners),
            "passed": len(winners) > 1, "rates": rates}
