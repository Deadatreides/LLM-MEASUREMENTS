"""heterostep.py — программный генератор задач HETEROSTEP (эксп. 14).

Три шага РАЗНОГО когнитивного рода в одной цепочке:

    READ    -- вытащить величину из полуструктурированной записи (чтение)
    FORMAT  -- привести дату к ISO (знание конвенции, не вычисление)
    COMPUTE -- посчитать сумму со ставкой региона (арифметика + поиск по таблице)

Шесть требований к полигону (TASK_EXPERIMENT14.md §3), каждое — урок из провала MSARITH:

  1. декомпозиция задана руками (ветка A проекта);
  2. **инструкция шага -- спецификация, не формула**. MSARITH хранил
     `formula: "subtotal + 105"`, то есть шаг был исполним БЕЗ модели, и потолок сборки
     оказался ключом ответов, а не способностью системы. Здесь оракул считает генератор,
     а промпт получает только словесное требование;
  3. каждый шаг требует способности: READ -- разобрать запись, FORMAT -- знать, что
     "07 March 2026" это 2026-03-07, COMPUTE -- найти ставку в таблице и применить;
  4. каждый шаг механически проверяем (см. `seams.py`);
  5. шаги разного рода -- иначе разным моделям негде разойтись, и тезис непроверяем;
  6. ровно одно значение на шаг -- извлечение перестаёт быть узким местом (в MSARITH
     значение шага читалось лишь в 44.6% ячеек).

Оракулы вычисляются кодом. Ни одно ожидаемое значение не вписано руками.
"""

from __future__ import annotations

import random
from typing import Optional

GEN_SEED = 140001
N_TASKS = 200
SPLIT_SEED = 20260819

# -- словари генератора (фиксированы до прогона) ---------------------------------------

_ITEMS = ("widget", "bracket", "sensor", "cable", "adapter", "bearing", "valve", "relay")
_MONTHS = ("January", "February", "March", "April", "May", "June",
           "July", "August", "September", "October", "November", "December")
_REGIONS = (("EU", 21), ("UK", 20), ("US", 0), ("CH", 8), ("NO", 25))
# Тариф за место. Ревизия v2 после пилота (предрегистрированная точка остановки
# «полигон слишком тяжёлый»): в v1 шаг COMPUTE требовал qty x price x (1+ставка) с
# двузначными десятичными -- модели 1-2B дали РОВНО 0.000 PASS на всех шести. Здесь
# арифметика однозначная (места x тариф), но вся нагрузка перенесена на ПОИСК:
# перевозчика надо найти в записи, тариф -- в таблице, число мест -- среди отвлекающих
# чисел. Способность по-прежнему требуется, вычислимость восстановлена.
_CARRIERS = (("DHL", 6), ("UPS", 9), ("FedEx", 7), ("TNT", 5), ("GLS", 8))

STEP_KINDS = ("READ", "FORMAT", "LOOKUP", "COMPUTE")


def _record(rng: random.Random) -> dict:
    """Одна синтетическая запись заказа + её истинные поля."""
    qty = rng.randint(2, 19)
    price = round(rng.uniform(3.0, 89.0), 2)
    day = rng.randint(1, 28)
    month_idx = rng.randrange(12)
    year = rng.choice((2024, 2025, 2026))
    region, rate = rng.choice(_REGIONS)
    item = rng.choice(_ITEMS)
    carrier, tariff = rng.choice(_CARRIERS)
    order_id = f"{rng.choice('ABCDEFGH')}-{rng.randint(1000, 9999)}"

    # отвлекающие числа: их наличие делает шаг READ настоящим чтением, а не
    # «возьми единственное число из текста»
    weight = round(rng.uniform(0.4, 40.0), 1)
    parcels = rng.randint(1, 9)

    text = (
        f"ORDER {order_id} | {qty} x {item} @ {price:.2f} | "
        f"shipped {day:02d} {_MONTHS[month_idx]} {year} via {carrier} | "
        f"region {region} | {parcels} parcels, {weight} kg\n"
        "carrier tariff per parcel: " + ", ".join(f"{c} {v}" for c, v in _CARRIERS)
    )
    return {
        "text": text, "order_id": order_id, "qty": qty, "price": price,
        "day": day, "month_idx": month_idx, "year": year,
        "region": region, "rate": rate, "item": item,
        "weight": weight, "parcels": parcels, "carrier": carrier, "tariff": tariff,
    }


def _build_task(idx: int, rng: random.Random) -> dict:
    r = _record(rng)

    shipping = float(r["parcels"] * r["tariff"])
    iso = f"{r['year']:04d}-{r['month_idx'] + 1:02d}-{r['day']:02d}"

    return {
        "task_id": f"HS_{idx:03d}",
        "record": r["text"],
        "meta": {k: r[k] for k in ("order_id", "region", "item", "carrier", "tariff")},
        "steps": [
            {
                "name": "parcels",
                "kind": "READ",
                # спецификация, а не формула: что найти, но не как посчитать
                "instruction": ("Найди в записи ЧИСЛО МЕСТ (parcels) в отправлении — "
                                "не количество товара и не вес. Выведи только это целое число."),
                "seam": "numeric",
                "oracle": float(r["parcels"]),
                "depends_on": [],
            },
            {
                "name": "ship_date",
                "kind": "FORMAT",
                "instruction": ("Приведи дату отгрузки из записи к формату YYYY-MM-DD. "
                                "Выведи только дату в этом формате."),
                "seam": "token",
                "oracle": iso,
                "depends_on": [],
            },
            {
                "name": "tariff",
                "kind": "LOOKUP",
                "instruction": ("Определи перевозчика этой отправки и найди его тариф за "
                                "одно место в таблице тарифов. Выведи только это число."),
                "seam": "numeric",
                "oracle": float(r["tariff"]),
                "depends_on": [],
            },
            {
                "name": "shipping_cost",
                "kind": "COMPUTE",
                # Ревизия v3 после пилота v2: раньше COMPUTE требовал ОДНОВРЕМЕННО найти
                # перевозчика, найти его тариф в таблице и умножить -- получилось 5% PASS,
                # цепочка вырождалась в один непроходимый шаг. Поиск вынесен в отдельный
                # род LOOKUP, здесь остаётся чистая арифметика над ДВУМЯ подтверждёнными
                # величинами. Роды шагов стали различаться сильнее, а не слабее.
                "instruction": ("Умножь число мест на тариф за одно место и выведи "
                                "произведение. Выведи только число."),
                "seam": "numeric",
                "oracle": shipping,
                "depends_on": ["parcels", "tariff"],
            },
        ],
    }


def build_tasks(n: int = N_TASKS, seed: int = GEN_SEED) -> dict:
    rng = random.Random(seed)
    return {f"HS_{i:03d}": _build_task(i, rng) for i in range(n)}


def split(task_ids: list, seed: int = SPLIT_SEED) -> dict:
    """train/test пополам. Разделение фиксировано seed'ом ДО прогона: таблица
    маршрутизации плеча C обучается на train и применяется к test, иначе C был бы
    выбором задним числом."""
    ids = sorted(task_ids)
    random.Random(seed).shuffle(ids)
    half = len(ids) // 2
    return {"train": sorted(ids[:half]), "test": sorted(ids[half:])}


# -- промпты ------------------------------------------------------------------------------


def whole_prompt(task: dict) -> str:
    """Плечо A: вся задача одним откликом. Контракт вывода объявлен ЯВНО — урок arch1
    этапа 8: модели не нарушали инструкцию, а НЕ ИМЕЛИ её."""
    lines = [f"{i + 1}. {s['instruction']}" for i, s in enumerate(task["steps"])]
    names = ", ".join(s["name"] for s in task["steps"])
    return (
        "Запись:\n" + task["record"] + "\n\n"
        "Ответь на три вопроса по этой записи:\n" + "\n".join(lines) + "\n\n"
        f"Формат ответа: ровно три строки вида «имя = значение», имена: {names}. "
        "Никакого текста кроме этих трёх строк."
    )


def _fmt(v) -> str:
    """Целое количество не должно выглядеть как `16.0` — это лишний повод для модели
    начать пересчитывать вместо того, чтобы использовать данное."""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def step_prompt(task: dict, step_idx: int, upstream: Optional[dict] = None) -> str:
    """Плечи B/C/D: один шаг = один вопрос = одно значение.

    `upstream` -- значения предыдущих шагов, от которых зависит текущий. Что именно в
    нём лежит, решает плечо: подтверждённые швом (B/C) или собственные прошлые ответы
    модели (D). Сам промпт различий не знает и потому побайтово одинаков при равных
    значениях -- это проверяется контрольным случаем.
    """
    step = task["steps"][step_idx]
    parts = ["Запись:\n" + task["record"], ""]
    if step["depends_on"] and upstream:
        known = [f"{n} = {_fmt(upstream[n])}" for n in step["depends_on"] if n in upstream]
        if known:
            parts.append("Уже установлено: " + "; ".join(known))
            parts.append("")
    parts.append(step["instruction"])
    parts.append("Выведи ТОЛЬКО одно значение, без пояснений и без имени поля.")
    return "\n".join(parts)


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    tasks = build_tasks()
    sp = split(list(tasks))
    print(f"задач: {len(tasks)}, train {len(sp['train'])}, test {len(sp['test'])}")
    t = tasks["HS_000"]
    print("\n--- пример записи ---")
    print(t["record"])
    print("\n--- оракулы ---")
    for s in t["steps"]:
        print(f"  {s['kind']:8s} {s['name']:16s} -> {s['oracle']!r}")
    print("\n--- промпт целостного плеча ---")
    print(whole_prompt(t))
    print("\n--- промпт шага COMPUTE (upstream подтверждён) ---")
    print(step_prompt(t, 2, {"units": t["steps"][0]["oracle"]}))
