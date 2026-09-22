# PROTOCOL.md — agent_life9_decomp_alpha (LIFE-9)

Written and locked BEFORE any Phase A/B number and BEFORE any Phase C
live call. Не переписывается после результатов; интерпретация — только в
BLOCKERS.md/REPORT_LIFE9.md, рядом с механическим вердиктом, никогда
вместо него.

## 0. Рамка

После LIFE-8 (`B3_STACK` — организм не выходит за пределы B3 на 6
моделях/4 фиксированных шагах): есть ли α>0 (другая цепочка/разрез
оживляет UNION-мёртвых), и если да — зафиксировать минимальный
D-пайплайн и его `r`; если нет — формально закрыть S0. Один
`run_all.py`, без остановки после любой фазы. Без новых моделей, без
evolve/HGT/Pareto/stall-сеток/block registry — этот пакет НЕ строит
ASSEMBLE-генотипы и не запускает `Evolution` нигде вообще (единственное
касание `arch2/` — `arch2/fitness.py::paired_delta_r`, чистая статистика,
без зависимости от генотипов).

Изоляция: пишем только в `agent_life9_decomp_alpha/`. Читаем
`agent_life3..8/`, `arch2/`, `agent_a5_live_m/`, `experiment14/`,
`experiment11/` — без правок нигде.

## 1. P0 — grid=reuse

Живая сетка скопирована byte-identical из
`agent_life8_close_b3/metrics/live_grid/` (md5-проверено до Phase A),
тот же `GEN_SEED=150002`/`SPLIT_SEED=20260821`, что LIFE-3..8. 0 новых
`generate()` для Phase A/B — весь новый живой вызов только в Phase C.

## 2. Находка сессии (меняет форму Phase B — до любого числа)

`whole_grid.json` имеет ПОЛНОЕ покрытие 6 моделей на TRAIN и покрытие
ТОЛЬКО ОДНОЙ модели на TEST — победителя B0 по train
(`qwen2.5-coder-1.5b-instruct-q4_0`, 100 test-строк; остальные 5 моделей
— 0 test-строк). Это независимо подтверждает выбор P1 ниже.

Прямая проверка 200 сырых ответов этой модели вскрыла реальный,
нераскрытый ранее дефект РАЗБОРА в общем, неправимом
`experiment14/seams.py::whole_seam`/`_ASSIGN_RE`: нумерованный список
("1. parcels = 5") не матчится регэкспом, привязанным к началу строки —
КАЖДОЕ поле уходит в "missing" → INAPPLICABLE, даже если все значения
верны. Проверено эмпирически (не предположено, первая попытка с
дефолтным seed `experiment14/tasks/heterostep.py`'s (140001) дала
бессмыслицу — поймано и исправлено на `GEN_SEED=150002`): снятие
ведущего `^\s*\d+[.)]\s*` с каждой строки ПЕРЕД тем же неизменным
`whole_seam` восстанавливает 89 из 200 (raw 0.295 → reparsed 0.740), 0
регрессий.

**Обработка**: `alpha_B2_raw` (как есть) и `alpha_B2_reparsed` (локальная
поправка, не правка `experiment14/`) — оба в BLOCKERS.md.
`alpha_B2 := alpha_B2_reparsed` — число, на которое реально смотрят
пороги ниже (репарс только восстанавливает потерянное форматированием,
никогда не изобретает).

## 3. Самоопределённые премисы (P1-P5)

- **P1**: `B2_model := qwen2.5-coder-1.5b-instruct-q4_0` (победитель B0
  по train whole-task rate) — у B2 самого по себе нет модели-личности (B2
  — маршрут per-kind top-2, до 4 разных пар); лучший обоснованный
  заменитель — эмпирически лучший одиночный whole-task решатель, и (см.
  §2) единственная модель с существующим test-покрытием whole.
- **P2**: kill-step(ы) задачи `t` = кинды с 0 проходящих моделей из 6.
  Задача может иметь >1 kill-step. `fail_s[kind]` считает по U_dead
  задачам, где `kind` — (один из) kill-step(ов) — намеренно
  двойной счёт при 2+ kill-kinds.
- **P3**: `ρ(t) := ` доля НЕ-PASS среди ВСЕХ 24 ячеек (4 кинда × 6
  моделей) задачи `t` (не только kill-step — тот тривиально всегда 1.0
  по построению). `ρ̄_FAIL := mean(ρ(t) for t in U_dead)`.
- **P4**: пайплайн Phase C:
  - Planner (`B2_model`) получает запись + ТОЛЬКО итоговую инструкцию
    (`task["steps"][-1]["instruction"]`) — не готовую раскладку на 4
    шага. JSON-список 2-4 объектов `{"kind": READ|LOOKUP|COMPUTE,
    "name", "brief"}` (FORMAT исключён — по заданию). Markdown-ограда
    снимается перед `json.loads`; невалидный/вне диапазона JSON =
    провал планировщика для этой задачи (потрачен generate() для пола,
    не-PASS для α̂_decomp/r_D_test) — логируется, не падение.
  - Executor: по одному вызову на подшаг, модель = `orders[kind][0]`
    (B3-greedy ранг-1 для этого кинда, панель, `lean_seeds.py`
    неизменно). Промпт — запись + "Уже установлено: ..." (из ранее
    извлечённых подшагов) + `brief` + стандартная закрывающая строка
    (тот же узор, что `tasks14.step_prompt`).
  - У промежуточных подшагов НЕТ своего оракула (это LLM-придуманная
    раскладка, не настоящие именованные шаги задачи) — их значения
    только извлекаются (`numeric_seam(text, 0.0)["extracted"]`, тот же
    механизм тир-1/тир-2, фиктивный оракул) и пробрасываются дальше как
    "установлено", никогда не оцениваются PASS/FAIL сами по себе.
  - Финальная проверка: сырой текст ПОСЛЕДНЕГО подшага (после снятия
    того же list-prefix, что и §2) против РЕАЛЬНОГО итогового оракула
    задачи (`task["steps"][-1]["oracle"]`, всегда numeric, всегда
    "shipping_cost" — все 200 задач по одному шаблону,
    `experiment14/tasks/heterostep.py::_build_task`), через неизменный
    `numeric_seam`, толеранс 0.011. Это и есть критерий PASS для
    α̂_decomp/r_D_test — ничего другого.
- **P5**: **Поправка бюджета, до прогона.** `r_D_test` определено на
  `U_dead∪U_ok` = ВЕСЬ test (100), не только `U_dead`. Оценка задания
  "~40-150 вызовов"/"|U_dead|×(1+3)" покрывает только `U_dead`-часть.
  Дизайн: ОДИН прогон пайплайна на всех 100 test-задачах (1 planner + до
  3 executor ≈ 300-400 вызовов, не 40-150); α̂_decomp — pass-rate на
  подмножестве `U_dead` ЭТОГО ЖЕ прогона, r_D_test — pass-rate на всех
  100. Дешевле и честнее двух отдельных прогонов. Пол floor пересчитан
  под это реальное ожидаемое число, не под "40-150".

## 4. Phase A — residual anatomy (CPU)

`src/residual.py::compute_residual_anatomy(ds, task_ids)`. На test (100)
и отдельно на panel (train[:80], для Phase B's alpha_any). Сверка:
`len(U_ok_test)/100` должно ТОЧНО совпасть с `ceiling.json`'s (LIFE-8,
скопирован для сверки в `metrics/life8_ceiling_reference.json`)
`UNION.r_test` (0.570), допуск ±1 задача (ожидается точное совпадение —
тот же грид, та же логика `_r_for_route`, скопирована неизменной).
→ `metrics/residual_anatomy.json`.

## 5. Phase B — α̂_whole (CPU)

`src/whole_alpha.py`. `alpha_B2_raw`/`alpha_B2_reparsed` на `U_dead`
(test). `alpha_B2 := alpha_B2_reparsed`. `alpha_any_reparsed` на
`U_dead` (panel) — диагностика, НЕ гейтит пороги.

Пороги (не двигать):
```
STRONG   : alpha_B2 >= 0.20
MODERATE : 0.05 <= alpha_B2 < 0.20
WEAK     : alpha_B2 < 0.05
```

**C-skip**: `if alpha_B2 < 0.05: skip Phase C, basket=S0_CLOSED сразу`.
Иначе Phase C обязательна (whole_grid есть, но alpha_B2 не WEAK).

## 6. Phase C — α̂_decomp / r_D_test (live, только если не C-skip)

`src/decomp_pipeline.py`, P4/P5. Смоук: 4 задачи U_dead, ручная
проверка (JSON парсится, executor выполняется, финальная проверка
считается) ДО полного прогона. Лог — `metrics/decomp_calls.jsonl`,
детерминированные seed'ы `LIVE_SEED_BASE_9=9000 + stable_hash(task_id,
role, model_id)`.

**Пол (уточнено против §3 P5's числа, которое было про ОБЪЁМ, не про
ИНФРАСТРУКТУРУ)**: пол — это доля вызовов `generate()` БЕЗ
`generation_failed=True` среди ВСЕХ фактически совершённых вызовов,
≥80%. Невалидный JSON от планировщика или "не тот" под-шаг — это
СОДЕРЖАТЕЛЬНЫЙ результат (не-PASS для этой задачи), не сбой пола;
пол ловит именно поломку generate()/загрузки модели (`ALPHA_BLOCKED`,
Next = "починить generate/parse, не теория"), а не низкое качество
JSON-планов, что было бы находкой, а не блокером.

## 7. Phase D — продолжение в том же run_all

`alpha_star = max(alpha_B2, alpha_decomp если Phase C бежала иначе -inf)`.

**alpha_star < 0.05** (достижимо только если Phase C бежала и всё равно
слабо — C-skip уже закрывает случай "whole сразу говорит weak"):
```
Корзина: S0_CLOSED
Формула: 1-UNION доминирует, B2 whole/decomp не поднимает пол.
Next: "S0+6 моделей исчерпаны; следующий пакет = новая задача/шаги
       ИЛИ ортогональный атом с проверкой P(PASS|all FAIL)≥τ;
       не registry ради cardinality, не HGT"
```

**alpha_star >= 0.05**:
```
D1. кандидат-продукт = best of {whole B2 (reparsed, весь test, n=100,
    уже есть), decomp pipeline (r_D_test)} по test-r
    -> metrics/d_route.json
D2. таблица: r_B1, r_B2_route, r_B3, r_UNION (из life8_ceiling_reference.json),
    r_org (LIFE-8's R=0.456, взято как есть, не пересчитано),
    r_whole_B2, r_D_test
D3. корзина (проверяется в этом порядке):
    D_BEATS_B3   : r_D_test > r_B3 AND paired_delta_r(D,B3)'s bootstrap
                   CI/sign-test устойчиво не в пользу B3 (n=100>=30,
                   arch2.fitness.paired_delta_r, неизменная)
    D_BEATS_ORG  : r_D_test > r_org + 0.03
    D_ONLY_ALPHA : alpha_star>=0.05, но r_D_test <= r_B3
    Next: "слой D + E_i = default; HGT на S0 не развивать"
D4. Список task_id: U_dead -> ожили/не ожили — в отчёт.
```

**Phase C floor fail**: `ALPHA_BLOCKED`; Next = "починить
generate/parse, не теория".

## 8. Чего не делать (задание §5, соблюдается буквально)

evolve/HGT/8-seed организм; добор моделей/скан `models/*.gguf`;
G_STALL/Pareto/Imp-seed glue; повтор life8 ceiling как единственный
результат; остановка после Phase B с вопросом.

## 9. Бюджет

Phase A+B: секунды CPU. Phase C (если не skip): ~300-400 живых вызовов
(§3 P5) — смоук первым, затем полный прогон, одна сессия.
