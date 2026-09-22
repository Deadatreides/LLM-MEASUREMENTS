# REPORT_DELTA0 — iron-clad Δ measurement on a new polygon

Спека: `PROTOCOL.md`. F2 (multi-hop record) на транзакциях, НЕ HETEROSTEP. 0 Evolution/HGT/Pareto/registry/ASSEMBLE в этом пакете.

## 1. Класс T и почему T1-T6

F2: JOINT-фильтр (category AND region) -> агрегат (SUM/COUNT/MAX amount) -> производный ДА/НЕТ. T1 (не одношаговая): 3 логических хода. T2 (final oracle детерминирован): код генератора. T3 (3 checkpoint, не HETEROSTEP kinds): matched_ids/aggregate_value/final. T4 (контекст): db_text, K=18 записей, 2082-2186 символов (измерено C2). T5: HIGH_SINGLE допустим, не отменяет пакет. T6: N_TRAIN=80, N_TEST=40, SEED_T=20260823.

## 2. B и B_atom

B=4200, B_atom=1050 (>= B/4 отношение соблюдено по построению), B_total_union потолок = 8400. Число получено из фактически измеренного размера промпта (C2/C5), не угадано.

## 3. Таблица r_m по моделям, m*, cost

| model | r_m | n_pass/n | mean_cost |
|---|---|---|---|

| llama-3.2-1b-instruct-q4_0 | 0.000 | 0/40 | 861 |
| qwen2.5-coder-1.5b-instruct-q4_0 | 0.500 | 20/40 | 992 |
| qwen3-1.7b-q4_0-unsloth **<- m\***  | 0.525 | 21/40 | 975 |
| internvl3-2b-q4_k_m | 0.500 | 20/40 | 992 |
| gemma-3-it-1b-q5_k_s | 0.500 | 20/40 | 968 |
| smollm2-1.7b-instruct-q4_k_m | 0.500 | 20/40 | 1098 |

## 4. Спека A (атомы)

- **FILTER**: in={'db_text': 'str (full DB)', 'category': 'str', 'region': 'str'}, out={'ids': 'comma-separated TXN-#### tokens, or НЕТ'}, max_tokens=220
- **AGGREGATE**: in={'id_amount_slice': 'dict[id -> amount], only matched ids', 'op': 'SUM|COUNT|MAX'}, out={'value': 'bare number'}, max_tokens=40
- **DERIVE**: in={'aggregate_value': 'float', 'threshold': 'float', 'comparator': '>=|<'}, out={'answer': 'ДА|НЕТ'}, max_tokens=20

## 5. r_∪_lower, метод M1

M1 (oracle-route, золотой план, реальные E, executor=m\*=qwen3-1.7b-q4_0-unsloth): r_∪_lower=0.025 (1/40), mean_cost=1020. Разрыв цепочки (без golden подстановки): {'FILTER': 39}.

## 6. Δ, корзина, HIGH_SINGLE

Δ = r_∪_lower - r_m* = -0.500. Корзина: **DELTA_NONPOS**.

HIGH_SINGLE (r_m*>=0.70): нет (r_m*=0.525)

Cost: mean(whole,m\*)=975 vs mean(union-chain)=1020 (потолок 2B=8400) -> в пределах потолка

### 6.1 Forensic (раскрыто рядом, вердикт §6 не переписан)

39/40 обрывов цепочки — все на FILTER (`break_at={'FILTER': 39}`), 0 на AGGREGATE/DERIVE.
Полная детализация: `metrics/union_lower_detail.json` (raw-текст каждого атома, добавлено
после первого прогона — см. BLOCKERS.md). Количественно (по всем 40 задачам, FILTER v=0
случаи): precision (доля извлечённых id, которые верны) = **0.545**, recall (доля золотых
id, которые нашлись) = **0.819**, Jaccard = **0.499** (медиана 0.5). Разбивка: 19/40 —
модель нашла ВСЕ золотые id ПЛЮС лишние (superset); 18/40 — смесь попаданий/промахов/лишних;
2/40 — пропустила часть золотых, без лишних (subset); **0/40** — полностью мимо (disjoint).

Модель НЕ игнорирует задачу и не угадывает случайно (0 disjoint) — она вовлечена в
JOINT-фильтрацию и обычно близка (recall 82%), но почти никогда не достаточно точна для
СТРОГОГО совпадения множеств (precision 55%, 1/40 exact). Whole-task's мягкий финальный
ДА/НЕТ прощает эту неточность (не идеальная фильтрация всё ещё часто даёт правильный
агрегат по ЭТУ сторону порога); FILTER-чекпойнт — нет. Тот же исполнитель (m\*), тот же
живой вызов, но разная строгость проверки — не баг пайплайна (см. BLOCKERS.md).

## 7. Next

сменить семейство T или атомы; не отбор/HGT; текущий T не даёт зазора роя

## 8. Non-claims

- r_∪_lower требует ВСЮ цепочку (FILTER И AGGREGATE И DERIVE), не "хотя бы один checkpoint".
- Цепочка не подставляет golden-значения при провале атома -- реальный обрыв, не рескью.
- Whole-промпт просит только финальный ДА/НЕТ -- intermediate не элиситируются и не влияют на v_final.
- `experiment11/` не отредактирован. `arch2/`, `experiment14/`, `agent_life*/`, `agent_a5*/` не читались и не изменялись.
- Пороги §6/PROTOCOL не двигались после первого посчитанного Δ.

## 9. Приложение
CHECKLIST.md: `agent_delta0_new_grid/CHECKLIST.md` (все пункты [x]).

