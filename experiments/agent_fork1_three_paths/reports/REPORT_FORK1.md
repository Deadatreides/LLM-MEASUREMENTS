# REPORT_FORK1 — three outcomes in one pass (A · B · C)

Спека: `PROTOCOL.md`. 0 Evolution/HGT/Pareto/registry/ASSEMBLE/HETEROSTEP в этом пакете.

## 1. Дуга LIFE-8/9/DELTA-0

LIFE-8: организм не выходит за пределы B3 на старом полигоне (B3_STACK). LIFE-9: недекомпозированный whole бьёт весь степ-декомпозированный аппарат, сконструированная декомпозиция (LLM-planner) проигрывает ещё сильнее. DELTA-0: на F2 (новый полигон, золотой план, реальный executor) Δ=-0.500 — LLM-исполнитель неточен на exact-set чекпойнте (precision 0.545, recall 0.819), whole's мягкий финал прощает эту неточность. FORK-1 разводит: была ли проблема в ИСПОЛНЕНИИ (Path A, det-атомы) или в СНИСХОДИТЕЛЬНОСТИ threshold-финала (Path B, T_hard=EXTRACT-SUM, точный финал).

## 2. Path A: числа + TOOL_PIPELINE

r_A=1.000 (40/40), cost_A=0, Δ_A=r_A-r_m*_F2=0.475. **TOOL_PIPELINE=true** (0 LLM-вызовов в критических атомах -- НЕ победа LLM-роя, §6/§9 PROTOCOL.md).

## 3. Path B: T_hard спека, r_m*_B, r_∪ LLM/DET, Δ

T_hard=EXTRACT-SUM: транзакции, JOIN(category AND region), final = точная СУММА amount (не threshold-ДА/НЕТ). 2 чекпойнта: matched_ids (exact-set), sum_value (numeric, = final_oracle).

r_m*_B=0.000 (m*_B=llama-3.2-1b-instruct-q4_0). r_∪_B_LLM=0.025 (break_at={'FILTER': 39}), r_∪_B_DET=1.000 (break_at={}).

Δ_B_LLM=0.025, Δ_B_DET=1.000. Cost: whole(m*_B)=656 vs union-LLM=812 (потолок 2B_hard=7200) -> в пределах потолка

### 3.1 Forensic: почему r_m*_B=0.000 у ВСЕХ 6 моделей (не баг, см. BLOCKERS.md)

Количественно по 240 ответам (6 моделей × 40 задач), из `metrics/path_b_whole_detail.json`:
~35% (85/240) перечисляют слагаемые по одной строке вместо суммы (`extract_number`'s
"ровно одна строка-число" корректно даёт v=0, не гадает); ~65% (155/240) выводят ОДНО
число как просят, но НИ РАЗУ (0/155) не попадают в толеранс 0.01 — включая 2 модели
(qwen2.5-coder-1.5b, smollm2-1.7b), которые 40/40 раз дали single-line ответ и ни разу
не были правы. `r_∪_B_DET=1.000` подтверждает: задача точно решаема, оракул корректен —
маленькие (1-2B) модели просто не считают точную сумму 2-5 операндов в один проход без
рассуждений.

## 4. Path C: продуктовая спека

# Path C — product spec (K8, 0 new generate() calls)

Из уже посчитанных чисел: Δ_A=0.475 (TOOL_PIPELINE), Δ_B_LLM=0.025, Δ_B_DET=1.000.

## Продукт

- **primary**: single whole `m*` на классе short structured QA (F2's m*=0.525-tier reference; T_hard's m*_B=llama-3.2-1b-instruct-q4_0, r_m*_B=0.000).

- **optional**: детерминированный tool pipeline, когда нужен точный set/sum (Path A/B-DET показали потолок ~1.0 при 0 LLM-вызовах в критическом атоме).

- **explicit non-goal**: multi-LLM HGT на 1-2B без показанного Δ>0 в критическом LLM-атоме (LLM_SWARM_HAS_SENSE=False на этом прогоне).


## 5. Сводная таблица

| path | r | r_m* ref | Δ | LLM в критическом атоме? | cost |
|---|---|---|---|---|---|

| A det-F2 | 1.000 | 0.525 | 0.475 | no | 0 |
| B whole | 0.000 | 0.000 | 0.000 | n/a | 656 |
| B ∪ LLM | 0.025 | 0.000 | 0.025 | yes | 812 |
| B ∪ DET | 1.000 | 0.000 | 1.000 | no | 0 |

## 6. Корзина + Next

**FORK_TOOL_ONLY**

HIGH_SINGLE: нет


Next: продукт = whole + det tools; LLM-рой на 1-2B не приоритет; R&D отбора/HGT стоп

## 7. Non-claims

- Δ_A>0 НЕ объявлена победой LLM-роя (TOOL_PIPELINE).
- FILTER-цепочка не подставляет golden ids при провале -- реальный обрыв.
- SUM всегда детерминирован (и в B-LLM, и в B-DET) -- LLM никогда не считает сумму.
- T_hard не усложнялся после Δ_B; exact-set не смягчался после цифр.
- `agent_delta0_new_grid/`, `experiment11/` прочитаны, не изменены. `arch2/`, `experiment14/`, `agent_life*/`, `agent_a5*/` не читались.
- Пороги §8 не двигались после первых чисел.

## 8. Приложение
CHECKLIST.md: `agent_fork1_three_paths/CHECKLIST.md` (все пункты [x]).

