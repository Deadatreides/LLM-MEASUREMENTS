# PROTOCOL.md — agent_life7_selection

Написано ДО первого evolve/юнит-числа этого пакета. Не переписывается
после того, как числа увидены — интерпретация добавляется в BLOCKERS.md/
REPORT_LIFE7.md, не сюда.

## 0. Рамка

LIFE-6 (`agent_life6_retention/`, завершён): `RET_A_FAIL` (slot-safe
mutate) → `RET_B_FAIL` (carrier immunity, M=2 youngest). Оба рычага
подтверждены рабочими на уровне кода; forensic разбор показал, ПОЧЕМУ
`median_L` не сдвинулся у обоих — slot-safe защищает межпоколенческое
наследование слота, не жизнь родителя; immunity (M=2-youngest) двигает
mean_L (2.41 vs 1.57), но не медиану, т.к. защита временна (теряется за
1-2 поколения). Default-канал (complementary slot-HGT) НЕ считается
сломанным.

LIFE-7 не повторяет ни slot-safe, ни M=2-youngest. Меняется слой ПОД
ними: КТО умирает (économic D1/D2 → economic D1/D2, отфильтрованный
Pareto-доминированием по `(r, n_imp)`) и КОГДА популяция объявляется
застрявшей (не «побил ли кто-то B-эталон», а «улучшился ли лучший r/
лучший n_imp за прогон»).

**Явно из LIFE-6 §5.1 (не забывать в этом пакете)**: нельзя объявлять
FAIL только по тому, что `median_L_imp==1.0`, если `mean_L`/`p90_L`/
`occ_imp` выросли — медиана здесь ВТОРИЧНА, не единственный сигнал
(§2/§3 ниже используют mean/p90/occ как primary именно по этой причине).

Писать ТОЛЬКО в `agent_life7_selection/`. Читаем `arch2/`,
`agent_life2_complementary/`, `agent_life3_block_live/`,
`agent_life4_block_fix/`, `agent_life5_slot_map/`,
`agent_life6_retention/`, `agent_a5_live_m/` — без правок. Grid: REUSE
(копия LIFE-6, md5-проверена), 0 новых `generate()`. Без block_registry/
BLOCK_INSERT/R1-R4/A1-A5. Без slot-safe-mutate, без M=2-youngest
immunity, без новых атомов/моделей.

## 1. Default-канал

`p_slot_hgt=0.35`, `transfer_slot` — комплементарный-only (byte-копия
LIFE-6's `slot_hgt.py`), НЕТ slot-safe mutate, НЕТ ImmuneEvolution/M=2-
youngest. lean-старт assert, `references ∉ donors`, `donor_score_fn`
отсутствует, `pop=20`, `budget_per_task=3000`, `max_gen=32`.
`G_STALL=20` — ТО ЖЕ число, что LIFE-6, но критерий stall у WITH-вариантов
переопределён (§3 A2), не «побил ли B1/B2/B3».

## 2. Метрики (предрегистрация)

Primary (по слотам READ/LOOKUP/COMPUTE, gen≥1, каждый seed):
- `mean_L_imp`, `p90_L_imp` — среднее и 90-й перцентиль лайфспана
  (`death_gen - birth_gen`) среди носителей с Imp(s)=1 (пул всех Imp=1-
  сигнатур слота вместе).
- `occ_imp` — доля ПОКОЛЕНИЙ, где ≥1 alive non-reference несёт Imp(s)=1
  (считается по `summary["generations"][i]["survivors"]` — состояние
  ПОСЛЕ обработки поколения i, НЕ `["population"]`, которое, как
  подтверждено прямым чтением `arch2/evolve.py:449-469` в этой сессии,
  фиксируется ДО смерти этого поколения и отражает состояние на ВХОДЕ,
  на одно поколение "устаревшее").
- `frac_gen_AB_alive` — доля поколений, где ≥1 alive non-reference несёт
  `|ImpSet|>=2` (тот же источник, `survivors`).
- `N_AB_first_assembly` (формула — как LIFE-1..6, admitted births).
- `extinct_gen`, `any_gate_win` — только для отчёта; `extinct_gen` берёт
  ИСКЛЮЧИТЕЛЬНО верхнеуровневые `summary["extinct"]`/`summary["n_
  generations"]`, НИКОГДА `summary["generations"][i]["stall_count"/
  "extinct"]` (см. §3 A2 — этот подтверждённый нюанс: попо-генерационная
  история навсегда хранит АРХ2-ORIGINAL gate-based вердикт того
  поколения, даже после нашего override живых атрибутов).

Secondary (не решающие для корзины): `median_L_imp` (ожидаемо липкая,
см. §0), `n_imp_sig`, `loss:gain`, best `r` на panel — без claim B3.

Агрегаты — mean по seed, WITH vs CTRL, парно по позиции.

## 3. Branch A — всегда первым: Pareto-survival + stall-от-улучшения

### A1 — Pareto D1∪D2 (`src/pareto_death.py::ParetoEvolution`)

Подтверждено чтением `arch2/evolve.py`+`arch2/fitness.py` этой сессией:
`scored[cid]["r"]` (строится `F.evaluate_population`→`F.metrics`,
`fitness.py:263-288`) — плоский АБСОЛЮТНЫЙ resolve-rate, уже передан
override-методам как их же аргумент `scored`, независимого `F.metrics`
вызова не требуется. D1's `ids` = `self.population` (полный,
нефильтрованный — сам `self.population` не сужается до строки 469, ПОСЛЕ
обоих death-вызовов); D2's `ids` = `survivors` (уже отфильтрованный D1
локальный список). **Критично**: пул для проверки доминирования всегда
берётся из ПАРАМЕТРА `ids`, никогда из `self.population` напрямую — иначе
override D2 ошибочно считал бы жертв D1 всё ещё живыми.

Алгоритм: вызвать `super()._d1_screen_deaths`/`super()._d2_subsidy_deaths`
(штатная экономическая логика, без изменений) → получить список кандидатов
на смерть → генотип-кандидат ОСТАЁТСЯ (исключается из списка), если среди
alive non-ref в этом же `ids` НЕТ другого генотипа, который слабо
доминирует его по вектору `(r, extra)`: `r` не меньше, `extra` покоординатно
не меньше, и хотя бы одна координата строго больше. `extra` конфигурируем:
`(|ImpSet|,)` для Branch A (`SCALAR_NIMP`), `(Imp_READ,Imp_LOOKUP,
Imp_COMPUTE)∈{0,1}³` для Phase A3 (`VECTOR_SLOTS`) — один класс, один флаг
конструктора, не два подкласса.

Это НЕ то же самое, что arch2's собственный `front` (передаваемый в D2) —
подтверждено: `front` — это ТРЁХцелевой Pareto-фронт по `(r, -c, u)`
(`fitness_vector`, `fitness.py:284-285`, `evolve.py:396-398`), локальная
переменная, никогда не атрибут `self.` — не пересекается по имени и
семантике с новым `(r, n_imp)`-понятием этого пакета.

Реализация — подкласс `Evolution`, конструируется НАПРЯМУЮ (как везде в
этой линии; `arch2.evolve.build()` нигде не используется), без
monkeypatch.

### A2 — stall-от-улучшения (`src/stall_from_improvement.py`)

Подтверждено прямым чтением: stall/extinct — ВСТРОЕННЫЙ блок в
`run_generation` (`evolve.py:427-443`), НЕ отдельный переопределяемый
метод. Единственное другое место внутри ТОГО ЖЕ вызова, читающее эти
поля — прямое копирование в `report` (строки 458-459), происходящее ДО
любого внешнего override. **Подтверждено безопасным**: перезаписать
`ev.stall_count`/`ev.extinct` как обычные атрибуты СРАЗУ после возврата
`ev.run_generation(gen)` — ничто внутри уже завершившегося вызова не
перечитывает и не переветвляется по новым значениям; следующий вызов
`run_generation` использует `self.current_gen`/т.д. независимо, а
`self.stall_count`, прочитанный arch2's собственной формулой `self.
stall_count + 1` в НАЧАЛЕ следующего вызова, всё равно будет ПЕРЕЗАПИСАН
нашим трекером сразу после того вызова тоже — промежуточное
арх2-вычисление никогда не используется этим пакетом.

Задокументированное следствие, не устраняется: `ev.generations[-1]
["stall_count"/"extinct"]` (уже сериализованная построчная история)
навсегда показывает ОРИГИНАЛЬНЫЙ (gate-based) вердикт arch2 для того
поколения — только ЖИВЫЕ атрибуты и финальные `summary()["extinct"]`/
`summary()["n_generations"]` отражают override. `metrics_lib.py`
намеренно никогда не читает построчную историю этих двух полей.

Алгоритм: `current_best_r`/`current_best_n_imp` среди alive non-ref (из
`ev.population`, уже равного `survivors` этого поколения к моменту
возврата, и `latest_report["scores"][cid]["r"]`, бесплатно уже
посчитанного); `improved = current_best_r > best_r_so_far OR
current_best_n_imp > best_n_imp_so_far`; `stall_count = 0 если improved
иначе +1`; `extinct = stall_count >= G_STALL`. References остаются для
метрик `r` (`ref_test_metrics` в конце), не участвуют в вымирании (как и
раньше).

### Дизайн: WITH_A vs CTRL_A

WITH_A = default + A1(`SCALAR_NIMP`) + A2. CTRL_A = default LIFE-6
(обычные D1/D2, обычный gate-based stall, без переопределений).
seeds_A = 20262001..20262006 (6+6=12 клеток).

### Корзина (не двигать после чисел)

```
growth_signal =
  (mean_mean_L_imp(WITH) >= mean(CTRL)+0.3 на >=2 из {READ,LOOKUP,COMPUTE})
  OR (mean_p90_L_imp(WITH) >= mean(CTRL)+1.0 на >=1 слоте)
  OR (mean_occ_imp(WITH) >= mean(CTRL)+0.10 на >=2 слотах)

SEL_*_WORKS = growth_signal
  AND mean_frac_gen_AB_alive(WITH) >= mean(CTRL)
  AND mean_N_AB_first_assembly(WITH) >= mean(CTRL) - 1
  AND mean_extinct_gen(WITH) >= mean(CTRL)

SEL_*_FAIL = (NOT growth_signal)
  OR mean_N_AB_first_assembly(WITH) <= mean(CTRL) - 3
  OR (число seed, где extinct_gen(WITH) < extinct_gen(CTRL), парно) >= 4 из 6

SEL_*_PARTIAL = иначе
```
Precedence: WORKS → FAIL → иначе PARTIAL. `classify_sel(...)` — ОДНА
функция, буквально те же пороги, вызывается для A, A3 и B без
модификации. Обратить внимание: FAIL's extinct_gen-условие — ПАРНЫЙ
подсчёт по seed, отдельный от WORKS's простого сравнения средних — две
разные вычисления, не одно переиспользованное число.

## 4. Ветвление внутри `run_all.py` (без нового брифа)

```
SEL_A_WORKS   -> Phase A2: 4 seed WITH_A-only (20262011..14), сравнение
                 с уже посчитанным CTRL_A. Next: default = HGT + Pareto
                 (scalar n_imp) + stall-from-improvement; следующий
                 крупный = r vs B2/B3 на этом отборе.

SEL_A_PARTIAL -> Phase A3: ужесточить Pareto — VECTOR_SLOTS вместо
                 SCALAR_NIMP. 4 WITH_A3 + 4 CTRL_A3 (20262021..24).
                 Reclassify той же classify_sel.
                   WORKS-подобен -> default = векторный Pareto, done.
                   иначе -> Phase B в этом же прогоне (бюджет позволяет,
                     см. §7; если внезапно нет — BLOCKERS явно, REPORT
                     всё равно с A(+A3) и next-строкой "запустить B по
                     PROTOCOL §4 без переписывания").

SEL_A_FAIL    -> Phase B сразу: ТОЛЬКО A2 (stall-from-improvement), БЕЗ
                 Pareto-смерти (обычные D1/D2, plain Evolution). Проверяет,
                 виноват именно gate/extinct-критерий (arch2's) или сам
                 закон смерти D1/D2. 4 WITH_B + 4 CTRL_B (20262031..34).
                   SEL_B_WORKS -> default = HGT + stall-from-improvement
                     (смерть НЕ трогать).
                   SEL_B_FAIL -> «ни Pareto-смерть, ни stall-без-B3 не
                     держат слот во времени при pop=20; следующий пакет —
                     смена размера pop/темпа replace ИЛИ измерение r vs
                     B2/B3 на HGT-only; не атомы, не registry, не
                     youngest-immunity».
```

Запрещено в этом пакете: slot-safe mutate, M=2-youngest immunity,
«защищать самых старых» как основной рычаг, новые модели/атомы.

## 5. Смоук (до 12 клеток A)

- lean 20 различных, `|ImpSet|<2` у не-эталонов.
- ≥1 `TRANSFER_SLOT` complementary попытка.
- Юнит `ParetoEvolution`: искусственные `scored`/`accounts` — доминируемый
  кандидат остаётся в списке мёртвых, недоминируемый — исключается, для
  ОБОИХ `_d1_screen_deaths`/`_d2_subsidy_deaths`, с использованием ПАРАМЕТРА
  `ids` (не `self.population`).
- Юнит `StallFromImprovement`: два поколения без улучшения ни r, ни n_imp →
  `stall_count` растёт на 2; улучшение n_imp при том же r → `stall_count`
  сбрасывается в 0.
- 0 registry-импортов через `ast` (не substring по докстрингам — LIFE-6's
  собственная находка и фикс, переиспользуется, не повторяется вслепую).

Провал → BLOCKERS.md, стоп, без 12 клеток.

## 6. Forensic (в `run_all.py`, не потом)

На первом WITH-seed использованной ветки: доля death-кандидатов (`d1+d2`),
исключённых Pareto-фильтром (инструментировано в `ParetoEvolution.
_filter_non_dominated`, накапливается в `ev._pareto_forensic` за
поколение, дублируется в `summary["pareto_forensic"]`); средний `n_imp`
среди живых WITH vs CTRL; был ли хоть один `improved=True`-сброс stall
(`summary["a2_improved_log"]`), который arch2's собственный gate-критерий
НЕ засчитал бы. Если WITH-метрики ≈ CTRL при ненулевом Pareto-save —
явная строка в отчёте («фильтр срабатывал N раз, агрегаты не сдвинулись»),
не скрывается за плоской таблицей.

## 7. Бюджет

Худший случай A(12)+A3(8)+B(8)=28 клеток при ~170-190с/клетка
(max_gen=32/G_STALL=20, наблюдение LIFE-4..6) ≈ 80-90 мин CPU в фоне;
лучший случай (WORKS сразу) A(12)+A2(4)=16 клеток ≈ 45-50 мин. Один
сеанс, не дробить на 7a/7b.

## 8. Файлы

```
agent_life7_selection/
  PROTOCOL.md README.md BLOCKERS.md
  src/  scripts/  metrics/  runs/  reports/REPORT_LIFE7.md
```
