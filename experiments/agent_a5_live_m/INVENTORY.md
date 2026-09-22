# INVENTORY.md — agent_a5_live_m

Провенанс-ведомость (в духе `agent_a3_mstar/INVENTORY.md`): что скопировано
без изменений, что скопировано с изменениями, что написано с нуля. A5 —
больший структурный отход от A2→A3→A4 (живой backend, не offline-replay),
поэтому эта ведомость восстановлена как отдельный файл (A4 обошлась без
неё, но там менялся только оператор рекомбинации/параметр, не источник
данных).

## Импорт без изменений (arch2/, experiment14/, experiment11/)

- `arch2/evolve.py` (`Evolution`, `split_pools` — не используется, свой
  `split_pools_a5`), `arch2/genotype.py`, `arch2/mutate.py`, `arch2/runner.py`,
  `arch2/fitness.py` — весь модуль.
- `arch2/heterostep.py` — `HeterostepBackend`, `Registry`, `MODEL_IDS`,
  `STEP_KINDS`, `PROMPT_VARIANT`, `TEMPERATURE`, `MAX_TOKENS_STEP`,
  `SEAM_ID`. **НЕ** используется: `Dataset` (хардкодит старые seed'ы
  генератора задач внутри `__init__` — см. `src/live_dataset.py`
  докстринг), `default_dataset`/`default_registry` (синглтоны на процесс —
  здесь вместо этого своя `LiveDataset` и СВЕЖИЙ `Registry()` на каждый
  seed, см. §"Осознанные отклонения" ниже).
- `arch2/heterostep_seeds.py` — весь модуль (`b1_route`/`b2_route`/
  `b3_route`/`route`/`seed_complexes`/`REFERENCE_NAMES`/`BASELINE_NAME`/
  `GATE_REFERENCE_NAME`).
- `experiment14/tasks/heterostep.py` — `build_tasks`/`split` (вызваны с
  НОВЫМИ seed-аргументами, PROTOCOL.md §2), `step_prompt`/`whole_prompt`
  (без изменений).
- `experiment14/seams.py` — весь модуль (`numeric_seam`/`token_seam`/
  `check_step`/`whole_seam`/`self_test`).
- `experiment11/configs/model_registry.py` — `load_model`/`generate`/
  `MODEL_IDS`.

## Скопировано (не импортом — изоляция пакетов `agent_aN`), с точечными правками

- `src/recombination.py::swap_assemble_slot` ← `agent_a4_m_plateau/src/
  recombination.py`. Правка: только строка `origin` (`"mutate:A5_swap_
  assemble_slot"` вместо `"...A4..."`). Логика оператора идентична.
- `src/orchestrator.py::build_seeds_pop20` ← `agent_a4_m_plateau/src/
  orchestrator.py::build_seeds_pop20`. Правка: константа RNG для
  случайного добора популяции (`SEED_SEEDS_RNG=20260820` вместо A4's
  `20260999` — собственное значение пакета, не пересекается ни с чем).
  Состав/структура НЕ менялись.
- `src/orchestrator.py::custom_reproduce` ← `agent_a4_m_plateau/src/
  orchestrator.py::custom_reproduce`. Логика (донор-пул без `ev.references`,
  клэмп `min_frac_local`) идентична; здесь клэмп при `p_cross=0.60`
  никогда не включается (см. PROTOCOL.md §6) — не правка кода, следствие
  другого фиксированного `p_cross`.

## Написано с нуля (специфично для LIVE)

- `src/live_dataset.py::LiveDataset` — единственное необходимое отличие
  от `arch2.heterostep.Dataset`: параметризованные seed'ы генератора задач
  (`build_tasks(n=200, seed=150001)`/`split(..., seed=20260820)` вместо
  вызова без аргументов). Форма (`.cells`/`.models`/`.kinds`/
  `.initial_state`/`.coverage`) идентична, чтобы `HeterostepBackend`/
  `heterostep_seeds` работали без изменений.
- `src/live_grid_builder.py` — единственное место, где вызывается
  `generate()` для шаговой сетки (PROTOCOL.md §4). Модель-мажорный обход,
  резюмируемость по (task_id,kind,model), запись в `live_call_log.jsonl`.
- `src/call_log.py` — общий писатель анти-фрод журнала.
- `src/baselines.py` — B0 (живые целостные вызовы, данные решают
  победителя) + B1-B3 (чистая конструкция генотипа из `live_grid`).
- `src/verdict.py::compute_basket` — механическая реализация
  предрегистрированной формулы PROTOCOL.md §8.
- `scripts/verify_seams.py` — швы + ровно один смоук-вызов (PROTOCOL.md §1).
- `scripts/run_all.py` — сквозной драйвер шагов 2-6 + генерация
  `REPORT_LA5.md`.

## Осознанные отклонения от A4's паттерна (не баг, разница в требованиях)

- **Свежий `HSTEP.Registry()` на каждый seed**, а не общий
  `HSTEP.default_registry()` синглтон на процесс (который использовали
  A2/A3/A4 для ВСЕХ своих ячеек в одном процессе). PROTOCOL.md §6 требует
  5 НЕЗАВИСИМЫХ эволюционных прогонов; общий Registry рискует протечкой
  зарегистрированных композитов из seed N в `generator_ids()` seed'а N+1.
  Задокументировано в докстринге `src/orchestrator.py`, не отдельным
  BLOCKERS-пунктом (это не дефект, а более строгий выбор для более
  строгого требования).
- `LiveDataset` (Dataset) переиспользуется ОДИН экземпляр на все 5 seed
  (через `LD.default_dataset()`) — это КОРРЕКТНО, а не небрежность: сами
  данные (`live_grid`) общие и фиксированные по построению, независимость
  нужна только у ПОИСКА (Registry/Evolution), не у данных.
