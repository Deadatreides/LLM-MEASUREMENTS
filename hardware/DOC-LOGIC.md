# Логика реализованного кода

Строго по коду: структура владения, порядок выполнения, потоки данных, точки
интеграции с llama.cpp. Формулы вынесены в `DOC-MATH.md`.

---

## 1. Структура системы

Организм — физическая система с глобальными полями. Подсистемы не командуют друг
другом: они читают поля и публикуют в них наблюдения. Runtime — один орган из
нескольких, а не центр управления.

```
llama_context
  └─ mycelium : llama_mycelium_organism*     прямой член, создаётся в конструкторе
       │
       ├── ГЛОБАЛЬНЫЕ ПОЛЯ (физическая система)
       │    slice      — нарезка модели на псевдоэкспертов
       │    behavior   — B(t), дрейф, семантическая инерция
       │    heat       — поле теплоты (слои / коридоры / KV-клетки)
       │    hilbert    — многомерная кривая локальности
       │    swarm      — сборка ансамбля, диссипативные структуры
       │    thermo     — свободная энергия, негэнтропия, параметр порядка
       │    boundary   — граничные условия (pinned, sink, veto, rigidity)
       │    graph      — связи J_ij, спины
       │
       ├── ОРГАНЫ
       │    runtime    — corridors, kv, spec, energy, flow, pending, ring
       │    guardian   — пороги вето
       │    eviction   — очередь эвикций (фиксированная ёмкость 64)
       │
       └── p1 : llama_mycelium_impl*          MCB, обращённый к Python
```

### Владение полями — один писатель на поле

| Поле | Единственный писатель |
|---|---|
| `heat.*` | `myc-heat.cpp` |
| `hilbert.*` | `myc-hilbert.cpp` |
| `swarm.*` | `myc-swarm.cpp` |
| `thermo.*` | `myc-thermo.cpp` |
| `boundary.*` | `myc-boundary.cpp` |
| `behavior.*` | `myc-behavior.cpp` |
| `slice.*` | `myc-slice.cpp` |
| `graph.*` | `myc-graph.cpp` |
| `runtime.corridors.*` | `myc-corridors.cpp` |
| `runtime.kv.*` | `myc-kv.cpp` |
| `runtime.spec.*` | `myc-spec.cpp` |
| `runtime.energy.*` | `myc-energy.cpp` |
| `runtime.flow.*` | `myc-flow.cpp` |
| `runtime.pending.*` | `myc-epoch.cpp` |
| `runtime.ring.*` | `myc-telemetry.cpp` |
| `guardian.*` | `myc-guardian.cpp` |
| `heat.layers[].state` | `myc-slice.cpp` (не heat — см. §8) |

Исключение: `myc-slice.cpp` пишет `heat.layers[].state`, потому что резидентность —
решение роя, а не измерение теплоты. Правило проверяется статическим прогоном
(`no multi-writer fields detected`).

---

## 2. Жизненный цикл

```
llama_context::llama_context()
    mycelium = llama_mycelium_runtime_create(this)
        p1 = llama_mycelium_p1_create(ctx)
        инициализация всех полей
    llama_mycelium_slice_ensure(this, hparams.n_layer)     нарезка модели

llama_context::~llama_context()
    llama_mycelium_runtime_destroy(mycelium)
        llama_mycelium_p1_destroy(p1)
```

Реестра нет, мьютексов нет. Разрешение контекста в организм —
`llama_mycelium_runtime_of(ctx)`, реализовано в `llama-context.cpp` (единственная
единица трансляции, знающая layout `llama_context`) и возвращает `ctx->mycelium`.

`llama_mycelium_init` / `runtime_ext_init` сохранены для Python-моста, но больше не
аллоцируют — они разрешают. `teardown` не освобождает память (владеет контекст), а
отключает блок.

Для автономной сборки substrate (`LLAMA_MYCELIUM_STANDALONE`) резолвер поставляется
`myc-standalone.cpp` — таблица на 8 слотов, только для тестов, в продакшн-пути не
компилируется.

---

## 3. Шаг организма — порядок и зависимости

`llama_mycelium_full_step_hook` вызывается из `llama_decode` **после** `ctx->decode()`.

| # | Операция | Читает | Пишет |
|---|---|---|---|
| 1 | `post_decode_hook` (P1) | логиты | `mcb.runtime.drift`, `curvature` |
| 2 | `myc_kv_report_step` | аргументы хука | `kv.hit_rate/miss/retention` |
| 3 | `myc_kv_thermo_update` | `kv_stats`, `heat.cell_contention` | `kv.pressure`, `fragmentation`, `dissipation_ema` |
| 4 | `myc_corridors_update_step` | `mcb.flow.j_star` | `corridors.meta[].usage_ema`, `j_star` |
| 5 | `myc_spec_step` | аргументы хука | `spec.*` |
| 6 | `myc_flow_relax_step` | `mcb.flow` | `flow.j[]` |
| 7 | `coherence = max usage_ema` | corridors | — |
| 8 | `myc_heat_step` | slice, corridors | `heat.layers[].hotness_ema`, `corridor_heat`, `mean/max/gradient` |
| 9 | `myc_hilbert_step` | heat, corridors, flow | `hilbert.key/neighbor/dist/cost` |
| 10 | `myc_swarm_step` | heat, hilbert | `swarm.assembly_weight/dissipation/structure_*` |
| 11 | `graph load/relax/observables` | corridors | `graph.node_*`, `hamiltonian`, `laplacian_cost`, `connectivity` |
| 12 | `myc_energy_propagate` | graph, swarm | `energy.energy[]` |
| 13 | `myc_thermo_step` | kv, graph, swarm, hilbert, heat, **boundary(t−1)** | `thermo.*` |
| 14 | `myc_behavior_step` | mcb, kv, swarm, thermo, spec, graph | `behavior.*`, `thermo.j_behavior` |
| 15 | `myc_guardian_check` | `kv.pressure`, `behavior.drift`, `V(R)` | `mcb.veto`, `guardian.total_vetos` |
| 16 | `myc_boundary_step` | `p1.active.pinned`, `mcb.epoch`, `mcb.veto` | `boundary.*` |
| 17 | `myc_slice_step` | swarm, `boundary.frozen` | `heat.layers[].state` |
| 18 | `graph_accumulate_feedback` | `energy.selected_count`, `graph.node_activity` | `graph.pending_delta`, `pending_steps` |
| 19 | `myc_energy_clear_selection` | — | `energy.selected_count` ← 0 |
| 20 | `myc_epoch_on_step` | thermo, `boundary.veto` | `mcb.epoch`, коммит границы |
| 21 | `myc_telemetry_write` | всё | `runtime.ring` |

### Два намеренных запаздывания на шаг

Оба задокументированы в коде, оба неизбежны — это обратные связи, и одно ребро цикла
обязано нести задержку:

1. **`thermo.pressure` ← `boundary.rigidity(t−1)`.** Цикл:
   `pressure → rigidity → veto → guardian → V(R) → pressure`. Ребро выбрано
   естественно: граничные условия по определению есть то, что зафиксировано на прошлом
   коммите.

2. **`j_behavior` попадает в `J_total` следующего шага.** Поведение считается после
   термодинамики (нужен `η`) и до гвардиана (нужен дрейф). Это не баг задержки: дрейф
   есть свойство только что завершённого перехода, а не собираемого сейчас состояния.

---

## 4. Точки интеграции с llama.cpp

| Файл | Что вставлено |
|---|---|
| `llama-context.h` | член `mycelium` |
| `llama-context.cpp` | создание/уничтожение организма, `llama_mycelium_runtime_of`, `slice_ensure`, обратное чтение выбранных экспертов, обратное чтение энтропии внимания, реальное давление KV в хук |
| `llama-graph.h` | поле `lctx`, входной класс смещения, `t_mycelium_selected`, поля энтропии внимания |
| `llama-graph.cpp` | смещение маршрутизации перед `argsort_top_k`, `ensure_topology`, узлы энтропии внимания, подача теплоты слоёв из `cb()` |
| `llama-kv-cache.h` | `lctx_cached`, счётчики `slot_clean`/`slot_displaced` |
| `llama-kv-cache.cpp` | наблюдение состояния организма в `find_slot`, применение граничных условий, публикация наблюдения |
| `common/speculative.cpp` | окно черновика через существующий `n_max`, доклад верификации |

### Влияние на маршрутизацию MoE

```
build_moe_ffn()
    selection_probs = <штатный конвейер: логиты → гейтинг → exp_probs_b →
                       архитектурные ветки → групповая маскировка>
    ┌── если lctx ≠ null:
    │     llama_mycelium_graph_ensure_topology(lctx, n_expert)
    │     bias = новый входной тензор [n_expert]
    │     selection_probs = ggml_add(selection_probs, bias)      ← ЕДИНСТВЕННОЕ ВЛИЯНИЕ
    └──
    selected_experts = ggml_argsort_top_k(selection_probs, n_expert_used)   без изменений
    ggml_set_output(selected_experts)                                       для обратной связи
    <агрегация, взвешивание: без изменений>
```

Смещение добавляется **после** всех архитектурных веток и групповой маскировки,
поэтому применяется единообразно и не может быть искажено маскировкой (конечное
смещение к `−INFINITY` остаётся `−INFINITY`).

Значения тензора заполняются в `set_input` — **после** аллокации графа, поэтому форма
графа никогда не зависит от состояния организма.

Формы тензоров, `n_expert`, `n_expert_used`, топология графа и логика агрегации не
изменены нигде.

### Влияние на KV-кэш

```
find_slot()
    llama_mycelium_observe(lctx_cached, &obs)          состояние, не политика
    ┌── кэш САМ интерпретирует:
    │     heat_strict |= (pressure > 0.70 ∨ veto ≠ 0 ∨ rigidity > 0.75)
    │     heat_limit   = threshold_base·(1 − 0.5·pressure) + 0.25·heat_mean
    └──
    для каждой клетки-кандидата:
        если позиция удержана граничными условиями → отказ
        если heat_strict и heat_score > heat_limit → отказ
    llama_mycelium_kv_publish(&observation)            публикация выборок в поля
```

Организм не выдаёт кэшу политику. Он предоставляет состояние; интерпретация —
работа кэша, потому что выбор слота это его работа.

### Влияние на спекуляцию

```
common_speculative_draft()
    k = llama_mycelium_spec_should_run(spec->ctx_tgt)
    для каждой активной последовательности:
        k = 0        → drafting = false          (классический путь этот шаг)
        k < n_max    → n_max = k                 (только ужесточение)
```

Организм может укоротить или запретить черновик, но не удлинить сверх разрешённого
вызывающим.

---

## 5. Замкнутые контуры

### Маршрутизация (модели MoE)

```
доступ к слою (cb) → теплота слоя → теплота полосы → теплота псевдоэксперта
    → гильбертова окрестность → вес в рое → энергия эксперта
    → смещение selection_probs → argsort_top_k → фактически выбранные эксперты
    → обратное чтение в process_ubatch → selected_count
    → накопление Δ весов рёбер → КОММИТ НА ГРАНИЦЕ ЭПОХИ → веса связей
```

### Резидентность (плотные модели)

```
доступ к слою → теплота полосы → вес в рое → диссипативные структуры
    → класс резидентности слоя (HOT/WARM/COLD)
```

Это единственный контур, работающий на плотной модели: у неё нет физических экспертов,
и проекция энергии ни на что не приземляется.

### Давление KV

```
find_slot: чистые/вытесненные клетки, удержанные границей
    → cell_contention, hit_rate → kv.pressure
    → вето гвардиана / составное давление → observable → следующий find_slot
```

### Негэнтропия

```
энтропия внимания (граф)     → J_attn      (0 при flash-attention)
hit_rate (KV)                → J_kv
связность (граф)             → J_graph
дрейф поведения              → J_behavior  (со сдвигом на шаг)
приёмка спекуляции           → J_spec
    → J_total → χ → η, V(R) → длина эпохи, окно спекуляции, вето
```

---

## 6. Эпоха как граница согласованности

Требование `L_active(t) = L_snapshot ∀t ∈ epoch` реализовано так: структурные изменения
не применяются внутри эпохи, а копятся.

```
IDLE     → организм сам начинает эпоху, длина = f(стабильность) ∈ [8, 256]
ACTIVE   → step_in_epoch++;  структура заморожена
           веса рёбер: копятся Δ, не применяются
           резидентность: не переклассифицируется (boundary.frozen)
           очередь pending: копится
           V(R) критично и вето активно → КОММИТ + EMERGENCY
исчерпана → КОММИТ → IDLE

КОММИТ = myc_graph_commit  (Δ/pending_steps → веса)
       + myc_epoch_pending_flush
       + llama_mycelium_kv_flush_eviction_queue
```

Внешнее управление главнее: `llama_mycelium_epoch_begin` из control plane задаёт фазу,
этот код её несёт. Но отсутствие control plane больше не означает, что механика эпох
не работает вовсе.

`step_in_epoch` инкрементируется ровно в одном месте (`myc-epoch.cpp`).

---

## 7. Граничные условия

Pinned-позиции и sink-диапазон — не список и не политика, а граничные условия: области,
где поля удерживаются фиксированными. Живут рядом с Epoch и Guardian.

Используется **закоммиченный** набор (`p1.active.pinned_positions`), а не отложенный из
`mcb.residency` — эпохальная согласованность требует, чтобы шаг видел границу,
зафиксированную на прошлом коммите.

`rigidity` — скалярная сводка: сколько системы сейчас удержано. Питает составное
давление.

---

## 8. Почему теплота не классифицирует резидентность

Теплота — измерение. Резидентность — решение, и оно принадлежит рою через полосу
нарезки: важен вес псевдоэксперта в ансамбле, а не сырая температура одного слоя.

Ранее `myc_heat_touch_layer` классифицировал состояние прямо из `hotness_ema`, что
давало второго писателя на поле, которым владеет `myc-slice.cpp`. Они дрались каждый
шаг: вердикт роя писался в конце шага и перетирался в начале следующего, то есть жил
меньше одного шага.

Следствие для затухания: «какие слои существуют» — свойство нарезки, а не класса
резидентности, поэтому цикл затухания идёт по `slice.n_layer`.

---

## 9. Взаимодействие с Python-мостом

C ABI сохранён полностью: все 27 связанных символов сохраняют имена и сигнатуры,
layout `llama_mycelium_mcb` (P1) не изменён — от него зависят ctypes-зеркала.

`llama_myc_runtime_ext` (P2) реструктурирован, но Python отражает его заглушкой только
с полем `magic`, поэтому перемещённые поля никто на той стороне не читает.

Направления обмена:

```
Python → substrate:  flow.j_star, residency plan, pinned/prefetch,
                     spec window, emergency, epoch begin/end, пороги гвардиана
substrate → Python:  mcb.runtime (h_attn, curvature, drift, r_miss, t_bus),
                     mcb.epoch (v_lyapunov, фаза), mcb.veto, kv_stats,
                     кольцо телеметрии, export_json
```

---

## 10. Что не изменено в llama.cpp

Осознанно оставлено без изменений: `llama-model.cpp` (модель на чекпоинт, организм на
сессию — их слияние сломало бы разделение одной модели между контекстами),
`llama-batch.cpp`, интерфейс `llama-memory`, планировщик `ggml_backend_sched_*`,
примитивы ggml, владение буферами, механизм жизненного цикла тензоров
(`no_alloc` → `set_input` → `set_output`) — на нём вся интеграция и построена,
`llama-sampler.cpp` (нулевая связанность), механизм `argsort_top_k` и агрегации MoE.

---

## 11. Состояние сборки и проверки

- Компилятора в окружении разработки нет (`cl.exe`, `g++`, `clang++`, `cmake` отсутствуют). **Ничего не проверено компиляцией.**
- Проверено механически: отсутствие дублирующих символов, разрешимость всех идентификаторов и обращений к полям организма, объявленность всех вызовов из `llama.cpp`, отсутствие полей с двумя писателями.
- `llama.cpp` не имеет системы сборки в этом checkout и не может быть собран по причинам, предшествующим этой работе (отсутствуют ~15 заголовков).
- Собирается только substrate, автономно:
  `cmake -S substrate -B substrate/build && cmake --build substrate/build && ctest --test-dir substrate/build`
- Тесты (`test-myc-organism.cpp`) написаны и механически провалидированы, но **ни разу не исполнялись**.

---

## 12. Известные ограничения

| Ограничение | Следствие |
|---|---|
| `J_attn = 0` при flash-attention | `J_total` теряет член с весом 0.35; распределение внимания не материализуется |
| Инструментован один слой внимания | `H_attn` — выборка поля, не полное измерение |
| Один граф на все слои MoE | параметр `il` принимается, но не используется для дифференциации |
| Нарезка — равномерные непрерывные полосы | не адаптивная; ветвление/слияние топологии относится к медленному контуру MGE |
| `flow.j[]` интегрируется, но не читается | контур потока не замкнут |
| `entropy`, `free_energy`, `ds_ext/int`, `I_s`, `structure_coherence` | вычисляются и экспортируются, но не влияют ни на одно решение |
| Резидентность меняется только вне эпохи | при постоянно активной эпохе классификация не применяется |
