# Математическое описание реализованного кода

Строго по коду. Каждая формула выписана из конкретного файла и строки; ничего не
достроено и не обобщено. Где величина объявлена, но не вычисляется, это указано явно.

Обозначения: `C = LLAMA_MYC_MAX_CORRIDORS = 32` (число псевдоэкспертов),
`E = LLAMA_MYC_MAX_EXPERTS_RUNTIME = 256`, `L = LLAMA_MYC_MAX_LAYERS_RUNTIME`.

---

## 0. Базовые операторы

**EMA-сглаживание** (`myc-fields.h`), используется повсеместно:

```
EMA(x_prev, x, α) = α·x_prev + (1−α)·x
```

Константы затухания, объявленные в коде:

| Константа | Значение | Где применяется |
|---|---|---|
| `MYC_EMA_LAYER` | 0.90 | объявлена, **в текущем коде не используется** |
| `MYC_EMA_CORRIDOR` | 0.92 | `usage_ema` коридоров |
| `MYC_EMA_KV` | 0.95 | все поля KV |
| `MYC_EMA_SPEC` | 0.90 | спекулятивные EMA, `n_ema` |
| `MYC_EMA_FIELD` | 0.88 | `corridor_heat`, `j_attn`, дрейф поведения |

**Ограничение**: `clamp(v, lo, hi) = max(lo, min(v, hi))`.

---

## 1. Нарезка модели на псевдоэкспертов (`myc-slice.cpp`)

Модель из `n_layer` слоёв режется на `n_c` непрерывных полос:

```
n_c   = min(C, n_layer)
base  = ⌊n_layer / n_c⌋
rem   = n_layer mod n_c
span(c) = base + [c < rem]
```

`first(c)` — префиксная сумма `span`. Отображение слоя в псевдоэксперта:
`corridor_of_layer(l) = c`, где `first(c) ≤ l < first(c) + span(c)`.

Свойство по построению: размеры полос отличаются не более чем на 1, каждый слой
принадлежит ровно одной полосе, пустых псевдоэкспертов нет.

**Теплота полосы** — среднее по слоям полосы:

```
H_band(c) = (1/span(c)) · Σ_{l ∈ band(c)} hotness_ema(l)
```

---

## 2. Поле теплоты (`myc-heat.cpp`)

**Скалярная выборка поля** — взвешенная сумма пяти компонент:

```
h(x) = w₀·hotness + w₁·reuse + w₂·semantic + w₃·corridor + w₄·migration
```

Веса — параметры поля, инициализированные значениями (`myc-fields.h`):

```
w₀ = +0.35   (hotness)
w₁ = +0.30   (reuse)
w₂ = +0.20   (semantic)
w₃ = +0.15   (corridor)
w₄ = −0.25   (migration)
```

Прочие параметры поля: `decay = 0.90`, `temperature = 1.0`, `threshold_base = 0.45`.

**Выборка по слою** (вызывается из `cb()` графа с `access_weight = 1` для
исполнившегося слоя, и из `myc_heat_step` с `0` для затухания):

```
hotness_ema(l) ← EMA(hotness_ema(l), access_weight, decay)
```

**Выборка по псевдоэксперту** — подстановка в `h(x)`:

```
activation = usage_ema(c)
background = H_band(c)                     если нарезка построена
             (1/n_layer)·Σ hotness_ema(l)  иначе
reuse      = clamp(j*(c), 0, 1) если j*(c) > 0, иначе 0
corridor   = 1 если state(c) ∈ {HOT, ACTIVE}, иначе 0
migration  = kv_pressure(c)

corridor_heat(c) ← EMA( corridor_heat(c),
                        clamp(h(activation, reuse, background, corridor, migration)·T, 0, 1),
                        MYC_EMA_FIELD )
```

**Наблюдаемые поля**:

```
mean     = (1/C)·Σ_c corridor_heat(c)
max      = max_c corridor_heat(c)
gradient = |mean(t) − mean(t−1)|
```

---

## 3. Гильбертово поле (`myc-hilbert.cpp`)

Размерность `D = 4`, разрядность оси `B = 8`, число соседей `K = 8`.

**Фазовое пространство** — четыре оси, каждая квантуется в целую сетку:

```
q(v, α) = round( clamp(v·α, 0, 1) · (2^B − 1) )

x₀ = q(corridor_heat(c), α₀)
x₁ = q(usage_ema(c),     α₁)
x₂ = q(kv_pressure(c),   α₂)
x₃ = q(J(c)/2 + 1/2,     α₃)
```

`α_i` — веса осей, инициализированы в 1.0.

**Индекс кривой** — преобразование Скиллинга (точное, не аппроксимация):
обратное развёртывание вращений, код Грея по осям, затем чередование битов осей в
единый ключ `key(c) ∈ [0, 2³²)`.

**Расстояние** — модуль разности ключей:

```
d(i,j) = |key(i) − key(j)|
```

**Окрестность** — K ближайших по `d`, отсортированных по возрастанию, без петель.

**Стоимость локальности**:

```
C_hilbert = Σ_i Σ_{k ∈ N(i)} corridor_heat(i)·corridor_heat(k) · d(i,k)/(2^{2B} − 1)

mean_dist = (1/|links|) · Σ d(i,k)
```

---

## 4. Термодинамический рой (`myc-swarm.cpp`)

**Сборка по теплоте.** Вес псевдоэксперта в ансамбле — собственная теплота,
модулированная когерентностью окрестности:

```
w(i,k)   = 1 / (1 + d(i,k)·10⁻⁴)
support(i) = Σ_k w(i,k)·corridor_heat(k) / Σ_k w(i,k)

A(i) = corridor_heat(i) · (0.5 + 0.5·support(i))
A_total = Σ_i A(i)
```

**Локальное производство энтропии** — расхождение с окрестностью:

```
σ(i) = (1/|N(i)|) · Σ_{k ∈ N(i)} (corridor_heat(i) − corridor_heat(k))²
σ_total = Σ_i σ(i)
```

`σ(i) ≥ 0` по построению (сумма квадратов).

**Конденсация диссипативных структур.** Порог относительный:

```
Ā   = A_total / C
θ   = 1.25·Ā
```

Заливка по графу гильбертовых окрестностей: узел `i` входит в структуру, если
`A(i) ≥ θ` и он связан рёбрами окрестности с уже принятым узлом. Максимум структур —
`MYC_MAX_STRUCTURES = 8`.

Масса и когерентность структуры `s`:

```
M(s)  = Σ_{i ∈ s} A(i)
Coh(s) = (1/|links(s)|) · Σ_{i,k ∈ s} 1/(1 + d(i,k)·10⁻⁴)
```

**Параметр порядка ансамбля**:

```
share = max_s M(s) / A_total
split = 1 / n_structures

order = clamp( share · (0.5 + 0.5·split), 0, 1 )
```

При равномерном поле все `A(i)` равны `Ā < θ` ⇒ ни один узел не проходит порог ⇒
`n_structures = 0`, `order = 0`. Это проверяется тестом.

---

## 5. Граф взаимодействия (`myc-graph.cpp`)

**Топология** — детерминированная, двудольная, много-ко-многим:

```
degree = min(8, n_expert)
to(c,k) = (7c + 13k + 1) mod n_expert,   k = 0 … degree−1
w₀ = 1/degree                             (начальный вес каждого ребра)
```

**Поля узлов**:

```
node_activity(c) = usage_ema(c)
node_field(c) = h(c) = usage_ema(c) + j*(c)
```

**Связи** индуцируются общей целью: два коридора связаны через эксперта, в которого
оба проецируются:

```
J(a,b) = weight(a)·weight(b),   если to(a) = to(b) и from(a) ≠ from(b)
```

**Релаксация спинов** — один синхронный проход при нулевой температуре:

```
local(i) = h(i) + Σ_{J(a,b)} J·s(партнёр)
s(i) ← sign(local(i)) ∈ {−1, +1}
```

**Наблюдаемые**:

```
H(s) = −Σ_{пары} J(a,b)·s_a·s_b − Σ_i h(i)·s(i)

C_graph = Σ_{пары} J(a,b)·(u_a − u_b)²      где u = node_activity   (дирихлеева форма)

p_conn = |{c : ∃ ребро из c с весом > 10⁻⁴}| / C
```

**Обратная связь (накопление).** Хеббовское правило по фактическому исходу
`argsort_top_k`:

```
fb     = clamp(1 + N_ema, 0.1, 1.5)
target = node_activity(from)  если selected_count(to) > 0, иначе 0

Δ(i) ← Δ(i) + η_pl · fb · (target − weight(i)),   η_pl = LLAMA_MYC_GRAPH_PLASTICITY = 0.05
```

Веса **не меняются** на этом шаге — только накапливается `Δ`, счётчик `pending_steps++`.

**Коммит на границе эпохи** (усреднение по числу накопленных шагов):

```
weight(i) ← clamp( weight(i) + Δ(i)/pending_steps, 0, 1 )
Δ ← 0,  pending_steps ← 0
```

Это реализация требования `L_active(t) = L_snapshot ∀t ∈ epoch`.

---

## 6. Проекция на физических экспертов (`myc-energy.cpp`)

```
Energy(e) = Σ_{ребро (c → e)} A(c) · weight(c → e)
```

Пересчитывается с нуля каждый шаг (не накапливается), поэтому величина ограничена по
построению. Потребляется вес **в рое** `A(c)`, а не сырая активность коридора.

---

## 7. Термодинамика (`myc-thermo.cpp`)

### 7.1 Негэнтропия

Компоненты (измеряемые):

```
J_kv   = clamp(hit_rate, 0, 1)
J_graph = clamp(p_conn, 0, 1)
J_spec = clamp(acceptance_rate_ema, 0, 1)
J_attn = EMA(J_attn, 1 − H_attn/H_max, MYC_EMA_FIELD)     публикуется графом
J_behavior = clamp(1 − D_B, 0, 1)                          публикуется поведением
```

Свёртка:

```
J_total = 0.35·J_attn + 0.20·J_kv + 0.25·J_graph + 0.10·J_behavior + 0.10·J_spec
```

**J_attn = 0 при включённом flash-attention** — распределение внимания не
материализуется, измерения нет; ноль означает «нет измерения», прокси не подставляется.

### 7.2 Составное давление

```
P = clamp( 0.25·occupancy + 0.15·fragmentation + 0.25·contention
         + 0.10·migration + 0.15·transport + 0.10·rigidity , 0, 1 )
```

где `occupancy = kv_occupancy_frac`, `contention = heat.cell_contention`,
`transport = C_hilbert`, `rigidity` берётся **на шаг старше** (см. §9 логики).
Веса в сумме дают 1, поэтому `P ∈ [0,1]` при любых входах.

### 7.3 Свободная энергия (среднее поле)

Двухуровневая система на каждом узле, `β = 1` по инициализации:

```
ln Z = Σ_i ln( 2·cosh(β·h(i)) )
F    = −ln Z / β

p↑(i) = 1/(1 + e^{−2βh(i)}),   p↓ = 1 − p↑
S     = (1/C) · Σ_i [ −p↑ln p↑ − p↓ln p↓ ]
```

### 7.4 Баланс Пригожина

```
Φ_diss = σ_total + dissipation_ema(KV)
Φ_in   = coherence + J_total + order

χ = (Φ_in − Φ_diss)/Φ_diss     при Φ_diss > 10⁻⁶, иначе 0

d_iS/dt = Φ_diss   (≥ 0 по построению)
d_eS/dt = −J_total
```

### 7.5 Параметр порядка

Уравнение Ландау, один шаг Эйлера:

```
a = clamp(χ − χ_c, −1, 1),   χ_c = 0
τ_η = 8,   b = 1

Δη = (a·η − b·η³)/τ_η
η ← clamp( η + Δη , 0, 1)
```

Затравка: если `η < 10⁻³` и `a > 0`, то `η ← 10⁻²` — иначе система не может покинуть
неустойчивую точку `η = 0` (при `η = 0` производная тождественно ноль).

### 7.6 Функция Ляпунова

```
V(R) = 0.30·P + 0.20·migration_instability + 0.20·collapse_rate
     + 0.10·C_hilbert − 0.40·J_total
```

где

```
migration_instability = min(1, fragmentation + 0.5·pending_count/QUEUE_DEPTH)
collapse_rate = collapse_rate_ema (спекулятивный)
```

### 7.7 Легаси-скаляр негэнтропии

Сохранён, используется в экспорте и кольце телеметрии:

```
N_current = coherence − collapse_rate − migration_instability
N_ema     = EMA(N_ema, N_current, MYC_EMA_SPEC)
```

где `coherence = max_c usage_ema(c)`.

---

## 8. Поведение (`myc-behavior.cpp`)

Вектор состояния `B(t)`, `dim = 8`, все компоненты нормированы в источнике:

```
B = [ H_attn, curvature, drift_logit, kv_pressure,
      structure_order, η, acceptance_rate_ema, p_conn ]
```

```
D_B_raw = ‖B(t) − B(t−1)‖₂ / √dim
D_B     = EMA(D_B, clamp(D_B_raw, 0, 1), MYC_EMA_FIELD)
I_s    += D_B_raw
```

Деление на `√dim` — максимально возможный шаг (все компоненты 0→1 одновременно),
поэтому `D_B ∈ [0,1]` по построению.

**Первая выборка ничего не публикует**: без `B(t−1)` дрейф не определён, а не равен нулю.

---

## 9. KV (`myc-kv.cpp`)

```
pressure ← EMA( pressure, clamp(0.6·occupancy + 0.4·contention, 0, 1), MYC_EMA_KV )
fragmentation = clamp(occupancy · miss_rate · 2, 0, 1)
dissipation_ema ← EMA(dissipation_ema, kv_evicted/n_kv_cells, MYC_EMA_KV)
```

Из публикации наблюдения кэша (`taken = n_clean + n_displaced > 0`):

```
clean_frac = n_clean / taken
hit_rate  ← EMA(hit_rate,  clean_frac,     MYC_EMA_KV)
miss_rate ← EMA(miss_rate, 1 − clean_frac, MYC_EMA_KV)
retention_rate ← EMA(retention_rate, clean_frac, MYC_EMA_KV)

cell_contention ← EMA(cell_contention, n_protected/n_examined, MYC_EMA_KV)
cell_mean_heat  ← EMA(cell_mean_heat,  mean_heat,              MYC_EMA_KV)
```

---

## 10. Спекуляция (`myc-spec.cpp`)

```
acc = accepted/drafted
acceptance_rate_ema ← EMA(·, acc, MYC_EMA_SPEC)
collapse_rate_ema   ← EMA(·, [accepted = 0], MYC_EMA_SPEC)
speedup_ema         ← EMA(·, t_classic/t_spec, MYC_EMA_SPEC)   при обоих > 0
```

**Окно черновика** `k`, возвращаемое `spec_should_run`, равно 0 (классический путь), если
выполнено любое из: `force_spec_off`, фаза `EMERGENCY`, `H_attn > h_bifurc_threshold`,
`σ > sigma_emerg_threshold`, `0 < V(R) < v_lyapunov_min`. Иначе:

```
σ = 0.45·H_attn + 0.35·curvature + 0.20·drift
k = k_draft;  если acceptance_rate_ema < 0.5 то k ← max(1, ⌊k/2⌋)
k ← min(k, k_draft_max)
```

---

## 11. Граничные условия (`myc-boundary.cpp`)

```
rigidity = clamp( 0.4·min(n_pinned/64, 1) + 0.4·[veto ≠ NONE] + 0.2·[frozen] , 0, 1 )
```

Позиция удержана, если `pos < n_sink_protect` или `pos ∈ pinned[]`.

---

## 12. Guardian (`myc-guardian.cpp`)

Проверки в порядке эскалации, первая сработавшая возвращается:

```
1. VRAM:        vram_used/vram_total > vram_pressure_max      (0.90)
2. KV:          kv.pressure > kv_pressure_max                 (0.85)
3. Катастрофа:  σ > variance_spike_max·(1 − 0.5·D_B)          (0.45·H+0.35·curv+0.20·drift ; порог 1.20)
4. Качество:    0 < V(R) < v_lyapunov_min                     (0.30)
```

Дрейф поведения **ужесточает порог** третьей проверки, а не добавляет пятую: при
`D_B = 1` порог уменьшается вдвое.

---

## 13. Резидентность (`myc-slice.cpp`)

Относительно среднего по ансамблю `Ā = A_total/n_c`:

```
класс(c) = HOT   если A(c) > 1.25·Ā
           WARM  если A(c) > 0.75·Ā
           COLD  иначе

если structure_id(c) ≥ 0 и класс = COLD, то класс ← WARM
```

Слои с `lock_weights` или `pinned` не переклассифицируются. Применяется только когда
`boundary.frozen = false`.

---

## 14. Длина эпохи (`myc-epoch.cpp`)

```
stability = clamp( 0.5·J_total + 0.3·order − 0.3·P − 0.2·collapse_rate + 0.5 , 0, 1 )
length    = 8 + ⌊stability·(256 − 8)⌋
```

Аварийная эпоха имеет фиксированную длину 8 (`MYC_EPOCH_MIN_LEN`).

---

## 15. Энтропия внимания (`llama-graph.cpp`, `llama-context.cpp`)

Считается в графе на материализованном тензоре после `ggml_soft_max_ext`:

```
H_sum = −Σ_{все элементы} p · ln( clamp(p, 10⁻⁹, 1) )
```

Клэмп защищает только логарифм; умножение использует неклэмпнутое `p`, поэтому `p = 0`
даёт вклад ровно 0.

Нормировка при чтении:

```
H_norm = (H_sum / n_dists) / ln(n_kv)
```

где `n_kv = ne[0]` тензора, `n_dists = nelements/ne[0]` (число распределений = головы ×
токены). Инструментуется **только последний слой внимания**.

---

## 16. Что объявлено, но не вычисляется

Честный перечень полей, существующих в структурах, но не имеющих вычислителя в коде:

| Поле | Статус |
|---|---|
| `heat.threshold_base` | читается кэшем KV через `observable`, но сам кэш применяет свою формулу порога |
| `hilbert.axis_weight[]` | инициализируются в 1.0, никогда не изменяются |
| `flow.j[]` | интегрируется, но **ни один потребитель не читает** результат релаксации |
| `thermo.partition_log` | вычисляется, экспортируется, ни на что не влияет |
| `thermo.entropy`, `free_energy` | вычисляются, экспортируются в `observable`, ни на что не влияют |
| `thermo.ds_ext`, `ds_int` | вычисляются, ни на что не влияют |
| `swarm.structure_coherence[]` | вычисляется, ни на что не влияет |
| `behavior.inertia` (`I_s`) | накапливается, доступна через API, ни на что не влияет |
| `MYC_EMA_LAYER` | константа объявлена, не используется |

Это не ошибки — это величины, вычисленные и доступные наблюдателю, но пока не
замкнутые ни в один контур управления.
