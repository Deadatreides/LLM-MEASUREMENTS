# ARCHITECTURE — Σ_v8.8.1 «Мицелий»

Полное техническое описание архитектуры роевой системы.

---

## 1. Общая философия

Σ «Мицелий» строится на трёх принципах:

**1. Диссипативная структура (Пригожин)**  
Система — открытая, существует за счёт потока негэнтропии из внешней среды.  
Условие существования: `|d_e S / dt| > σ`  
Каналы экспорта энтропии: FAISS (упаковка информации), DPO (конвертация ошибок в обучение), Thompson (изоляция плохих агентов).

**2. Край хаоса (Лэнгтон, 1990)**  
Максимальная вычислительная ёмкость системы достигается при λ_L ≈ 0.  
λ_L < 0 → детерминированный pipeline, нет новизны.  
λ_L > 0 → галлюцинации, нестабильность.  
PID-регулятор удерживает систему в этой точке через K_ACT.

**3. Разделение временных масштабов (Ренормгруппа)**  
Быстрые процессы (4–8s) интегрируются в медленные (5 мин), те — в ещё более медленные (5 ч).  
Нарушение соотношения `η_DPO : η_bandit : η_PID = 1 : 20 : 200` ломает иерархию.

---

## 2. Модули и их взаимодействие

```
orchestrator.py
│
├── core/bandit.py          SwarmBandit
│   ├── Thompson Sampling   выбор модели
│   ├── TemperatureBandit   LinUCB для выбора температуры API
│   └── Exp3 drift          детектор дрейфа наград
│
├── core/chaos.py           ChaosController
│   ├── LyapunovEstimator   λ_L по скользящему окну
│   └── PIDController       K_ACT = f(λ_L), β_KL = f(λ_L)
│
├── core/hypercycle.py      Hypercycle + AgentEnergy + NoveltySignal
│   ├── Hypercycle          квоты каст G/C/S с мутацией
│   ├── AgentEnergy         рождение/смерть/деление агентов
│   └── NoveltySignal       R_ext = внешняя энергия
│
├── core/memory.py          Memory (фасад)
│   ├── HybridEmbedder      BGE-m3 + E5-small → 1408d
│   ├── FAISSIndex          IVFFlat векторный поиск
│   ├── PhysarumGraph       граф каузальных связей
│   ├── SQLiteStore         полные тексты трасс
│   └── LeakyIntegrator     I(t) — RC-память
│
├── core/blackboard.py      SemanticBlackboard
│   ├── SemanticBlock       блок с версиями и энтропией
│   └── entropy_from_emb    Shannon через softmax
│
├── core/scorer.py          QualityScorer + EmergenceMetric
│   ├── QualityScorer       Q_ext, q_env (_q_code_safe), consistency
│   └── EmergenceMetric     E_total = E_base + delta_I
│
├── tools/search.py         SearchModule
│   ├── GitHub API          поиск кода (async)
│   ├── Reddit API          поиск дискуссий (async)
│   └── DuckDuckGo          fallback web (async)
│
└── agents/trainer.py       NightlyTrainer
    ├── collect_pairs()     hot/cold буферы из SQLite
    ├── save_dataset()      JSONL для trl DPOTrainer
    └── run_dpo_lm_studio() генерация data/run_dpo.py
```

---

## 3. Жизненный цикл одного шага (run_step)

Время ≈ 5–9 секунд в зависимости от API latency.

```
t=0.00s  embed(task) → task_vec [1408d]          GPU/CPU 47ms

t=0.05s  FAISS.search(task_vec, k=50)            CPU 4ms
         PPR re-ranking (α=0.85, 20 iter)        CPU 13ms
         → past[0..4]: похожие прошлые решения

t=0.07s  ASYNC: search.search_all(task[:200])    → фон, не блокирует

t=0.07s  chaos.update(I) → λ_L, K_ACT            PID каждые 50 шагов
         hypercycle.update(mean_e) → alloc{G,C,S}
         bandit.select(K_ACT) → model_ids, temp

t=0.07s  parallel API calls:
           alloc[G] генераторов + system_G prompt → Y_G[0..G-1]
           (3.5s P50, 6s P95 с учётом Groq/DeepSeek)

t=3.6s   Value-отсев: q_ext(Y_i) для каждого G
         sort by q_ext → top_G

t=3.6s   parallel API calls:
           alloc[C] критиков + (task + top_G черновики) → critique

t=5.5s   wait search results (уже готовы)
         format_context(480 токенов)

t=5.5s   alloc[S] синтезатор + (task + черновики + критика + поиск) → Y_syn

t=7.5s   scorer:
           q_ext(Y_syn, task_type)         без LLM, мгновенно
           q_env(Y_syn, task_type)         _q_code_safe() при task_type=code
           E_score = q_syn - max_i q_g_i
           D_sem via embeddings

         if H(blackboard) > 0.08:
           _run_depth_loop(...)             максимум 2 итерации

t=7.5s   I_before = memory.I
         memory.add(e_base=0, ...)         FAISS + graph + SQLite
         delta_I = memory.I - I_before    ← ЗДЕСЬ, после add()
         e_total = emergence.apply_delta_i(e_base, delta_I)
         memory.update_trace_e_total(...)

t=7.6s   novelty = external_energy(task_vec, past)
         r_ext = _compute_r_ext(novelty, q_env)
         energy.step(..., R_ext=r_ext) → death/split events
         bandit.update(rewards)
         hypercycle.update(mean_e_by_caste)

         if e_total >= 0.80 and task_type == 'code':
           git commit swarm/evolution/{ts}.py

t=7.6s   return {answer, e_total, delta_I, models, step_time, ...}
```

---

## 4. Математика

### 4.1 Эмбеддинг (1408d)

```
v₁ = BGE-m3(T) ∈ ℝ¹⁰²⁴,  ‖v₁‖ = 1
v₂ = E5-small(T) ∈ ℝ³⁸⁴,   ‖v₂‖ = 1
v_raw = [v₁ ; 0.68·v₂] ∈ ℝ¹⁴⁰⁸
v_hyb = v_raw / ‖v_raw‖
```

FAISS: IndexIVFFlat (nlist=4096, nprobe=32).  
Апгрейд до IVFFlat при 10k+ векторов. При старте — IndexFlatL2.

### 4.2 Thompson Sampling

```
Состояние агента i: (μᵢ, σᵢ, rewards_last20, context_vec_16d)

Сэмплинг:
  Q_i(x) = μᵢ + cᵢᵀ·x_task          контекстный prior
  θᵢ ~ N(Q_i(x), σᵢ²)
  selected = argtop-K_ACT(θᵢ), max 2 от провайдера

Обновление:
  s² = Var(rewards_last20)
  τ̂  = 1/(s² + 0.01)                 adaptive precision
  σᵢ²_new = max((σᵢ⁻² + τ̂)⁻¹, 0.04)  σ_min = 0.2
  μᵢ_new  = μᵢ + 0.047·(r - μᵢ)      EMA α = 1/21
  cᵢ_new  = cᵢ + 0.001·(r - μᵢ)·x    медленный контекст

Exp3 drift detector:
  η_exp3 = √(ln(N) / (N·t))
  ŵᵢ ← ŵᵢ · exp(η_exp3 · r)
  drift = |μᵢ − ŵᵢ_norm| > 0.15  →  σᵢ = σ_init  (сброс)
```

### 4.3 Ляпунов + PID

```
δE_k = E_total(k) - E_total(k-1)
λ_L  = mean(log(|δE_k/δE_{k-1}| + 1e-6))  окно 20 шагов
Цель: λ_L = 0

e_t      = -λ_L
integral = 0.97·integral + e_t              anti-windup
u_t      = 0.5·e + 0.1·integral + 0.05·Δe
K_ACT    = clip(round(6 + u_t), 3, 9)
β_KL     = clip(0.05 + 0.1·max(0, λ_L), 0.05, 0.30)
```

Обновление каждые 50 шагов (не перегружать PID шумом).

### 4.4 Гиперцикл Эйгена

```
k_i ← 0.918·k_i + 0.082·Ē_i            каталитическая константа
Φ   = ∑_j α·k_j·x_j·x_{j-1}            каталитические потоки
Φ_s = max(Φ, 0.05)                       защита от нуля

raw_i  = x_i·(1 + α·k_i·x_{i-1}) / (1 + α·Φ_s)
norm_i = raw_i / ∑raw_j

x_i(t+1) = 0.97·norm_i + 0.01           мутация: noise floor
∑x_i = 0.97·1.0 + 3·0.01 = 1.0 ✓
```

Параметр α = 0.0207 — скорость каталитической реакции.  
Порядок цикла: G → C → S → G (x_{j-1} читается циклически).

### 4.5 Энергетика агентов

```
r_eff = r / (1 + 0.31·ln(1 + n_same))          fitness sharing
      · (1 + 0.047) if cos(Y_i, Y_j) > 0.82    diversity bonus

e_{t+1} = 0.992·e_t
         + 1.53·r_eff
         + 0.50·R_ext
         − tokens_out/1000
         + 0.11·(x_prev − 1/3)·3                нормированный hc_bonus
         − 0.020·e_t²                            квадратичное ограничение

e_t ≤ 0.0 → kill(agent)
e_t ≥ 8.1 → split(agent)  →  дочерний агент с мутацией σ·1.1
```

### 4.6 Shannon entropy блока

```
s_k  = cos(emb_i, emb_j)  для пар (i,j)
p_k  = exp(4·s_k) / ∑exp(4·s_j)       softmax α=4.0
H    = −∑p_k·log(p_k + 1e-9)
H_n  = H / log(n_pairs)               нормировка ∈ [0,1]
```

### 4.7 R_ext — внешняя энергия

```
novelty  = 1 − max_cos(task_vec, past_vecs[:5])
q_env    = _q_code_safe(Y_syn)  ∈ {0.0, 0.1, 0.2, 0.4, 0.7, 1.0}

R_ext    = 0.40·novelty + 0.125 + 0.35·q_env
Диапазон: [0.125, 0.875]
```

### 4.8 E_total

```
E_base = 0.381·tanh(E_score/0.089)     синтез > лучшего одиночки
       + 0.277·tanh(D_sem)·Θ(Q−0.60)   разнообразие при качестве
       − 0.182·(K/9)·(cost/budget)      экономия
       − 0.108·tanh(1.92·|λ_L|)         стабильность

delta_I = I(t+1) − I(t)                ПОСЛЕ memory.add()
E_total = E_base + 0.0015·delta_I      0.0015 = 1 − λ_decay

E_score = q_syn − max_i q_i            > 0 только если синтез лучше лучшего
D_sem   = (2/K(K-1))·∑(1 − cos(Ei, Ej))
```

### 4.9 Leaky Integrator

```
I(t+1) = 0.9985·I(t) + 0.0117·max(0, E_total)
τ      = 1/(1-0.9985) = 666 шагов ≈ 1.1 ч
I*     = 0.0117·E[E+] / 0.0015        устойчивый аттрактор
```

### 4.10 Physarum граф

```
D_ij(t+1) = clip(
    (1-γ)·D_ij + η·F_ij·(1+tanh(E))/2
    − competition_decay·∑_k D_ik,
    0.01, 1.0
)
γ = 0.047, η = 0.113
Levy: P(random edge) = 0.0055 / шаг

PPR(α=0.85, 20 iter) для re-ranking FAISS результатов
```

### 4.11 DPO Loss

```
hot  : E_total > Q̄ + 0.5σ  (верхние ~30%)
cold : E_total < Q̄ − 0.5σ или Q_env = 0

L = L_DPO(chosen=hot, rejected=cold, π_θ, π_ref)
  + β_KL · KL(π_θ ‖ π_ref)

β_KL ∈ [0.05, 0.30]  — адаптируется от chaos.beta_kl
π_ref обновляется ПОСЛЕ каждого DPO цикла
```

---

## 5. Пул моделей

### Провайдеры и роли

| Провайдер | Модели | RPM | Контекст | Роли | Надёжность |
|-----------|--------|-----|---------|------|-----------|
| Groq | llama-3.3-70b | 30 | 8192 | G,C,S | 95% |
| Groq | llama-3.1-8b-instant | 30 | 8192 | G | 98% |
| Groq | mixtral-8x7b-32768 | 30 | 32768 | G,C | 95% |
| Groq | gemma2-9b-it | 30 | 8192 | G | 96% |
| Groq | qwen-qwq-32b | 30 | 16384 | C,S | 85% |
| Groq | llama-3.1-70b | 30 | 8192 | G,C,S | 93% |
| DeepSeek | deepseek-chat (v3) | 60 | 32768 | G,C,S | 88% |
| DeepSeek | deepseek-coder | 60 | 16384 | G,C | 87% |
| Cerebras | llama3.3-70b | 30 | 8192 | G,S | 90% |
| Cerebras | qwen-3-32b | 30 | 16384 | G,C | 90% |
| SambaNova | Llama-3.3-70B | 20 | 8192 | G,S | 78% |
| SambaNova | Qwen2.5-72B | 20 | 8192 | G,C | 75% |
| OpenRouter | llama-3.3-70b:free | 20 | 8192 | G,S | 70% |
| OpenRouter | qwq-32b:free | 20 | 16384 | C,S | 65% |
| OpenRouter | mistral-nemo:free | 20 | 8192 | G | 72% |
| OpenRouter | phi-4:free | 15 | 16384 | G,C | 68% |
| OpenRouter | gemma-3-27b-it:free | 20 | 8192 | G | 73% |
| OpenRouter | deepseek-r1:free | 10 | 32768 | C,S | 55% |
| OpenRouter | qwen-2.5-72b:free | 20 | 32768 | G,C,S | 65% |
| Google | gemini-1.5-flash | 15 | 1M | G,C,S | 82% |
| Google | gemini-2.0-flash | 15 | 1M | G,C,S | 80% |
| Cloudflare | llama-3.1-8b | 50 | **512** | G | 80% |
| Cloudflare | mistral-7b-lora | 50 | **512** | G | 78% |
| LM Studio | qwen3-0.8b (local) | ∞ | 8192 | C | 99% |

> ⚠ Cloudflare: контекст строго 512 токенов. Не назначать роли C и S.

### Логика выбора каст

- **G (Генераторы)**: максимальное разнообразие, высокая температура (0.7–0.8). Цель — покрыть пространство решений.
- **C (Критики)**: умеренная температура (0.6–0.7). Задача — найти ошибки и противоречия в черновиках G.
- **S (Синтезаторы)**: низкая температура (0.5–0.6). Задача — объединить лучшее из черновиков с учётом критики.

Гиперцикл динамически перераспределяет квоты: если критики работают плохо (низкий `mean_e_C`), их `x_C` уменьшается, больше слотов достаётся генераторам.

---

## 6. FAISS стратегия

```
Шаг 0     : IndexFlatL2           (точный поиск, маленькая база)
Шаг 10000 : апгрейд → IndexIVFFlat
             nlist = min(4096, ntotal/10)
             nprobe = 32  (≈ 0.8% от nlist)

Recall@50 ≈ 87% vs точного
Память:    100K × 1408 × float32 = 560MB
           500K → 2.8GB → не влезает в RAM-диск!
           → используем IndexIVFPQ при > 200K (сжатие ~8x)
```

---

## 7. Физические аналогии

| Компонент | Аналог | Источник |
|-----------|--------|---------|
| E_total | Свободная энергия Гиббса | Термодинамика |
| Leaky Integrator | RC-цепь (ёмкость + резистор) | Электроника |
| Physarum граф | Physarum polycephalum | Биология (Tero et al., Science 2010) |
| Гиперцикл | Первичные рибозимы | Эйген, «Гиперцикл», 1979 |
| Край хаоса (λ_L ≈ 0) | Клеточные автоматы, правило 110 | Лэнгтон, 1990 |
| Тепловой двигатель (3 τ) | Цикл Карно | Термодинамика |
| Ренормгруппа (3 масштаба) | Критические явления | Вильсон, 1971 (Nobel 1982) |
| Thompson Sampling | Байесовский бандит | Томпсон, 1933 |
| DPO | RLHF без explicit reward model | Rafailov et al., 2023 |


---

## 9. Σ_v8.9: PPR-энтропия и термодинамика

### 9.1 Два графа — две физические сущности

```
G_M (граф памяти, core/memory.py):
  - глобальный, долгосрочный
  - узлы = trace_id (прошлые решения)
  - рёбра = Physarum (усиление потоком)
  - Levy flight, PPR re-ranking FAISS

G_B (граф структуры блока, core/blackboard.py):
  - локальный, строится при каждом вызове
  - узлы = AST-сущности (код) / предложения (текст)
  - рёбра = parent→child, call→def, Jaccard-overlap
  - НЕ содержит embeddings как основу
```

Смешивание G_B и G_M запрещено.

### 9.2 Граф G_B: код

```python
Узлы: FunctionDef, AsyncFunctionDef, ClassDef,
       Assign, AnnAssign, Import, ImportFrom,
       Return, Call, If, For, While

Рёбра:
  parent → child:  weight=1.0 (forward), weight=0.3 (backward)
  call → def:      weight=2.0 (forward), weight=0.5 (backward)
```

### 9.3 Граф G_B: текст

```
Узлы: предложения (split по '. '/'! '/'? ')
Рёбра:
  sequential i→i+1: weight=1.0, i+1→i: weight=0.4
  Jaccard(kw_i, kw_j) > 0.08:
    i→j: weight=Jaccard, j→i: weight=Jaccard·0.4
```

### 9.4 Матрица переходов

```
T_ij = A_ij / Σ_k A_ik   (row-stochastic)
Dangling node (Σ_k A_ik = 0): T_ii = 1  (self-loop)
Гарантии: T_ij ≥ 0, Σ_j T_ij = 1
```

### 9.5 Personalization vector

```
task_kw = множество ключевых слов задачи
s_i = Jaccard(keywords(node_i), task_kw)
v_i = softmax(s_i / τ_v),  τ_v = 1.0

Fallback: v_i = 1/|V_B|  если Σs_i < 1e-10
Свойства: v_i ≥ 0, Σv_i = 1
Определён ТОЛЬКО из текущего блока + задачи
```

### 9.6 PPR (power iteration)

```
p⁰ = v / ‖v‖₁
p^{k+1} = (1-α)·v + α·T^T·p^k,   α = 0.85
Сходимость: ‖p^{k+1} - p^k‖₂ < 1e-8, max 50 итераций
Гарантии: p_i ≥ 0, Σp_i = 1
```

### 9.7 Shannon entropy блока

```
H(r) = -Σ_i p_i · log(p_i)

Свойства:
  H = 0     при n = 1  (один узел)
  H = log(n) при равномерном p  (максимум)
  H ∈ [0, log(n)]  всегда
```

### 9.8 Термодинамика Пригожина

```
Каузальный порядок (строго):
  1. H_old = bb.get_entropy(block_id)   ← ПЕРЕД обновлением
  2. bb.upsert(block_id, answer)        ← обновление → H_new
  3. ΔH = H_old - H_new                ← ПОСЛЕ обновления
  4. π_r = max(0,ΔH)/c_r · Q_env
  5. d_eS = -ρ_r · ΔH · Q_env         ← ≤ 0
  6. d_iS = L_r · (π_r - π̄)²          ← ≥ 0
  7. Φ = |d_eS| - d_iS
  8. ρ̇_r = ρ_r·(π_r-π̄) + D·(ρ̄-ρ_r)  ← репликатор

S = -Σρ·log(ρ) + Σρ·H(r)            [полная энтропия]
Условие жизни: Φ > 0  (|d_eS| > d_iS)
```

### 9.9 Вклад в энергию агентов

```
r_ext_combined = r_ext + η_thermo · max(0, ΔH) · Q_env

η_thermo = 0.30  (config: energy.eta_thermo)

Семантика: агенты получают дополнительную энергию
пропорционально снижению структурной неопределённости
в блоке И качеству исполнения кода.
```

---

## 8. Ограничения и известные компромиссы

**VRAM 6GB — один GPU одновременно**  
LM Studio занимает ~1.25GB на inference. Ночной DPO пик — 2.9GB. Нельзя запускать DPO пока идёт inference.

**Embedder холодный старт**  
BGE-m3 и E5-small загружаются лениво. Первые 2–3 запроса медленнее обычного.

**FAISS только для CPU**  
faiss-cpu, не faiss-gpu. На GTX 1660S GPU-поиск не даст значимого ускорения при < 500K векторов.

**OpenRouter :free модели непредсказуемы**  
Надёжность 55–72%. deepseek-r1:free часто недоступна. Thompson Sampling со временем снижает их долю в пуле через низкие α.

**Cloudflare Workers AI — контекст 512 токенов**  
Не использовать для длинных задач. Хороши только для быстрого первого черновика (G-роль).

**SQLite однопоточный**  
При одновременном DPO и inference возможны SQLITE_BUSY ошибки. Решение: WAL mode (уже включён в memory.py).
