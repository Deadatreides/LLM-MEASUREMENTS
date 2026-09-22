# Σ_v8.8.1 «Мицелий-Живой»

> **Роевой оркестратор 27 LLM с самоорганизацией, гиперциклом и ночным обучением**  
> Версия: **8.9** · Статус: Production-ready · Лицензия: MIT

```
╔══════════════════════════════════════════════════════════╗
║  22 API-модели + 1 локальная + Thompson Sampling         ║
║  Гиперцикл Эйгена · Physarum-граф · Leaky Integrator    ║
║  Ляпунов + PID · Semantic Blackboard · Ночной DPO        ║
╚══════════════════════════════════════════════════════════╝
```

---

## Что это

Σ «Мицелий» — это не просто «обёртка над LLM API». Это **вычислительная экосистема**, вдохновлённая биологией и физикой сложных систем.

Каждый компонент имеет научный аналог:

| Компонент системы | Аналог в природе | Принцип |
|-------------------|-----------------|---------|
| Thompson Sampling | Естественный отбор | Модели конкурируют за вызовы, выживают сильнейшие |
| Гиперцикл Эйгена | Первичные рибозимы | Генераторы→Критики→Синтезаторы усиливают друг друга |
| Physarum-граф | *Physarum polycephalum* | Сильные пути памяти утолщаются, слабые отмирают |
| PID по Ляпунову | Термостат | Система удерживает себя на краю хаоса (λ_L ≈ 0) |
| Leaky Integrator | RC-цепь | Информация накапливается медленно, устойчивый аттрактор |
| DPO ночью | Консолидация памяти во сне | Обучение локальной модели на дневных следах |
| Semantic Blackboard | Рабочая память | Итеративное разрешение противоречий между ответами |

### Почему это работает лучше простой оркестрации

| | Простой оркестратор | Σ «Мицелий» |
|-|---------------------|-------------|
| Выбор модели | Round-robin или случайный | Thompson Sampling: учится на каждом ответе |
| Разнообразие | Фиксированный состав | Гиперцикл: квоты каст адаптируются к задачам |
| Память | Нет или простой буфер | FAISS + Physarum-граф + PPR re-ranking |
| Обучение | Нет | Ночной DPO на собственных трассах |
| Управление хаосом | Нет | PID удерживает λ_L ≈ 0 |
| Глубина мышления | Один проход | Depth loop: итерации при высокой семантической энтропии |

---

## Быстрый старт

### Минимальный запуск (5 минут)

```bash
# 1. Установить зависимости
pip install -r requirements.txt

# 2. Создать API ключи
cp .env.example .env
# Открыть .env, добавить GROQ_API_KEY и DEEPSEEK_API_KEY

# 3. Запустить
python main.py
```

### Проверить что всё работает

```bash
python -c "
from orchestrator import MyceliumOrchestrator
o = MyceliumOrchestrator()
alive = sum(1 for a in o.bandit.agents.values() if a.alive)
print(f'OK: {alive} агентов активно')
"
```

---

## Требования

### Железо

| Компонент | Минимум | Рекомендуется (тесты проводились на) |
|-----------|---------|--------------------------------------|
| GPU | GTX 1060 6GB | **GTX 1660 Super 6GB** |
| CPU | 4c/3GHz | **Ryzen 5 1600 (6c/12t)** |
| RAM | 16GB | **24GB DDR4** |
| SSD | 50GB | **100GB+ SATA** |
| RAM-диск | — | **4GB** (ускоряет FAISS на ~40%) |

### ПО

- Python 3.10+
- LM Studio (для локальной Critic-модели)
- Git (для эволюционных коммитов)
- CUDA 12.1+ (только для DPO обучения)

---

## Установка

Полное руководство: **[docs/SETUP.md](docs/SETUP.md)**

Краткая инструкция:

```bash
# 1. Зависимости
pip install -r requirements.txt

# 2. LM Studio: скачать Qwen3-0.8B Q4_K_M, запустить сервер на :1234

# 3. API ключи
cp .env.example .env
# Заполнить GROQ_API_KEY и DEEPSEEK_API_KEY (обязательно)

# 4. RAM-диск (Windows: ImDisk → R:\, 4GB)

# 5. Пути в конфиге
# config/settings.yaml → hardware.ramdisk_path

# 6. Проверка
python -c "from orchestrator import MyceliumOrchestrator; MyceliumOrchestrator(); print('OK')"
```

---

## Использование

### Интерактивный режим

```bash
python main.py
```

```
════════════════════════════════════════════════════════════
  Σ_v8.8.1 «Мицелий-Живой»
════════════════════════════════════════════════════════════

>>> Write a Python function to find all prime factors of a number
[code] Processing...

──────────────────────────────────────────────────────────
def prime_factors(n: int) -> list[int]:
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors

# Tests
assert prime_factors(12) == [2, 2, 3]
assert prime_factors(100) == [2, 2, 5, 5]
──────────────────────────────────────────────────────────
E=0.341 | λ=-0.019 | K=5 | 6.8s | EDGE | I=0.4231

>>> /status     # текущие метрики
>>> /save       # сохранить состояние
>>> /quit       # выход
```

### Одиночный запрос

```bash
python main.py --task "Explain backpropagation" --type general
python main.py --task "Write quicksort with tests" --type code
```

### Интеграция с OpenCode

```python
from orchestrator import ask

# Вызов роя из OpenCode
answer = ask("Write a binary search function", task_type="code")
print(answer)
```

Или через MCP в `config.json` OpenCode:
```json
{
  "mcpServers": {
    "mycelium": {
      "command": "python",
      "args": ["main.py", "--opencode"],
      "cwd": "/path/to/mycelium"
    }
  }
}
```

Полное руководство: **[docs/USAGE.md](docs/USAGE.md)**

---

## Архитектура

```
Запрос
  ↓
Embedder (BGE-m3 + E5-small → 1408d)
  ↓
FAISS kNN + Physarum PPR → контекст из памяти
  ↓
ChaosController → λ_L → PID → K_ACT ∈ {3..9}
  ↓
Hypercycle → квоты каст G/C/S с мутацией
  ↓
SwarmBandit (Thompson) → выбор K_ACT моделей
  ↓
┌─ Генераторы (async, 3.5s) ─────────────────┐
│  G1: Groq llama-3.3-70b                    │
│  G2: DeepSeek-v3                           │
│  G3: Groq mixtral-8x7b                    │
└────────────────────────────────────────────┘
  ↓  черновики Y_1..Y_K
┌─ Критики ──────────────────────────────────┐
│  C1: Groq qwq-32b                          │
└────────────────────────────────────────────┘
  ↓  критика + поиск GitHub/Reddit
┌─ Синтезатор ───────────────────────────────┐
│  S1: Cerebras llama-3.3-70b               │
└────────────────────────────────────────────┘
  ↓  Y_syn
[Depth Loop если H > 0.08: Shannon entropy → итерации]
  ↓  Y_final
Scorer: Q_ext + q_env (_q_code_safe sandbox)
  ↓
I_before = memory.I
memory.add(e_base) → FAISS + Physarum + SQLite
delta_I = memory.I - I_before          ← вычисляется ЗДЕСЬ
E_total = E_base + 0.0015·delta_I
update_trace_e_total(trace_id)
  ↓
energy.step() · bandit.update() · hypercycle.update()
  ↓
Ответ пользователю
```

Полная математика: **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**

---

## Метрики

### E_total — главная метрика эмерджентности

```
E_total = 0.381·tanh(E_score/0.089)     [синтез > лучшего одиночки]
        + 0.277·tanh(D_sem)·Θ(Q-0.60)  [разнообразие при качестве]
        - 0.182·(K/9)·(cost/budget)     [экономия ресурсов]
        - 0.108·tanh(1.92·|λ_L|)        [стабильность]
        + 0.0015·delta_I                 [рост информации]
```

| E_total | Интерпретация |
|---------|---------------|
| > 0.40 | Сильная эмерджентность — рой значительно лучше |
| 0.15–0.40 | Умеренная — синтез добавил ценность |
| 0.0–0.15 | Слабая — примерный паритет |
| < 0.0 | Регресс — лучший одиночный ответ был лучше |

### Режимы хаоса

| λ_L | Режим | PID реакция |
|-----|-------|-------------|
| (-0.1, +0.1) | **EDGE** | Ничего не делать — оптимум |
| < -0.1 | ORDER | Увеличить K_ACT |
| > +0.1 | CHAOS | Уменьшить K_ACT |

---

## Пул моделей (27 моделей)

Все провайдеры — бесплатные tier без верификации аккаунта.

| Провайдер | Основные модели | RPM | Роли |
|-----------|----------------|-----|------|
| **Groq** | llama-3.3-70b, qwq-32b, mixtral-8x7b | 30 | G,C,S |
| **DeepSeek** | deepseek-v3, deepseek-coder | 60 | G,C,S |
| **Cerebras** | llama-3.3-70b, qwen-3-32b | 30 | G,S |
| **SambaNova** | Llama-3.3-70B, Qwen2.5-72B | 20 | G,S |
| **OpenRouter** | 8 free models | 10–20 | G,C |
| **Google AI** | gemini-1.5/2.0-flash | 15 | G,C,S |
| **Cloudflare** | llama-3.1-8b ⚠512 токенов | 50 | G |
| **LM Studio** | Qwen3-0.8B (local) | ∞ | C |

---

## Ночное DPO обучение

```bash
# Ручной запуск
python main.py --nightly

# Проверить готовность
python -c "
from orchestrator import get_orchestrator
from agents.trainer import NightlyTrainer
o = get_orchestrator()
t = NightlyTrainer(o.cfg, o.memory.store)
pairs = t.collect_pairs()
print(f'Пар: {len(pairs)} (нужно ≥ 50)')
"
```

Расписание: каждую ночь в 02:00 при работающем `python main.py`.  
Время обучения: ~40 мин, VRAM пик: 2.9GB.  
Требования PyTorch: см. **[docs/SETUP.md#шаг-2](docs/SETUP.md)**.

---

## История исправленных багов

| Версия | Баг | Влияние |
|--------|-----|---------|
| **v8.9** | `H = 1-mean(cos)` — не информационная энтропия | Depth loop оптимизировал неверный сигнал |
| **v8.9** | G_B и G_M смешивались | Граф памяти влиял на локальную энтропию блока |
| **v8.9** | d_iS не был гарантированно ≥ 0 | Нарушение второго начала термодинамики |
|--------|-----|---------|
| **v8.8.1** | `H = 1-mean(cos)` не является энтропией | Depth loop оптимизировал неверный сигнал |
| **v8.8.1** | Depth loop без привязки к E_total | Мог «полировать» мусор при падении E_total |
| **v8.8.1** | `temp_min=1e-6` → залипание blackboard | Система переставала исследовать блоки |
| **v8.8.1** | Нет `_split_into_blocks` | Blackboard работал с монолитными ответами |
| **v8.7.1** | `delta_I` всегда 0 (P0) | Пятое слагаемое E_total не работало |
| **v8.7.1** | `history=[]` утечка RAM (P0) | OOM через 48+ часов работы |
| **v8.7.1** | Нет sandbox для кода (P1) | Потенциальное RCE при генерации кода |
| **v8.5** | E_score всегда ≤ 0 | Первое слагаемое E_total не работало |
| **v8.5** | Thompson precision=4 (нужно 16) | Медленное обучение бандита |
| **v8.5** | Exp3 + Thompson конфликт | Нестабильная маршрутизация |

Полная история: **[docs/CHANGELOG.md](docs/CHANGELOG.md)**

---

## Файловая структура

```
mycelium/
├── orchestrator.py            # Главный 13-шаговый цикл роя
├── main.py                    # Точка входа (REPL / --task / --opencode)
├── requirements.txt           # Зависимости Python
│
├── config/
│   ├── settings.yaml          # Все параметры (подробный справочник в docs/)
│   └── models.yaml            # 27 моделей: провайдер, роль, лимиты
│
├── core/
│   ├── bandit.py              # Thompson + LinUCB (температура) + Exp3 (drift)
│   ├── blackboard.py          # Shannon entropy + SemanticBlock + depth selection
│   ├── chaos.py               # Lyapunov estimator + PID anti-windup
│   ├── hypercycle.py          # Eigen hypercycle + AgentEnergy + NoveltySignal
│   ├── memory.py              # FAISS + Physarum + SQLite + LeakyIntegrator
│   └── scorer.py              # Q_ext + _q_code_safe + EmergenceMetric
│
├── agents/
│   └── trainer.py             # Ночной DPO (генерирует data/run_dpo.py)
│
├── tools/
│   └── search.py              # GitHub + Reddit + DuckDuckGo (async)
│
├── docs/
│   ├── ARCHITECTURE.md        # Полная техническая архитектура и математика
│   ├── SETUP.md               # Пошаговая установка
│   ├── USAGE.md               # Руководство по использованию
│   ├── CONFIG_REFERENCE.md    # Справочник по всем параметрам конфига
│   ├── MONITORING.md          # Метрики, логи, аналитика
│   ├── TROUBLESHOOTING.md     # Диагностика и решение проблем
│   └── CHANGELOG.md           # История изменений с причинами
│
├── .opencode/
│   └── rules.md               # Правила для OpenCode Agent
│
├── .env.example               # Шаблон API ключей
├── .gitignore                 # Исключает data/, logs/, .env
│
├── data/                      # Создаётся автоматически при запуске
│   ├── faiss.index            # Векторная база памяти
│   ├── graph.pkl              # Physarum граф
│   ├── traces.db              # SQLite: история трасс
│   ├── bandit_state.json      # Состояние Thompson (μ/σ всех агентов)
│   └── value_adapter/         # LoRA адаптер после первого DPO
│
├── swarm/
│   ├── context/               # Результаты поиска (async)
│   └── evolution/             # Код с E_total ≥ 0.80 + git commits
│
└── logs/
    ├── orchestrator.log       # Детальный лог каждого шага
    └── main.log               # Лог запуска и REPL
```

---

## Документация

| Документ | Содержание |
|----------|-----------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Все модули, математика, физические аналогии, VRAM-бюджет |
| [docs/SETUP.md](docs/SETUP.md) | Установка Python, LM Studio, RAM-диска, API-ключей, диагностика |
| [docs/USAGE.md](docs/USAGE.md) | Все режимы запуска, мониторинг, советы по эффективности |
| [docs/CONFIG_REFERENCE.md](docs/CONFIG_REFERENCE.md) | Каждый параметр settings.yaml и models.yaml с описанием |
| [docs/MONITORING.md](docs/MONITORING.md) | Метрики здоровья, grep-рецепты, SQLite аналитика, визуализация |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | 30+ решений проблем, быстрая диагностика, экстренное восстановление |
| [docs/CHANGELOG.md](docs/CHANGELOG.md) | История версий v6.6→v8.8.1 с причинами каждого изменения |

---

## Диагностика одной командой

```bash
python -c "
import sys, os, numpy as np
sys.path.insert(0, '.')

checks = []

try:
    from core.blackboard import entropy_from_embeddings
    h = entropy_from_embeddings(list(np.eye(3)))
    checks.append(('Shannon entropy', h > 0.9, f'H={h:.4f}'))
except Exception as e:
    checks.append(('Shannon entropy', False, str(e)))

try:
    import urllib.request
    urllib.request.urlopen('http://localhost:1234/v1/models', timeout=1)
    checks.append(('LM Studio', True, 'running'))
except:
    checks.append(('LM Studio', None, 'not running (optional)'))

from dotenv import load_dotenv; load_dotenv()
for k in ['GROQ_API_KEY', 'DEEPSEEK_API_KEY']:
    v = os.getenv(k, '')
    checks.append((k, bool(v), 'set' if v else 'MISSING'))

src = open('orchestrator.py').read()
for marker, label in [('i_before = self.memory.I', 'delta_I fix'),
                       ('_split_into_blocks', 'split_blocks'),
                       ('GLOBAL STOP', 'depth E_total stop')]:
    checks.append((label, marker in src, 'present' if marker in src else 'MISSING'))

print()
for name, ok, detail in checks:
    icon = '✓' if ok else ('⚠' if ok is None else '✗')
    print(f'  {icon} {name}: {detail}')
print()
all_critical = all(ok for _, ok, _ in checks if ok is not None)
print('Ready to run.' if all_critical else 'Fix issues above.')
"
```

---

## Лицензия

MIT — используй, модифицируй, делись.

---

## Версии

| Версия | Дата | Ключевые изменения |
|--------|------|-------------------|
| **v8.9**   | 2026-05 | PPR-энтропия на G_B, термодинамика Пригожина, G_B≠G_M |
| v8.8.1 | 2026-05 | Shannon entropy, depth loop → E_total, _split_into_blocks |
| v8.7.1 | 2026-04 | delta_I fix, sandbox, R_ext с q_env |
| v8.7 | 2026-03 | Semantic Blackboard, depth loop, Physarum competition |
| v8.5 | 2026-02 | E_score fix, Thompson precision, Physarum граф |
| v6.6 | 2026-01 | Первый прототип (множество известных багов) |

---

## Ночное DPO обучение — ОБЯЗАТЕЛЬНЫЕ требования

**До первого запуска** `python main.py --nightly` установить вручную:

```bash
# 1. PyTorch (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Training stack
pip install "trl>=0.8" "peft>=0.10" "transformers>=4.40,<4.43" \
            "accelerate>=0.28" "bitsandbytes>=0.43" "datasets>=2.18"
```

Ограничения версий: `constraints.txt` (torch<2.4, transformers<4.43).
Без этих пакетов DPO **молча пропускается** (warning в `logs/nightly.log`).
