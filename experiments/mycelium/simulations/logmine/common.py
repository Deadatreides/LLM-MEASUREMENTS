"""Общая загрузка логов прогонов. Вызовов LLM не делает, ничего не исполняет.

Все скрипты в этой папке читают только то, что уже записано:
  lab/data/results_*.jsonl  — по строке на пару (задача, условие)
  logs/runtime.log          — построчный лог оркестратора
  lab/data/baseline_pool.json

Ключевой факт, на котором держится половина замеров: в записи условия M
поле q_env_candidates выровнено по списку [c for c in calls if role=='G' and ok]
(см. lab/run_experiment.py:361 и :408 — оба списка строятся одним и тем же
фильтром в одном и том же порядке). Значит исход КАЖДОЙ модели на КАЖДОЙ
задаче восстановим, хотя отдельного поля с моделью кандидата нет.
"""
from __future__ import annotations

import glob
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))      # .../mycelium

# Пул GGUF (v8.9.2). Прогоны на старом safetensors-пуле (granite/qwen06/qwen17)
# несравнимы и отбрасываются: granite оказался granitemoe-базой, см. HANDOFF 3.2.
GGUF_POOL = {
    'qwen2.5-coder-1.5b-instruct-q4_0': 'coder15',
    'qwen3-1.7b-q4_0': 'qwen3',
    'smollm2-1.7b-instruct-q4_k_m': 'smol17',
    'smollm2-360m-instruct-q5_k_m': 'smol360',
    'qwen2.5-0.5b-instruct-q5_0': 'qwen05',
    # [MODELS_ONBOARDING §2] Кандидаты. Ключ — ровно то, что llm_server отдаёт
    # в поле model: splitext(basename файла .gguf).lower(), llm_server.py:135.
    # Не имя каталога и не имя из карточки модели. Проверено смоуком
    # (lab/smoke_new_models.py → lab/data/smoke_new_models.json).
    # ВНИМАНИЕ: у Gemma порядок частей в имени файла «-it-1B-», а не «-1b-it-»,
    # как предполагалось в задании. С неверным ключом generators() вернул бы
    # пустой список и шаги молча выпали бы из разбора.
    'gemma-3-it-1b-q5_k_s': 'gemma1b',
    'llama-3.2-1b-instruct-q4_0': 'llama1b',
}
BEST = 'coder15'

# Одиночный baseline на corpus30 (lab/data/baseline_pool.json)
BASELINE = {
    'coder15': (0.6844, 0.4667),
    'qwen3': (0.5689, 0.2667),
    'smol360': (0.4956, 0.1667),
    'smol17': (0.4822, 0.1333),
    'qwen05': (0.4822, 0.2667),
}

# ── семантика q_env (core/scorer.py::_q_code_safe) ─────────────────────────
# Значение q_env однозначно кодирует ИСХОД ПЕСОЧНИЦЫ, поэтому статус
# восстановим из числа, не запуская код повторно.
QENV_STATUS = {
    0.0: 'no_pytest/exception',   # pytest отсутствует или исключение
    0.1: 'timeout',               # бесконечный цикл
    0.2: 'syntax/import_error',   # не импортировался
    0.4: 'ratio<=0.5',            # схлопнутая шкала: от 0/N до 1/2 тестов
    0.5: 'no_tests_ran',          # тесты не собрались — СРЫВ ТЕСТОВ
    0.7: 'imported_no_tests',
    1.0: 'all_passed',
}


def qenv_status(q: float) -> str:
    for v, s in QENV_STATUS.items():
        if abs(q - v) < 1e-9:
            return s
    if 0.5 < q < 1.0:
        return 'partial>0.5'
    return f'other({q:.4f})'


def solved(q) -> bool:
    return q is not None and q >= 1.0 - 1e-9


def iter_records(pattern='lab/data/results_*.jsonl'):
    for path in sorted(glob.glob(os.path.join(ROOT, pattern))):
        tag = os.path.basename(path)[len('results_'):-len('.jsonl')]
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rec['_tag'] = tag
                yield rec


def gguf_only(rec) -> bool:
    """Запись сделана на GGUF-пуле?"""
    for c in rec.get('calls') or []:
        if c.get('model'):
            return c['model'] in GGUF_POOL
    # у BoN-G calls нет: отличаем по тегу
    return rec.get('_tag') not in ('full', 'smoke', 'cf', 'fixcheck')


def generators(rec):
    """[(short_model_name, q_env), ...] по черновикам шага условия M."""
    if rec.get('condition') != 'M' or rec.get('status') != 'ok':
        return []
    calls = rec.get('calls') or []
    g = [c for c in calls if c.get('role') == 'G' and c.get('ok')]
    q = rec.get('q_env_candidates') or []
    if len(g) != len(q):
        return []
    out = []
    for c, qq in zip(g, q):
        name = GGUF_POOL.get(c.get('model'))
        if name is None:
            return []
        out.append((name, float(qq), c.get('agent')))
    return out


# ── статистика ─────────────────────────────────────────────────────────────
def wilson(k, n, z=1.96):
    """Доверительный интервал Уилсона для доли. При малых n нормальный врёт."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def binom_p_greater(k, n, p0):
    """P(X >= k) при X~Bin(n,p0) — односторонний точный тест."""
    tot = 0.0
    for i in range(k, n + 1):
        tot += math.comb(n, i) * p0 ** i * (1 - p0) ** (n - i)
    return tot


def binom_p_less(k, n, p0):
    tot = 0.0
    for i in range(0, k + 1):
        tot += math.comb(n, i) * p0 ** i * (1 - p0) ** (n - i)
    return tot
