"""lab/gate1_qenv.py — ГЕЙТ 1 (TASK-MYCELIUM-RUN.md §2).

Проверяет, что scorer.q_env различает три случая:
  рабочее решение  → 1.0
  сломанное        → ≤ 0.4
  бесконечный цикл → 0.1 (timeout)

Плюс sanity: все 30 эталонных решений корпуса должны давать q_env = 1.0.
Если это не так — тесты/корпус собраны неверно, дальше идти нельзя.
"""
import json
import os
import sys
import time

os.environ.setdefault('MYCELIUM_ALLOW_WINDOWS_CODE_EXEC', '1')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import yaml
from core.scorer import QualityScorer


def load_cfg():
    with open(os.path.join(os.path.dirname(HERE), 'config', 'settings.yaml'),
              encoding='utf-8') as f:
        return yaml.safe_load(f)


def wrap(code: str) -> str:
    return f"```python\n{code}\n```"


def main():
    cfg = load_cfg()
    sc = QualityScorer(cfg)
    corpus = json.load(open(os.path.join(HERE, 'data', 'corpus30.json'),
                            encoding='utf-8'))['tasks']

    print(f"platform={sys.platform} "
          f"MYCELIUM_ALLOW_WINDOWS_CODE_EXEC="
          f"{os.getenv('MYCELIUM_ALLOW_WINDOWS_CODE_EXEC')}")
    print()

    # ── 1. Все 30 эталонов должны проходить свои тесты ────────────────────
    print("── эталонные решения корпуса ──")
    bad = []
    t0 = time.monotonic()
    for t in corpus:
        q = sc.q_env(wrap(t['reference']), 'code', tests=t['tests'])
        if q < 1.0:
            bad.append((t['task_id'], q))
    print(f"30 эталонов проверены за {time.monotonic()-t0:.1f}s; "
          f"не давших q_env=1.0: {len(bad)}")
    for tid, q in bad:
        print(f"   task {tid}: q_env={q}")
    print()

    # ── 2. Три случая §2 на конкретной задаче ─────────────────────────────
    probe = corpus[0]
    fn_name = None
    for line in probe['reference'].splitlines():
        s = line.strip()
        if s.startswith('def '):
            fn_name = s[4:].split('(')[0].strip()
            break
    print(f"── контрольная задача task_id={probe['task_id']} fn={fn_name} ──")

    good = probe['reference']
    broken = (f"def {fn_name}(*args, **kwargs):\n"
              f"    return None\n")
    hang = (f"def {fn_name}(*args, **kwargs):\n"
            f"    while True:\n"
            f"        pass\n")
    syntax = "def broken(:\n  pass\n"

    cases = [
        ('working', good, 1.0, 'eq'),
        ('broken', broken, 0.4, 'le'),
        ('infinite loop', hang, 0.1, 'eq'),
        ('syntax error', syntax, 0.2, 'eq'),
    ]
    results = {}
    ok = True
    for name, code, expect, mode in cases:
        t0 = time.monotonic()
        q = sc.q_env(wrap(code), 'code', tests=probe['tests'])
        dt = time.monotonic() - t0
        passed = (abs(q - expect) < 1e-9) if mode == 'eq' else (q <= expect + 1e-9)
        ok &= passed
        results[name] = q
        print(f"  {name:<15} q_env={q:<5} (ожидалось {mode} {expect}) "
              f"{'OK' if passed else 'FAIL'}  [{dt:.1f}s]")

    print()
    distinct = len({round(v, 3) for v in results.values()}) >= 3
    print(f"различает случаи: {distinct}")
    verdict = ok and distinct and not bad
    print(f"\nГЕЙТ 1: {'PASS' if verdict else 'FAIL'}")
    return 0 if verdict else 1


if __name__ == '__main__':
    sys.exit(main())
