"""Ж1. c = P(слабая решает | лучшая провалилась) — по имеющимся логам.

Задание: DEEP_ANALYSIS.md §2. Порог c* = 0.125.
  c < 0.125 → на этом пуле оркестрация не окупится никаким исполнением;
  c > 0.125 → вложенность неверна, проигрыш 23 п.п. — дефект бюджета.

Поправка к заданию. В DEEP_ANALYSIS сказано «у тебя есть исход каждой модели
на каждой из 200 задач в пуле BoN». Это не так: прогон sh2_200 гонял условие
BoN-G, а оно по построению использует ТОЛЬКО coder15 (run_experiment.py:326).
Исходы остальных четырёх моделей лежат в другом месте — в черновиках условия M,
где модель кандидата восстанавливается сопоставлением calls[role=G,ok] с
q_env_candidates. Это и есть источник.

Считаются три оценки, потому что «провал лучшей» определим тремя способами:
  A. попытка-против-попытки внутри шага (обе модели видели одну задачу в один
     момент, одинаковый контекст) — самая чистая пара;
  B. задача-уровень: лучшая считается провалившей задачу, если провалила ВСЕ
     свои попытки по всем прогонам (включая 8.4 генерации BoN-G). Это то,
     против чего реально идёт сравнение, и оценка получается строже;
  C. без условия — базовая решаемость слабых, для проверки вложенности.
"""
from __future__ import annotations

import collections
import json
import os

from common import (BEST, binom_p_less, generators, gguf_only,
                    iter_records, solved, wilson)

C_STAR = 0.125


def main():
    steps = []            # [(task_id, tag, [(model,q,agent)...])]
    best_attempts = collections.defaultdict(list)   # task_id -> [solved?]
    weak_attempts = collections.defaultdict(list)   # task_id -> [(model,solved?)]

    for rec in iter_records():
        if not gguf_only(rec):
            continue
        tid = rec.get('task_id')
        if rec.get('condition') == 'BoN-G' and rec.get('status') == 'ok':
            for q in rec.get('q_env_candidates') or []:
                best_attempts[tid].append(solved(q))
            continue
        gens = generators(rec)
        if not gens:
            continue
        steps.append((tid, rec['_tag'], gens))
        for name, q, _a in gens:
            if name == BEST:
                best_attempts[tid].append(solved(q))
            else:
                weak_attempts[tid].append((name, solved(q)))

    print(f"шагов M с восстановленными кандидатами: {len(steps)}")
    print(f"задач с попытками лучшей: {len(best_attempts)}, "
          f"попыток всего: {sum(len(v) for v in best_attempts.values())}")
    print(f"задач с попытками слабых: {len(weak_attempts)}, "
          f"попыток всего: {sum(len(v) for v in weak_attempts.values())}")

    # ── A. пара внутри шага ────────────────────────────────────────────────
    a_num = a_den = 0
    a_num_task = a_den_task = 0
    per_model = collections.defaultdict(lambda: [0, 0])
    for tid, tag, gens in steps:
        b = [s for n, q, _ in gens if n == BEST for s in (solved(q),)]
        w = [(n, solved(q)) for n, q, _ in gens if n != BEST]
        if not b or not w:
            continue
        if any(b):                      # лучшая на этом шаге решила — не наш случай
            continue
        a_den += len(w)
        a_num += sum(1 for _n, s in w if s)
        a_den_task += 1
        a_num_task += 1 if any(s for _n, s in w) else 0
        for n, s in w:
            per_model[n][1] += 1
            per_model[n][0] += 1 if s else 0

    p, lo, hi = wilson(a_num, a_den)
    print("\n── A. пара внутри шага (лучшая и слабая на одной задаче, "
          "один момент) ──")
    print(f"   шагов, где coder15 генерировал и провалил: {a_den_task}")
    print(f"   попыток слабых в этих шагах: {a_den}, из них решили: {a_num}")
    print(f"   c_A (на попытку) = {p:.4f}   95% ДИ [{lo:.4f}, {hi:.4f}]"
          f"   порог {C_STAR}")
    if a_den:
        pv = binom_p_less(a_num, a_den, C_STAR)
        print(f"   односторонний тест H0: c >= {C_STAR} → "
              f"P(X <= {a_num} | n={a_den}) = {pv:.5f}")
    pt, lot, hit = wilson(a_num_task, a_den_task)
    print(f"   c_A (на шаг: решила хоть одна слабая) = {pt:.4f} "
          f"[{lot:.4f}, {hit:.4f}]")

    print("\n   по моделям (решила | coder15 провалил на том же шаге):")
    for n in sorted(per_model, key=lambda x: -per_model[x][1]):
        k, tot = per_model[n]
        q, l, h = wilson(k, tot)
        print(f"      {n:<9} {k:>3}/{tot:<4} = {q:.3f}  [{l:.3f}, {h:.3f}]")

    # ── B. задача-уровень ──────────────────────────────────────────────────
    tasks_best_fail = {t for t, v in best_attempts.items() if v and not any(v)}
    tasks_best_ok = {t for t, v in best_attempts.items() if any(v)}
    b_den = b_num = 0
    b_den_task = b_num_task = 0
    for t in tasks_best_fail:
        w = weak_attempts.get(t) or []
        if not w:
            continue
        b_den += len(w)
        b_num += sum(1 for _n, s in w if s)
        b_den_task += 1
        b_num_task += 1 if any(s for _n, s in w) else 0
    p_b, lo_b, hi_b = wilson(b_num, b_den)
    print("\n── B. задача-уровень (лучшая провалила ВСЕ свои попытки, "
          "включая 8.4 генерации BoN-G) ──")
    print(f"   задач, где лучшая провалила всё: {len(tasks_best_fail)} "
          f"(решила хотя бы раз: {len(tasks_best_ok)})")
    print(f"   задач из них со слабыми попытками: {b_den_task}, "
          f"попыток слабых: {b_den}")
    print(f"   c_B (на попытку) = {p_b:.4f}  95% ДИ [{lo_b:.4f}, {hi_b:.4f}]")
    pt2, lo2, hi2 = wilson(b_num_task, b_den_task)
    print(f"   c_B (на задачу: решила хоть одна слабая) = {pt2:.4f} "
          f"[{lo2:.4f}, {hi2:.4f}]   ← формулировка DEEP_ANALYSIS дословно")
    if b_den:
        print(f"   односторонний тест H0: c >= {C_STAR} → "
              f"P(X <= {b_num} | n={b_den}) = "
              f"{binom_p_less(b_num, b_den, C_STAR):.3e}")

    # ── C. безусловная решаемость слабых + проверка вложенности ────────────
    all_weak = [(n, s) for v in weak_attempts.values() for n, s in v]
    p_c, lo_c, hi_c = wilson(sum(1 for _n, s in all_weak if s), len(all_weak))
    cond_ok_num = cond_ok_den = 0
    for t in tasks_best_ok:
        for _n, s in weak_attempts.get(t) or []:
            cond_ok_den += 1
            cond_ok_num += 1 if s else 0
    p_ok, _, _ = wilson(cond_ok_num, cond_ok_den)
    print("\n── C. база и вложенность ──")
    print(f"   P(слабая решает) безусловно      = {p_c:.4f} "
          f"[{lo_c:.4f}, {hi_c:.4f}]  n={len(all_weak)}")
    print(f"   P(слабая решает | лучшая решила) = {p_ok:.4f}  n={cond_ok_den}")
    print(f"   P(слабая решает | лучшая провалила) = {p_b:.4f}  n={b_den}")
    if p_ok > 0:
        print(f"   отношение = {p_b / p_ok:.3f}   "
              f"(1.0 = независимость, 0 = полная вложенность)")

    # ── D. что измеренное c даёт в модели дополнительности ─────────────────
    # Модель из sims/deep_complement.py, параметры оттуда же.
    P_BEST, P_WEAK, N, EFF = 0.47, 0.185, 8.4, 1.77 / 8.4

    def ceil_pure(n, pp, eff=EFF):
        return 1 - (1 - pp) ** (n * eff)

    def ceil_mix(n, share, cc, eff=EFF):
        nb, nw = n * share, n * (1 - share)
        fb = (1 - P_BEST) ** (nb * eff) if nb > 0 else 1.0
        fw = (1 - cc) ** nw if nw > 0 else 1.0
        return 1 - fb * fw

    target = ceil_pure(N, P_BEST)
    print("\n── D. подстановка измеренного c в модель дополнительности ──")
    print(f"   потолок чистого BoN-G при n={N}: {target:.3f}")
    print(f"   {'c':>10} {'источник':<26} {'потолок смеси':>14} {'итог':>10}")
    for cc, src in ((p, 'A: пара внутри шага'),
                    (hi, 'A: верх 95% ДИ'),
                    (p_b, 'B: задача-уровень'),
                    (C_STAR, 'порог c* из отчёта'),
                    (P_WEAK, 'если бы независимо')):
        m = ceil_mix(N, 0.402, cc)
        print(f"   {cc:>10.4f} {src:<26} {m:>14.3f} "
              f"{'ВЫИГРЫШ' if m > target else 'проигрыш':>10}")
    best_share, best_val = None, -1.0
    for i in range(10, 101):
        sh = i / 100
        v = ceil_mix(N, sh, p)
        if v > best_val:
            best_val, best_share = v, sh
    print(f"   при c={p:.4f} оптимальная доля лучшей в смеси = {best_share:.2f} "
          f"(потолок {best_val:.3f}); sigma_min=0.2 даёт 0.40")

    out = {
        'c_A_per_attempt': p, 'c_A_n': a_den, 'c_A_k': a_num,
        'c_A_ci': [lo, hi],
        'c_B_per_attempt': p_b, 'c_B_n': b_den, 'c_B_k': b_num,
        'c_B_ci': [lo_b, hi_b],
        'c_B_per_task': pt2, 'c_B_task_n': b_den_task, 'c_B_task_k': b_num_task,
        'p_weak_uncond': p_c, 'p_weak_given_best_ok': p_ok,
        'c_star': C_STAR,
        'tasks_best_fail_all': len(tasks_best_fail),
        'steps_used': len(steps),
    }
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'out_complement.json')
    json.dump(out, open(path, 'w', encoding='utf-8'), ensure_ascii=False,
              indent=1)
    print(f"\nсохранено: {path}")


if __name__ == '__main__':
    main()
