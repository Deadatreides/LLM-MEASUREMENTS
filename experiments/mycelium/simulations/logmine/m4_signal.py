"""Ж5-Ж8. Чистота сигнала, порог энтропии, частота крупных приростов, состав пула.

Задания:
  SUMMARY.md §Порядок работ п.6 — «Борьба со срывами тестов важнее расширения
      шкалы качества» (шум 0.10 стоит половины эффекта, SIM_REPORT_4 §1).
      Здесь: какая доля замеров q_env — не результат модели, а срыв стенда.
  HANDOFF.md §5.5 п.5 / RESULTS.md §5 — «entropy_threshold перекалибровать под
      H_norm (медиана 0.93, порог 0.08 не отсекает ничего)».
  SIM_REPORT_3 §5 / SIM_REPORT_4 §5.4 / TASK_ALPHA_POS §8 — «частоту событий
      крупного прироста надо померить на настоящем корпусе»: от неё зависит
      цена локализации кредита (проба стоит ~7 лишних проходов на событие).
  DEEP_ANALYSIS.md §4 — отсеять из пула модели с разрывом baseline < 0.09.

Значение q_env однозначно кодирует исход песочницы (core/scorer.py:300-331),
поэтому статус каждого замера восстановим из числа, ничего не исполняя.
"""
from __future__ import annotations

import collections
import json
import os
import statistics as st

from common import (BASELINE, GGUF_POOL, ROOT, generators, gguf_only,
                    iter_records, qenv_status, solved, wilson)

TAG = 'sh2_200'


def main():
    recs = [r for r in iter_records() if r['_tag'] == TAG]
    M = [r for r in recs if r['condition'] == 'M' and r.get('status') == 'ok']
    B = [r for r in recs if r['condition'] == 'BoN-G' and r.get('status') == 'ok']

    # ── Ж6. чистота внешнего сигнала ───────────────────────────────────────
    buckets = collections.Counter()
    src = collections.Counter()
    for r in M:
        for q in r.get('q_env_candidates') or []:
            buckets[qenv_status(q)] += 1
            src['черновики M'] += 1
        for q in r.get('q_env_synth_all') or []:
            buckets[qenv_status(q)] += 1
            src['синтезы M'] += 1
        buckets[qenv_status(r.get('q_env_final', 0.0))] += 1
        src['итог M'] += 1
    for r in B:
        for q in r.get('q_env_candidates') or []:
            buckets[qenv_status(q)] += 1
            src['кандидаты BoN-G'] += 1
    total = sum(buckets.values())
    print("── Ж6. из чего состоит внешний сигнал (все замеры q_env прогона) ──")
    print(f"   замеров всего: {total}  ({dict(src)})")
    for k, v in buckets.most_common():
        print(f"   {k:<22} {v:>5}  {v/total:6.2%}")
    harness = buckets['no_tests_ran'] + buckets['no_pytest/exception']
    print(f"\n   СРЫВ СТЕНДА (тесты не собрались / нет pytest): {harness} "
          f"= {harness/total:.2%}")
    print(f"   схлопнуто в 0.4 (от 0/N до 1/2 тестов неразличимы): "
          f"{buckets['ratio<=0.5']} = {buckets['ratio<=0.5']/total:.2%}")
    print("   для сравнения: SIM_REPORT_4 §1 — шум 0.10 съедает половину "
          "эффекта, огрубление до бинарного не стоит ничего")

    # вырожденность пер-агентной награды
    degen = same = 0
    for r in M:
        q = r.get('q_env_candidates') or []
        if len(q) > 1:
            same += 1
            if len(set(q)) == 1:
                degen += 1
    p, lo, hi = wilson(degen, same)
    print(f"\n   шагов с >1 черновиком: {same}, из них внешний сигнал "
          f"одинаков у всех: {degen} = {p:.2%} [{lo:.2%}, {hi:.2%}]")
    honest = sum(1 for r in M if (r.get('q_env_candidates') and
                                  len(set(r['q_env_candidates'])) == 1 and
                                  solved(r['q_env_candidates'][0])))
    print(f"   из них честная ничья (оба решили верно): {honest}; "
          f"остальные {degen-honest} — награда не различает черновики")

    # ── Ж7. порог энтропии ─────────────────────────────────────────────────
    H = [r['thermo']['H_old'] for r in M if r.get('thermo')]
    Hn = [r['thermo']['H_new'] for r in M if r.get('thermo')]
    dH = [r['thermo']['delta_H'] for r in M if r.get('thermo')]
    phi = [r['thermo']['Phi'] for r in M if r.get('thermo')]
    alive = sum(1 for r in M if (r.get('thermo') or {}).get('alive'))
    print("\n── Ж7. entropy_threshold: отсекает ли он что-нибудь ──")
    print(f"   H_old: медиана {st.median(H):.4f}  мин {min(H):.4f}  "
          f"макс {max(H):.4f}")
    thr = 0.08
    below = sum(1 for h in H if h < thr)
    print(f"   порог depth.entropy_threshold = {thr}: ниже него "
          f"{below}/{len(H)} = {below/len(H):.1%} шагов")
    for q in (0.05, 0.10, 0.25, 0.50):
        srt = sorted(H)
        print(f"      квантиль {q:.2f} распределения H = "
              f"{srt[int(q*(len(srt)-1))]:.4f}")
    print(f"   ΔH: нулевых {sum(1 for d in dH if abs(d)<1e-12)}/{len(dH)}, "
          f"медиана {st.median(dH):.5f}")
    print(f"   Φ: ненулевых {sum(1 for v in phi if abs(v)>1e-12)}/{len(phi)}, "
          f"alive=True в {alive}/{len(M)}")

    # депт-луп: что он делал
    dstat = collections.Counter(r['depth']['stop_reason'] for r in M
                                if r.get('depth'))
    dsteps = collections.Counter(r['depth']['steps'] for r in M
                                 if r.get('depth'))
    print(f"   депт-луп: причины остановки {dict(dstat)}")
    print(f"   шагов депт-лупа за раз: {dict(sorted(dsteps.items()))}")

    # ── Ж8. частота событий крупного прироста ──────────────────────────────
    # Прямого аналога «прирост при расширении контекста» в этих логах нет:
    # лестница расширения контекста в Мицелии не запускалась ни разу. Ближайшее
    # измеримое — прирост внешнего качества от каждой ступени оркестрации.
    print("\n── Ж8. частота событий крупного прироста (цена локализации кредита) ──")
    print("   ВНИМАНИЕ: лестница расширения контекста в Мицелии не запускалась,")
    print("   поэтому это прирост от ступеней ОРКЕСТРАЦИИ, а не от контекста.")
    events = collections.Counter()
    n_ev = 0
    for r in M:
        n_ev += 1
        drafts = r.get('q_env_candidates') or []
        syn = r.get('q_env_synth_all') or []
        pre = r.get('q_env_pre_depth')
        fin = r.get('q_env_final', 0.0)
        best_draft = max(drafts) if drafts else 0.0
        if syn:
            g = max(syn) - best_draft
            if g >= 0.30:
                events['синтез дал >= +0.30 к лучшему черновику'] += 1
            if syn[0] - best_draft >= 0.30:
                events['первый синтез (тот, что берётся) >= +0.30'] += 1
        if pre is not None and fin - pre >= 0.30:
            events['депт-луп дал >= +0.30'] += 1
        if not solved(best_draft) and solved(fin):
            events['итог решил там, где черновики не смогли'] += 1
    for k, v in events.most_common():
        pp, l, h = wilson(v, n_ev)
        print(f"   {k:<48} {v:>3}/{n_ev} = {pp:.2%} [{l:.2%}, {h:.2%}]")
    if events:
        rate = events['синтез дал >= +0.30 к лучшему черновику'] / n_ev
        print(f"\n   при пробе ~7 лишних вызовов на событие цена локализации "
              f"кредита = {7*rate:.2f} вызова на задачу "
              f"(+{7*rate/ (sum(r['n_calls'] for r in M)/n_ev):.1%} к текущим "
              f"{sum(r['n_calls'] for r in M)/n_ev:.1f} вызовам)")

    # ── Ж9. отбор среди синтезов: сколько теряется без приёмки ─────────────
    n9 = k9 = 0
    for r in M:
        syn = r.get('q_env_synth_all') or []
        if len(syn) > 1:
            n9 += 1
            if solved(max(syn)) and not solved(syn[0]):
                k9 += 1
    p9, lo9, hi9 = wilson(k9, n9)
    print("\n── Ж9. приёмка среди самих синтезов (берётся s_ok[0], не лучший) ──")
    print(f"   шагов с >1 синтезом (основной + LOO-пересинтезы): {n9}")
    print(f"   лучший синтез проходил, а взятый первый — нет: {k9} = "
          f"{p9:.2%} [{lo9:.2%}, {hi9:.2%}]")

    # общий оракул: лучшее из всего, что система произвела на шаге
    orc = sum(1 for r in M
              if any(solved(q) for q in (r.get('q_env_candidates') or [])
                     + (r.get('q_env_synth_all') or []) + [r.get('q_env_final', 0)]))
    got = sum(1 for r in M if solved(r.get('q_env_final', 0)))
    print(f"   ОРАКУЛ по всему, что M произвела на шаге: {orc}/{len(M)} = "
          f"{orc/len(M):.1%}; фактически взято {got}/{len(M)} = {got/len(M):.1%}")
    print(f"   разрыв приёмки = {(orc-got)/len(M):.1%} п.п. "
          f"(для сравнения: разрыв M против BoN-G = 23.0 п.п.)")

    # ── Ж5. состав пула по разрешимости ────────────────────────────────────
    print("\n── Ж5. какие пары пула отбор в принципе способен различить ──")
    names = sorted(BASELINE, key=lambda n: -BASELINE[n][0])
    print("   разрыв baseline q_env между соседями:")
    for a, b in zip(names, names[1:]):
        g = BASELINE[a][0] - BASELINE[b][0]
        need = 'разрешима' if g >= 0.09 else 'НЕ разрешима (нужно >2000 набл.)'
        print(f"      {a:<9} − {b:<9} = {g:.4f}   {need}")
    keep = [names[0]]
    for n in names[1:]:
        if BASELINE[keep[-1]][0] - BASELINE[n][0] >= 0.09:
            keep.append(n)
    print(f"   пул после отсева порогом 0.09: {keep} "
          f"({len(keep)} из {len(names)})")

    out = {'harness_fail_share': harness / total,
           'collapsed_04_share': buckets['ratio<=0.5'] / total,
           'degenerate_reward_steps': degen, 'steps_with_2_drafts': same,
           'H_median': st.median(H), 'H_min': min(H),
           'entropy_threshold': thr, 'H_below_threshold': below,
           'oracle_all_products': orc / len(M), 'taken': got / len(M),
           'acceptance_gap': (orc - got) / len(M),
           'events': dict(events), 'n_steps': n_ev,
           'pool_after_filter': keep}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'out_signal.json')
    json.dump(out, open(path, 'w', encoding='utf-8'), ensure_ascii=False,
              indent=1)
    print(f"\nсохранено: {path}")


if __name__ == '__main__':
    main()
