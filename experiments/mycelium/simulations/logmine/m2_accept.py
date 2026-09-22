"""Ж2. f — доля ложных приёмов правил приёмки. Порог 0.11.

Задание: DECOMPOSITION.md §4 — «Что проверить в имеющихся логах, без новых
вызовов: долю ложных приёмов критика (случаи, где критик одобрил вариант, не
прошедший тесты). Если она выше 0.11 — прошлый вердикт об оркестрации получен
на неработающей приёмке».

Уточнение по коду. Критик в Σ не выносит вердикта: system_c просит текстовую
критику (orchestrator.py:530), и его текст в записи прогона не сохраняется.
Приёмка в системе есть, но она в трёх других местах, и все три измеримы:

  F1  отбор в BoN-G: best = argmax q_ext (run_experiment.py:303) — q_ext это
      статическая эвристика (core/scorer.py::_q_code), а не исполнение;
  F2  финал в M: final_answer = s_ok[0] (orchestrator.py:577) — первый
      успешный синтезатор, БЕЗ какого-либо отбора по качеству;
  F3  депт-луп: принять кандидата, если h_new <= h_old+eps ИЛИ
      q_candidate >= q_current (orchestrator.py:1097) — снова q_ext.

Ложный приём = принятое решение не проходит тесты, при том что в наличии был
вариант, который проходит. Плюс отдельно измерен маржинальный вклад критиков
через сохранённую пер-ролевую mu роли C (это и есть LOO-награда критика).
"""
from __future__ import annotations

import collections
import json
import os

from common import (BEST, GGUF_POOL, ROOT, generators, gguf_only, iter_records,
                    qenv_status, solved, wilson)

F_STAR = 0.11
TAG = 'sh2_200'


def main():
    recs = [r for r in iter_records() if r['_tag'] == TAG]
    M = [r for r in recs if r['condition'] == 'M' and r.get('status') == 'ok']
    B = [r for r in recs if r['condition'] == 'BoN-G' and r.get('status') == 'ok']
    print(f"прогон {TAG}: M={len(M)}, BoN-G={len(B)}\n")

    # ── F1. отбор по q_ext в BoN-G ─────────────────────────────────────────
    n_recoverable = fa = miss_rank = 0
    n_with_pass = 0
    oracle = chosen_ok = 0
    for r in B:
        qe = r.get('q_ext_candidates') or []
        qv = r.get('q_env_candidates') or []
        if len(qe) != len(qv) or not qv:
            continue
        n_recoverable += 1
        best_i = max(range(len(qe)), key=lambda i: qe[i])
        any_pass = any(solved(q) for q in qv)
        oracle += 1 if any_pass else 0
        chosen_ok += 1 if solved(qv[best_i]) else 0
        if any_pass:
            n_with_pass += 1
            if not solved(qv[best_i]):
                fa += 1
        # ранговая ошибка: выбранный хуже лучшего доступного по q_env
        if qv[best_i] < max(qv) - 1e-9:
            miss_rank += 1

    p, lo, hi = wilson(fa, n_with_pass)
    print("── F1. приёмка BoN-G: argmax q_ext (статическая эвристика) ──")
    print(f"   задач с восстановимой парой q_ext/q_env: {n_recoverable}")
    print(f"   есть хотя бы один проходящий кандидат:   {n_with_pass} "
          f"({n_with_pass/max(n_recoverable,1):.1%} — это потолок отбора)")
    print(f"   выбранный не проходит при наличии проходящего: {fa}")
    print(f"   f1 = {p:.4f}   95% ДИ [{lo:.4f}, {hi:.4f}]   порог {F_STAR}")
    print(f"   решено фактически {chosen_ok}/{n_recoverable} = "
          f"{chosen_ok/max(n_recoverable,1):.1%} против потолка "
          f"{oracle/max(n_recoverable,1):.1%} "
          f"→ отбор теряет {(oracle-chosen_ok)/max(n_recoverable,1):.1%} п.п.")
    print(f"   выбран строго не лучший по q_env: {miss_rank}/{n_recoverable} "
          f"= {miss_rank/max(n_recoverable,1):.1%}")

    # сколько бы дал случайный выбор и худший выбор — калибровка «эвристика
    # вообще что-нибудь знает?»
    import statistics as st
    rnd = st.mean([sum(1 for q in (r.get('q_env_candidates') or [])
                       if solved(q)) / len(r['q_env_candidates'])
                   for r in B if r.get('q_env_candidates')])
    print(f"   для сравнения: случайный кандидат решает {rnd:.1%}, "
          f"argmax q_ext решает {chosen_ok/max(n_recoverable,1):.1%}")

    # ── F2. финал M против черновиков ──────────────────────────────────────
    n2 = fa2 = 0
    lost = collections.Counter()
    for r in M:
        qv = r.get('q_env_candidates') or []
        if not qv:
            continue
        if any(solved(q) for q in qv):
            n2 += 1
            if not solved(r.get('q_env_final', 0.0)):
                fa2 += 1
                lost[qenv_status(r.get('q_env_final', 0.0))] += 1
    p2, lo2, hi2 = wilson(fa2, n2)
    print("\n── F2. приёмка M: final = s_ok[0], отбора нет вообще ──")
    print(f"   шагов, где хотя бы один черновик прошёл тесты: {n2}")
    print(f"   из них итог НЕ проходит: {fa2}")
    print(f"   f2 = {p2:.4f}   95% ДИ [{lo2:.4f}, {hi2:.4f}]   порог {F_STAR}")
    print(f"   во что превратился потерянный ответ: {dict(lost)}")

    # обратное: сколько раз синтез спас там, где все черновики провалили
    saved = sum(1 for r in M
                if (r.get('q_env_candidates') and
                    not any(solved(q) for q in r['q_env_candidates']) and
                    solved(r.get('q_env_final', 0.0))))
    n_allfail = sum(1 for r in M if r.get('q_env_candidates')
                    and not any(solved(q) for q in r['q_env_candidates']))
    print(f"   обратное событие: все черновики провалили, итог прошёл — "
          f"{saved}/{n_allfail} = {saved/max(n_allfail,1):.1%}")
    print(f"   баланс приёмки: потеряно {fa2}, спасено {saved} "
          f"→ чистый итог {saved - fa2:+d} задач")

    # ── F3. депт-луп ───────────────────────────────────────────────────────
    n3 = fa3 = up3 = same3 = 0
    changed = 0
    for r in M:
        if not r.get('depth_changed_answer'):
            continue
        changed += 1
        pre = r.get('q_env_pre_depth')
        fin = r.get('q_env_final')
        if pre is None or fin is None:
            continue
        n3 += 1
        if solved(pre) and not solved(fin):
            fa3 += 1
        elif not solved(pre) and solved(fin):
            up3 += 1
        elif abs(pre - fin) < 1e-9:
            same3 += 1
    p3, lo3, hi3 = wilson(fa3, sum(1 for r in M
                                   if r.get('depth_changed_answer')
                                   and solved(r.get('q_env_pre_depth', 0))))
    print("\n── F3. приёмка депт-лупа: h_new<=h_old+eps ИЛИ q_ext не хуже ──")
    print(f"   шагов, где депт-луп изменил ответ: {changed}/{len(M)}")
    print(f"   сломал проходивший ответ: {fa3}   починил сломанный: {up3}   "
          f"не изменил q_env: {same3}")
    print(f"   f3 = {p3:.4f} [{lo3:.4f}, {hi3:.4f}] "
          f"(доля сломанных среди тех, что приходили в депт-луп рабочими)")

    # ── критики: маржинальный вклад по сохранённой mu роли C ───────────────
    last = M[-1]
    mur = last.get('bandit_mu_role') or {}
    callsr = last.get('bandit_calls_role') or {}
    print("\n── критики: LOO-награда r_C = 0.5 + (q_с_критикой − q_без) ──")
    print("   (>0.5 критика помогла, <0.5 помешала, =0.5 не изменила)")
    tot_c = 0
    for a in sorted(mur):
        v = (mur[a] or {}).get('C')
        n = (callsr.get(a) or {}).get('C', 0)
        if v is None:
            continue
        tot_c += n
        print(f"   {a:<12} mu_C={v:.4f}  вызовов C={n}")
    print(f"   вызовов роли C всего: {tot_c}")

    out = {'f1': p, 'f1_n': n_with_pass, 'f1_k': fa,
           'f2': p2, 'f2_n': n2, 'f2_k': fa2, 'f2_saved': saved,
           'f3': p3, 'f3_k': fa3, 'f3_up': up3,
           'bon_oracle': oracle / max(n_recoverable, 1),
           'bon_chosen': chosen_ok / max(n_recoverable, 1),
           'bon_random': rnd, 'f_star': F_STAR}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'out_accept.json')
    json.dump(out, open(path, 'w', encoding='utf-8'), ensure_ascii=False,
              indent=1)
    print(f"\nсохранено: {path}")


if __name__ == '__main__':
    main()
