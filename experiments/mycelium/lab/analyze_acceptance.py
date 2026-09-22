"""lab/analyze_acceptance.py — разбор прогона с приёмкой (TASK_ACCEPTANCE §D).

Пять величин, которые §D предписывает мерить:
  1. f1, f2, f3 с интервалами Уилсона — против порога 0.11;
  2. solved в M и BoN-G, разрыв против прежних 23.0 п.п.;
  3. разрыв приёмки: оракул по всему произведённому минус фактически взятое;
  4. метрика VISION §9.4 — доля задач, решённых ТОЛЬКО композицией, и
     зеркальная — потерянных композицией;
  5. сколько раз новая приёмка выбрала иначе, чем s_ok[0], и с каким исходом
     в каждую сторону.

ВАЖНО: все числа на ДВУХАССЕРТНОЙ шкале (test_0 ушёл в приёмку). Историческое
solved 29.0 % на трёхассертной шкале соответствует 32.0 % на этой — пересчитано
офлайн на текстах sh2_200, см. simulations/logmine/ACCEPTANCE_ONBOARDING.md.

Запуск:
    python lab/analyze_acceptance.py --tag acc_200 --baseline sh2_200
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))

# Опорные числа sh2_200 (трёхассертная шкала) и их двухассертные эквиваленты.
REF = {'solved_M_3': 0.290, 'solved_M_2': 0.320,
       'solved_BoN_3': 0.523, 'gap_3': -0.230,
       'f1_old': 0.230, 'f2_old': 0.368, 'f3_old': 0.071,
       'acc_gap_old': 0.180, 'only_comp_old': 0, 'lost_old': 32}
F_STAR = 0.11


def verdict(p, lo, hi, n):
    """Вердикт по порогу. При n=0 вердикта НЕТ.

    Первая версия печатала «ВЗЯТ» на пустой ветке (0/0 = 0.0 < 0.11) — то есть
    выносила заключение об отсутствующих данных. Ровно тот класс ошибки, что
    описан в HANDOFF §3.3: вывод из сломанного инструмента.
    """
    if n == 0:
        return 'ДАННЫХ НЕТ — вердикта нет'
    if hi < F_STAR:
        return 'ВЗЯТ (весь ДИ ниже порога)'
    if p < F_STAR:
        return 'точка ниже порога, ДИ накрывает'
    return 'НЕ взят'


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def solved(q):
    return q is not None and q >= 1.0 - 1e-9


def load(tag):
    p = os.path.join(HERE, 'data', f'results_{tag}.jsonl')
    rows = [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]
    M = [r for r in rows if r['condition'] == 'M' and r.get('status') == 'ok']
    B = [r for r in rows if r['condition'] == 'BoN-G'
         and r.get('status') == 'ok']
    return M, B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='acc_200')
    a = ap.parse_args()
    M, B = load(a.tag)
    print(f"прогон {a.tag}: M={len(M)}, BoN-G={len(B)}")
    print("шкала ДВУХАССЕРТНАЯ (test_0 в приёмке); опорное solved M на ней "
          f"= {REF['solved_M_2']:.1%}\n")

    # ── 2. solved и разрыв ────────────────────────────────────────────────
    sm = sum(1 for r in M if solved(r.get('q_env_final')))
    sb = sum(1 for r in B if solved(r.get('q_env_final')))
    qm = st.mean([r['q_env_final'] for r in M]) if M else 0
    qb = st.mean([r['q_env_final'] for r in B]) if B else 0
    print("── 2. solved и разрыв ──")
    print(f"   M     : {sm}/{len(M)} = {sm/max(len(M),1):.1%}   "
          f"q_env {qm:.3f}   (было {REF['solved_M_2']:.1%} на этой шкале)")
    if B:
        print(f"   BoN-G : {sb}/{len(B)} = {sb/len(B):.1%}   q_env {qb:.3f}")
    else:
        print("   BoN-G : данных нет (ветка ещё не гонялась)")
    gap = (sm/len(M) - sb/len(B)) if (M and B) else None
    if gap is None:
        print("   разрыв: НЕ СЧИТАЕТСЯ — нужна обе ветки")
    else:
        print(f"   разрыв: {100*gap:+.1f} п.п.   (было −23.0 п.п. на трёх "
              f"ассертах)")
    tm = st.mean([r['tokens_completion'] for r in M]) if M else 0
    tb = st.mean([r['tokens_completion'] for r in B]) if B else 0
    print(f"   токенов: M {tm:.0f}, BoN-G {tb:.0f}")

    # ── 1. f1 / f2 / f3 ───────────────────────────────────────────────────
    print("\n── 1. доли ложных приёмов против порога 0.11 ──")

    def f_of(rows, cand_key):
        k = n = 0
        for r in rows:
            qs = r.get(cand_key) or []
            if any(solved(q) for q in qs) or solved(r.get('q_env_final')):
                pool = list(qs) + [r.get('q_env_final')]
                if any(solved(q) for q in pool):
                    n += 1
                    if not solved(r.get('q_env_final')):
                        k += 1
        return k, n

    k1, n1 = f_of(B, 'q_env_candidates')
    p1, lo1, hi1 = wilson(k1, n1)
    print(f"   f1 BoN-G: {k1}/{n1} = {p1:.4f} [{lo1:.4f}, {hi1:.4f}]  "
          f"было {REF['f1_old']:.3f}  {verdict(p1, lo1, hi1, n1)}")

    k2 = n2 = 0
    for r in M:
        pool = list(r.get('q_env_candidates') or []) \
            + list(r.get('q_env_synth_all') or []) + [r.get('q_env_final')]
        if any(solved(q) for q in pool):
            n2 += 1
            if not solved(r.get('q_env_final')):
                k2 += 1
    p2, lo2, hi2 = wilson(k2, n2)
    print(f"   f2 M    : {k2}/{n2} = {p2:.4f} [{lo2:.4f}, {hi2:.4f}]  "
          f"было {REF['f2_old']:.3f}  {verdict(p2, lo2, hi2, n2)}")

    k3 = n3 = 0
    for r in M:
        if not r.get('depth_changed_answer'):
            continue
        pre, fin = r.get('q_env_pre_depth'), r.get('q_env_final')
        if solved(pre):
            n3 += 1
            if not solved(fin):
                k3 += 1
    p3, lo3, hi3 = wilson(k3, n3)
    print(f"   f3 депт : {k3}/{n3} = {p3:.4f} [{lo3:.4f}, {hi3:.4f}]  "
          f"было {REF['f3_old']:.3f}")

    # ── 3. разрыв приёмки ─────────────────────────────────────────────────
    print("\n── 3. разрыв приёмки ──")
    for name, rows, keys in (('M', M, ('q_env_candidates', 'q_env_synth_all')),
                             ('BoN-G', B, ('q_env_candidates',))):
        if not rows:
            print(f"   {name:<6} данных нет")
            continue
        orc = got = 0
        for r in rows:
            pool = [q for k in keys for q in (r.get(k) or [])] \
                + [r.get('q_env_final')]
            orc += 1 if any(solved(q) for q in pool) else 0
            got += 1 if solved(r.get('q_env_final')) else 0
        n = len(rows)
        print(f"   {name:<6} оракул {orc}/{n} = {orc/n:.1%}, взято "
              f"{got}/{n} = {got/n:.1%}, разрыв {100*(orc-got)/n:.1f} п.п."
              f"   (у M было 18.0 п.п.)")

    # ── 4. VISION §9.4 ────────────────────────────────────────────────────
    bidx = {r['task_id']: r for r in B}
    only = lost = 0
    only_ids = []
    for r in M:
        tid = r['task_id']
        fin = solved(r.get('q_env_final'))
        drafts = any(solved(q) for q in (r.get('q_env_candidates') or []))
        b = bidx.get(tid)
        bon = any(solved(q) for q in (b.get('q_env_candidates') or [])) if b else False
        if fin and not drafts and not bon:
            only += 1
            only_ids.append(tid)
        if not fin and drafts:
            lost += 1
    n = max(len(M), 1)
    print("\n── 4. VISION §9.4 ──")
    # Строгая метрика требует ОБЕИХ веток: «ни черновики, ни 8.4 сэмпла
    # лучшей». Без BoN-G условие ослаблено до «ни один свой черновик», и это
    # надо писать, а не выдавать одно за другое.
    strict = len(bidx) >= len(M) * 0.9
    print(f"   условие: {'СТРОГОЕ (черновики + кандидаты BoN-G)' if strict else 'ОСЛАБЛЕННОЕ — только свои черновики, ветка BoN-G ещё не готова'}")
    print(f"   решено ТОЛЬКО композицией: {only}/{n} = {only/n:.2%}  "
          f"(было 0/200 при строгом условии) {only_ids if only_ids else ''}")
    print(f"   потеряно композицией:      {lost}/{n} = {lost/n:.2%}  "
          f"(было 32/200)")
    print(f"   отношение потерь к приобретениям: "
          f"{lost/max(only,1):.0f}:1  (было 32:1)")

    # ── 5. что изменил гейт ───────────────────────────────────────────────
    print("\n── 5. как часто гейт выбрал иначе и с каким исходом ──")
    for name, rows in (('M', M), ('BoN-G', B)):
        g = [r for r in rows if r.get('gate')]
        if not g:
            print(f"   {name}: следа гейта нет")
            continue
        diff_first = [r for r in g if r['gate'].get('changed_vs_first')]
        diff_qext = [r for r in g if r['gate'].get('changed_vs_qext')]
        stages = collections.Counter(r['gate'].get('stage') for r in g)
        print(f"   {name}: шагов со следом {len(g)}, ступени {dict(stages)}")
        print(f"      выбор отличается от s_ok[0]/первого: {len(diff_first)}"
              f" = {len(diff_first)/len(g):.1%}")
        print(f"      выбор отличается от argmax q_ext:    {len(diff_qext)}"
              f" = {len(diff_qext)/len(g):.1%}")
        if name == 'M':
            up = sum(1 for r in diff_first
                     if solved(r.get('q_env_final'))
                     and not solved(r.get('q_env_pre_gate')))
            dn = sum(1 for r in diff_first
                     if not solved(r.get('q_env_final'))
                     and solved(r.get('q_env_pre_gate')))
            print(f"      из них исход улучшился: {up}, ухудшился: {dn}, "
                  f"не изменился: {len(diff_first)-up-dn}")
        st1 = [r['gate'].get('n_stage1', 0) for r in g]
        st2 = [r['gate'].get('n_stage2', 0) for r in g]
        print(f"      выживает после ст.1 медианно {st.median(st1):.0f}, "
              f"после ст.2 — {st.median(st2):.0f}; "
              f"ст.2 пуста на {sum(1 for x in st2 if x == 0)} шагах")

    # ── срывы вызовов ─────────────────────────────────────────────────────
    fails = [f for r in B for f in (r.get('failures') or [])]
    print(f"\n── срывы вызовов ──")
    print(f"   BoN-G: {len(fails)} сорванных вызовов; они исключены из "
          f"кандидатов, а не засчитаны как непройденные")
    if fails:
        print(f"   по моделям: "
              f"{dict(collections.Counter(f['model'] for f in fails))}")

    out = {'tag': a.tag, 'solved_M': sm/max(len(M), 1),
           'solved_BoN': sb/max(len(B), 1), 'gap': gap,
           'f1': p1, 'f2': p2, 'f3': p3,
           'only_composition': only, 'lost': lost}
    path = os.path.join(HERE, 'data', f'acceptance_{a.tag}.json')
    json.dump(out, open(path, 'w', encoding='utf-8'), ensure_ascii=False,
              indent=1)
    print(f"\nсохранено: {path}")


if __name__ == '__main__':
    main()
