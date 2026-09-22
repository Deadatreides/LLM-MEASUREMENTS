"""
lab/analyze.py — сводка по результатам прогона (TASK-MYCELIUM-RUN.md §7, §11).

    python lab/analyze.py --tag smoke
"""
import argparse
import json
import math
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
CONDS = ['M', 'M-depth', 'M-lambda', 'M-HC', 'M-fixed', 'M-swap',
         'BoN', 'BoN-G']


def load(tag):
    path = os.path.join(HERE, 'data', f'results_{tag}.jsonl')
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else float('nan')


def sd(xs):
    xs = [x for x in xs if x is not None]
    return st.stdev(xs) if len(xs) > 1 else 0.0


def sem(xs):
    xs = [x for x in xs if x is not None]
    return (st.stdev(xs) / math.sqrt(len(xs))) if len(xs) > 1 else 0.0


def corr(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return float('nan')
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    if sd(xs) < 1e-12 or sd(ys) < 1e-12:
        return float('nan')
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else float('nan')


def fmt(v, n=3):
    if v is None:
        return '—'
    if isinstance(v, float):
        if math.isnan(v):
            return '—'
        return f"{v:.{n}f}"
    return str(v)


def main(tag):
    rows = load(tag)
    ok = [r for r in rows if r.get('status') == 'ok']
    failed = [r for r in rows if r.get('status') != 'ok']
    by = {c: [r for r in ok if r['condition'] == c] for c in CONDS}
    tasks = sorted({r['task_id'] for r in ok})

    print(f"=== прогон '{tag}': {len(ok)} записей ok, {len(failed)} failed, "
          f"{len(tasks)} задач ===\n")
    if failed:
        for r in failed:
            print(f"  FAILED {r['condition']} task {r['task_id']}: "
                  f"{r.get('error')}")
        print()

    m = by['M']
    if m:
        print("ГЛАВНОЕ ЧИСЛО")
        print(f"  E_score (внешняя Q = исполнение тестов MBPP), полный Mycelium,")
        print(f"  среднее по {len(m)} задачам = {fmt(mean([r['E_score_ext'] for r in m]))}"
              f"  ± {fmt(sem([r['E_score_ext'] for r in m]))} (SEM)")
        print(f"  E_score нативный (q_ext Mycelium)               "
              f"= {fmt(mean([r['E_score_native'] for r in m]))}")
        print()

    # ── таблица условий ────────────────────────────────────────────────────
    hdr = (f"{'условие':<10}{'n':>3}{'q_env':>8}{'solved':>8}{'oracle':>8}"
           f"{'E_ext':>9}{'E_nat':>9}{'E_total':>9}{'токенов':>9}{'секунд':>8}")
    print("ТАБЛИЦА УСЛОВИЙ")
    print(hdr)
    print('-' * len(hdr))
    for c in CONDS:
        rs = by[c]
        if not rs:
            continue
        # oracle = взять лучшего кандидата, зная тесты. Верхняя граница отбора.
        orc = [max(r.get('q_env_candidates') or [0.0]) for r in rs]
        print(f"{c:<10}{len(rs):>3}"
              f"{fmt(mean([r['q_env_final'] for r in rs])):>8}"
              f"{fmt(mean([r['solved'] for r in rs])):>8}"
              f"{fmt(mean(orc)):>8}"
              f"{fmt(mean([r['E_score_ext'] for r in rs])):>9}"
              f"{fmt(mean([r.get('E_score_native') for r in rs])):>9}"
              f"{fmt(mean([r.get('E_total') for r in rs])):>9}"
              f"{fmt(mean([r['tokens_completion'] for r in rs]), 0):>9}"
              f"{fmt(mean([r['seconds'] for r in rs]), 0):>8}")
    print("  oracle — q_env лучшего из кандидатов при отборе по тестам "
          "(верхняя граница, недостижимая без оракула)")
    print()

    # ── Ш0.2: переформулированный критерий эмерджентности ──────────────────
    # E_score = Q_syn - max_i Q_i на дискретной шкале из шести уровней почти не
    # может быть положительным, когда потолок уже достигнут кандидатом: там
    # E_score <= 0 ПО ПОСТРОЕНИЮ, и среднее по всем задачам смешивает
    # "не смог улучшить" с "улучшать было нечего".
    #
    # recovery — доля задач, где среди кандидатов был q_env=1.0, а финальный
    # ответ его не получил. Это прямая цена селектора, и она чувствительнее
    # среднего q_env: величину надо гнать в ноль.
    print("Ш0.2 ПОТОЛОК И ЦЕНА СЕЛЕКТОРА")
    hdr2 = (f"{'условие':<10}{'потолок':>9}{'запас':>7}{'E_ext|зап':>11}"
            f"{'>0|зап':>8}{'потеряно':>10}{'recovery':>10}")
    print(hdr2)
    print('-' * len(hdr2))
    for c in CONDS:
        rs = by[c]
        if not rs:
            continue
        ceil, head, lost = [], [], 0
        e_head, pos_head = [], 0
        for r in rs:
            cands = r.get('q_env_candidates') or []
            if not cands:
                continue
            best = max(cands)
            fin = float(r.get('q_env_final') or 0.0)
            if best >= 1.0:
                ceil.append(r)
                if fin < 1.0:
                    lost += 1
            else:
                head.append(r)
                e = r.get('E_score_ext')
                if isinstance(e, (int, float)):
                    e_head.append(float(e))
                    if e > 0:
                        pos_head += 1
        n_ceil, n_head = len(ceil), len(head)
        rec = (lost / n_ceil) if n_ceil else float('nan')
        print(f"{c:<10}{n_ceil:>9}{n_head:>7}"
              f"{fmt(mean(e_head)):>11}{pos_head:>8}"
              f"{lost:>10}{fmt(rec, 2):>10}")
    print("  потолок  — задач, где кандидат уже дал q_env=1.0 (E_score<=0 по построению)")
    print("  запас    — задач, где улучшение в принципе возможно; только они содержательны")
    print("  потеряно — из потолочных: правильный кандидат был, финал его не получил")
    print("  recovery — потеряно/потолок. Цена селектора. Гнать в ноль")
    print()

    # ── парные разрывы по общим задачам ────────────────────────────────────
    print("РАЗРЫВЫ (парно, по задачам, где есть оба условия)")
    base = {r['task_id']: r for r in by['M']}
    for c in ['BoN', 'BoN-G', 'M-swap', 'M-depth', 'M-lambda', 'M-HC',
              'M-fixed']:
        other = {r['task_id']: r for r in by[c]}
        common = sorted(set(base) & set(other))
        if not common:
            continue
        d_q = [base[t]['q_env_final'] - other[t]['q_env_final'] for t in common]
        d_s = [base[t]['solved'] - other[t]['solved'] for t in common]
        tok_m = mean([base[t]['tokens_completion'] for t in common])
        tok_o = mean([other[t]['tokens_completion'] for t in common])
        print(f"  M − {c:<9} n={len(common):<3} "
              f"Δq_env={fmt(mean(d_q)):>7} ± {fmt(sem(d_q))}   "
              f"Δsolved={fmt(mean(d_s)):>7}   "
              f"токены M={fmt(tok_m,0)} vs {c}={fmt(tok_o,0)} "
              f"(отношение {fmt(tok_m/tok_o if tok_o else float('nan'),2)})")
    print()

    # ── §7 диагностика ─────────────────────────────────────────────────────
    print("ДИАГНОСТИКА §7 — активируются ли механизмы (условие M)")
    if m:
        phis = [r['thermo']['Phi'] for r in m]
        dhs = [r['thermo']['delta_H'] for r in m]
        alives = [1 if r['thermo']['alive'] else 0 for r in m]
        lams = [r['lambda_L'] for r in m]
        print(f"  1. Φ: min={fmt(min(phis),5)} max={fmt(max(phis),5)} "
              f"mean={fmt(mean(phis),5)};  Φ>0 в {sum(1 for p in phis if p > 0)}/{len(phis)} шагах;"
              f"  alive=True в {sum(alives)}/{len(alives)}")
        print(f"     ΔH: min={fmt(min(dhs),5)} max={fmt(max(dhs),5)} "
              f"mean={fmt(mean(dhs),5)}; ΔH≠0 в "
              f"{sum(1 for d in dhs if abs(d) > 1e-9)}/{len(dhs)}")
        print(f"  2. λ_L: min={fmt(min(lams),4)} max={fmt(max(lams),4)}; "
              f"вне [-0.1,0.1] в {sum(1 for l in lams if abs(l) > 0.1)}/{len(lams)}; "
              f"режимы: "
              f"{ {r['regime'] for r in m} }")
        print(f"     K_ACT: {sorted({r['k_act'] for r in m})}")
        devs = []
        allocs = set()
        for r in m:
            x = r.get('hypercycle_x_after') or {}
            if x:
                devs.append(max(abs(v - 1/3) for v in x.values()))
            if r.get('alloc_log'):
                allocs.add(r['alloc_log'].split('alloc:')[-1].strip())
        print(f"  3. квоты каст: max|x−1/3| по шагам = "
              f"{fmt(max(devs),5) if devs else '—'}; "
              f"фактические alloc: {sorted(allocs)}")
        print(f"  4. corr(Φ, solved)  = {fmt(corr(phis, [r['solved'] for r in m]))}")
        print(f"     corr(ΔH, solved) = {fmt(corr(dhs, [r['solved'] for r in m]))}")
        print(f"     corr(E_total, solved) = "
              f"{fmt(corr([r['E_total'] for r in m], [r['solved'] for r in m]))}")
        started = sum(1 for r in m if (r.get('depth') or {}).get('started'))
        steps = [(r.get('depth') or {}).get('steps', 0) for r in m]
        changed = sum(1 for r in m if r.get('depth_changed_answer'))
        dq = [r['q_env_final'] - r['q_env_pre_depth'] for r in m
              if r.get('q_env_pre_depth') is not None]
        reasons = {}
        for r in m:
            k = (r.get('depth') or {}).get('stop_reason')
            reasons[k] = reasons.get(k, 0) + 1
        print(f"  5. depth loop: запущен {started}/{len(m)}, "
              f"шагов в среднем {fmt(mean(steps),2)}, ответ изменён {changed}/{len(m)},")
        print(f"     Δq_env(после депта − до) = {fmt(mean(dq))}; причины остановки: {reasons}")
    print()

    # ── термодинамика по всем оркестрируемым условиям ──────────────────────
    print("ТЕРМОДИНАМИКА ШАГА ПО УСЛОВИЯМ")
    print(f"  {'условие':<9}{'ΔH mean':>10}{'ΔH≠0':>7}{'Φ mean':>10}"
          f"{'Φ>0':>7}{'alive':>7}{'corr(Φ,solved)':>16}{'corr(ΔH,solved)':>17}")
    for c in CONDS:
        rs = [r for r in by[c] if r.get('thermo')]
        if not rs:
            continue
        dh = [r['thermo']['delta_H'] for r in rs]
        ph = [r['thermo']['Phi'] for r in rs]
        sv = [r['solved'] for r in rs]
        print(f"  {c:<9}{fmt(mean(dh), 4):>10}"
              f"{sum(1 for v in dh if abs(v) > 1e-9):>4}/{len(rs):<2}"
              f"{fmt(mean(ph), 4):>10}"
              f"{sum(1 for v in ph if v > 1e-9):>4}/{len(rs):<2}"
              f"{sum(1 for r in rs if r['thermo']['alive']):>4}/{len(rs):<2}"
              f"{fmt(corr(ph, sv)):>16}{fmt(corr(dh, sv)):>17}")
    print()

    # ── депт-луп по условиям ───────────────────────────────────────────────
    print("ДЕПТ-ЛУП ПО УСЛОВИЯМ")
    print(f"  {'условие':<9}{'запущен':>9}{'шагов':>8}{'ответ изменён':>15}"
          f"{'Δq_env':>9}")
    for c in CONDS:
        rs = [r for r in by[c] if r.get('depth')]
        if not rs:
            continue
        print(f"  {c:<9}"
              f"{sum(1 for r in rs if r['depth'].get('started')):>6}/{len(rs):<2}"
              f"{fmt(mean([r['depth'].get('steps', 0) for r in rs]), 2):>8}"
              f"{sum(1 for r in rs if r.get('depth_changed_answer')):>12}/{len(rs):<2}"
              f"{fmt(mean([r['q_env_final'] - r['q_env_pre_depth'] for r in rs])):>9}")
    print()

    # ── нативная vs внешняя E_score ────────────────────────────────────────
    print("E_SCORE: НАТИВНАЯ (q_ext) ПРОТИВ ВНЕШНЕЙ (тесты)")
    for c in CONDS:
        rs = [r for r in by[c] if r.get('E_score_native') is not None]
        if not rs:
            continue
        nat = [r['E_score_native'] for r in rs]
        ext = [r['E_score_ext'] for r in rs]

        def sgn(v):
            return 1 if v > 1e-9 else (-1 if v < -1e-9 else 0)

        tbl = {}
        for a, b in zip(nat, ext):
            tbl[(sgn(a), sgn(b))] = tbl.get((sgn(a), sgn(b)), 0) + 1
        disagree = sum(v for (a, b), v in tbl.items() if a * b < 0)
        nat_pos_ext_neg = sum(v for (a, b), v in tbl.items() if a > 0 > b)
        print(f"  {c:<9} n={len(rs):<3} "
              f"нативная={fmt(mean(nat)):>7}  внешняя={fmt(mean(ext)):>7}  "
              f"corr={fmt(corr(nat, ext)):>7}  "
              f"противоположный знак {disagree}/{len(rs)}  "
              f"(нативная «улучшил», тесты «испортил»: {nat_pos_ext_neg})")
    if m:
        print("  таблица знаков для M (строки — нативная, столбцы — внешняя):")
        def sgn(v):
            return 1 if v > 1e-9 else (-1 if v < -1e-9 else 0)
        print(f"      {'ext<0':>7}{'ext=0':>7}{'ext>0':>7}")
        for a, lab in ((1, 'nat>0'), (0, 'nat=0'), (-1, 'nat<0')):
            row = [sum(1 for r in m
                       if sgn(r['E_score_native']) == a
                       and sgn(r['E_score_ext']) == b)
                   for b in (-1, 0, 1)]
            print(f" {lab:<5}" + ''.join(f"{v:>7}" for v in row))
    print()

    # ── распределения (§11) ────────────────────────────────────────────────
    def hist(label, values, buckets):
        counts = {b: 0 for b in buckets}
        for v in values:
            for lo, hi in buckets:
                if lo <= v < hi:
                    counts[(lo, hi)] += 1
                    break
        n = len(values)
        print(f"  {label} (n={n})")
        for (lo, hi), k in counts.items():
            bar = '#' * int(round(40 * k / n)) if n else ''
            print(f"    [{lo:>6.2f}, {hi:>6.2f})  {k:>3}  {bar}")

    print("РАСПРЕДЕЛЕНИЯ")
    e_buckets = [(-1.01, -0.5), (-0.5, -0.1), (-0.1, -1e-9),
                 (-1e-9, 1e-9), (1e-9, 0.1), (0.1, 0.5), (0.5, 1.01)]
    for c in ['M', 'M-fixed', 'BoN-G']:
        if by.get(c):
            hist(f"E_score_ext, {c}", [r['E_score_ext'] for r in by[c]], e_buckets)
    for c in ['M', 'M-fixed']:
        if by.get(c):
            phis = [r['thermo']['Phi'] for r in by[c]]
            hist(f"Φ, {c}", phis,
                 [(-1e9, -1e-9), (-1e-9, 1e-9), (1e-9, 0.05), (0.05, 0.2),
                  (0.2, 1.0), (1.0, 1e9)])
    if m:
        hist("λ_L, M", [r['lambda_L'] for r in m],
             [(-1e9, -1.0), (-1.0, -0.1), (-0.1, 1e-9), (1e-9, 0.1),
              (0.1, 1.0), (1.0, 1e9)])
        hist("x_G − 1/3, M", [r['hypercycle_x_after']['G'] - 1/3 for r in m],
             [(-1e9, -0.083), (-0.083, -0.001), (-0.001, 0.001),
              (0.001, 0.083), (0.083, 1e9)])
    print()

    # ── по задачам ─────────────────────────────────────────────────────────
    print("ПО ЗАДАЧАМ (q_env финального ответа)")
    hdr2 = f"{'task':>6}" + ''.join(f"{c:>10}" for c in CONDS)
    print(hdr2)
    print('-' * len(hdr2))
    for t in tasks:
        line = f"{t:>6}"
        for c in CONDS:
            r = next((x for x in by[c] if x['task_id'] == t), None)
            line += f"{fmt(r['q_env_final'], 2) if r else '—':>10}"
        print(line)
    print()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='smoke')
    main(ap.parse_args().tag)
