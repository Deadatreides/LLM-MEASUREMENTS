"""
lab/gate_report.py — сводка по гейтам плана для одного или нескольких прогонов.

Собирает всё, по чему принимается решение «идти дальше или стоп», в одну
таблицу, и считает ρ(mu, baseline) по ТОЙ величине, которая реально управляет
отбором:
  fix.per_role_mu = false -> скалярная mu
  fix.per_role_mu = true  -> mu_role['G'] (отбор генераторов идёт по ней)

Всё берётся из записей прогона, а не из data/bandit_state.json: тот пишется
на шаге 0 (step % 50 == 0) и на 30-шаговом прогоне содержит 1-2 вызова на
агента. Это уже трижды давало правдоподобный, но неверный вывод.

    python lab/gate_report.py --tags base_30,sh1_30,role_30,role2_30
"""

import argparse
import json
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

BASELINE = os.path.join(ROOT, 'lab', 'data', 'baseline_pool.json')

# агент -> модель пула (две реплики coder15 указывают на одну модель)
AGENT_MODEL = {
    'coder15_a': 'qwen2.5-coder-1.5b-instruct-q4_0',
    'coder15_b': 'qwen2.5-coder-1.5b-instruct-q4_0',
    'qwen3_17':  'qwen3-1.7b-q4_0',
    'smol17':    'smollm2-1.7b-instruct-q4_k_m',
    'smol360':   'smollm2-360m-instruct-q5_k_m',
    'qwen05':    'qwen2.5-0.5b-instruct-q5_0',
}


def _rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0] * len(xs)
    for pos, i in enumerate(order):
        r[i] = pos
    return r


def spearman(mu: dict, base: dict):
    pairs = [(mu[a], base['models'][AGENT_MODEL[a]]['q_env'])
             for a in mu if AGENT_MODEL.get(a) in base['models']]
    if len(pairs) < 3:
        return float('nan'), 0
    rm, rb = _rank([p[0] for p in pairs]), _rank([p[1] for p in pairs])
    n = len(pairs)
    d2 = sum((x - y) ** 2 for x, y in zip(rm, rb))
    return 1 - 6 * d2 / (n * (n * n - 1)), n


def load(tag):
    path = os.path.join(ROOT, 'lab', 'data', f'results_{tag}.jsonl')
    if not os.path.exists(path):
        return None, None
    rows = [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]
    M = sorted([r for r in rows if r['condition'] == 'M'], key=lambda r: r['task_id'])
    B = sorted([r for r in rows if r['condition'] == 'BoN-G'], key=lambda r: r['task_id'])
    return M, B


def analyse(tag, base):
    M, B = load(tag)
    if not M:
        return None
    last = M[-1]

    # величина, по которой реально идёт отбор генераторов
    mu_role = last.get('bandit_mu_role') or {}
    muG = {a: v['G'] for a, v in mu_role.items()
           if isinstance(v, dict) and v.get('G') is not None}
    mu_scalar = last.get('bandit_mu') or {}
    if muG:
        drive, drive_name = muG, "mu_role['G']"
    else:
        drive, drive_name = mu_scalar, 'mu (скаляр)'

    rho, n_rho = spearman(drive, base)
    rho_scalar, _ = spearman(mu_scalar, base) if mu_scalar else (float('nan'), 0)
    top = max(drive, key=drive.get) if drive else '?'
    top_base = max(base['models'], key=lambda m: base['models'][m]['q_env'])

    d = [a['q_env_final'] - b['q_env_final'] for a, b in zip(M, B)] if B else []
    ceil = [r for r in M if max(r['q_env_candidates'] or [0]) >= 1.0]
    lost = [r for r in ceil if r['q_env_final'] < 1.0]
    head = [r for r in M if max(r['q_env_candidates'] or [0]) < 1.0]

    return {
        'tag': tag, 'n': len(M), 'drive_name': drive_name,
        'rho': rho, 'rho_scalar': rho_scalar,
        'spread': (max(drive.values()) - min(drive.values())) if len(drive) > 1 else float('nan'),
        'top': top, 'top_ok': AGENT_MODEL.get(top) == top_base,
        'qM': st.mean([r['q_env_final'] for r in M]),
        'qB': st.mean([r['q_env_final'] for r in B]) if B else float('nan'),
        'sM': st.mean([r['solved'] for r in M]),
        'sB': st.mean([r['solved'] for r in B]) if B else float('nan'),
        'gap': st.mean(d) if d else float('nan'),
        'sem': (st.pstdev(d) / len(d) ** 0.5) if len(d) > 1 else float('nan'),
        'ceil': len(ceil), 'lost': len(lost),
        'rec': (len(lost) / len(ceil)) if ceil else float('nan'),
        'head': len(head),
        'pos': sum(1 for r in M if r['E_score_ext'] > 0),
        'et': st.mean([r['E_total'] for r in M if r.get('E_total') is not None]),
        'kact': len({r['k_act'] for r in M}),
        'dx': max((max(abs(v - 1/3) for v in (r.get('hypercycle_x_after') or {1: 1/3}).values())
                   for r in M), default=float('nan')),
        'sec': st.mean([r['seconds'] for r in M]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', default='base_30,sh1_30,role_30,role2_30')
    args = ap.parse_args()
    base = json.load(open(BASELINE, encoding='utf-8'))

    res = [analyse(t.strip(), base) for t in args.tags.split(',') if t.strip()]
    res = [r for r in res if r]
    if not res:
        print('нет данных'); return 2

    tags = [r['tag'] for r in res]
    w = 12
    print(f"\nэталонный порядок baseline: " +
          " > ".join(f"{m.split('-')[0]}({base['models'][m]['q_env']:.3f})"
                     for m in sorted(base['models'],
                                     key=lambda m: -base['models'][m]['q_env'])))
    print()
    hdr = f"{'':<24}" + ''.join(t.rjust(w) for t in tags)
    print(hdr); print('-' * len(hdr))

    def row(label, key, fmt):
        print(f"{label:<24}" + ''.join(
            (fmt.format(r[key]) if isinstance(r[key], (int, float))
             and r[key] == r[key] else '—').rjust(w) for r in res))

    print(f"{'величина отбора':<24}" + ''.join(r['drive_name'].rjust(w) for r in res))
    row('rho(отбор, baseline)', 'rho', '{:+.3f}')
    row('rho(скаляр mu)', 'rho_scalar', '{:+.3f}')
    row('spread', 'spread', '{:.4f}')
    print(f"{'лидер':<24}" + ''.join(r['top'].rjust(w) for r in res))
    print(f"{'лидер верен':<24}" + ''.join(
        ('ДА' if r['top_ok'] else 'нет').rjust(w) for r in res))
    print()
    row('q_env M', 'qM', '{:.3f}')
    row('q_env BoN-G', 'qB', '{:.3f}')
    row('solved M', 'sM', '{:.1%}')
    row('solved BoN-G', 'sB', '{:.1%}')
    row('разрыв M-BoN-G', 'gap', '{:+.3f}')
    row('  SEM', 'sem', '{:.3f}')
    print()
    row('потолок', 'ceil', '{:.0f}')
    row('потеряно', 'lost', '{:.0f}')
    row('recovery', 'rec', '{:.2f}')
    row('запас', 'head', '{:.0f}')
    row('E_ext>0', 'pos', '{:.0f}')
    row('E_total', 'et', '{:+.4f}')
    print()
    row('K_ACT значений', 'kact', '{:.0f}')
    row('max|dx| гиперцикл', 'dx', '{:.5f}')
    row('сек/шаг', 'sec', '{:.0f}')

    last = res[-1]
    print("\n" + "=" * len(hdr))
    print(f"ГЕЙТ Ш1 для {last['tag']}:")
    c1 = last['spread'] > 0.1 and last['top_ok']
    c2 = last['rho'] >= 0.7
    c4 = abs(last['gap']) > 2 * last['sem']
    print(f"  1. spread > 0.1 И лидер верен            {'PASS' if c1 else 'FAIL'}"
          f"   (spread={last['spread']:.3f}, лидер={last['top']})")
    print(f"  2. rho >= +0.7                            {'PASS' if c2 else 'FAIL'}"
          f"   (rho={last['rho']:+.3f})")
    print(f"  3. recovery                               {last['rec']:.2f}"
          f"   (было 0.70 при выключенных флагах)")
    print(f"  4. разрыв за 2 SEM                        {'PASS' if c4 else 'FAIL'}"
          f"   ({last['gap']:+.3f} ± {last['sem']:.3f})")
    return 0


if __name__ == '__main__':
    sys.exit(main())
