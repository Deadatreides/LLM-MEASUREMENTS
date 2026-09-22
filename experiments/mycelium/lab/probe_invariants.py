"""
lab/probe_invariants.py — Ш0.3. Валит прогон, если механизм не активировался.

Задача не «посчитать метрики», а не дать константе притворяться работающим
механизмом. Три из восьми проверок ниже поймали бы R1, R5 и R6 в первый день:
K_ACT был ровно 6 во всех 198 шагах, max|x-1/3| = 0.0027 при пороге 0.0833,
mu всех девяти агентов совпадали до четвёртого знака.

Возвращает код 1, если есть хотя бы один FAIL. Это нужно, чтобы вызывать
из скрипта прогона и получать красный результат сразу, а не через сутки при
чтении отчёта.

Запуск:
    python lab/probe_invariants.py --tag full
    python lab/probe_invariants.py --tag full --condition M
    python lab/probe_invariants.py --results lab/data/results_full.jsonl
"""

import argparse
import collections
import json
import os
import re
import statistics as st
import sys

GREEN = '\033[32m'
RED = '\033[31m'
YELLOW = '\033[33m'
OFF = '\033[0m'


class Probe:
    def __init__(self):
        self.rows = []

    def check(self, name, ok, detail, why=''):
        self.rows.append((name, bool(ok), detail, why))

    def report(self):
        fails = [r for r in self.rows if not r[1]]
        width = max(len(r[0]) for r in self.rows) if self.rows else 10
        for name, ok, detail, why in self.rows:
            mark = f"{GREEN}PASS{OFF}" if ok else f"{RED}FAIL{OFF}"
            print(f"  [{mark}] {name:<{width}}  {detail}")
            if not ok and why:
                print(f"         {YELLOW}↳ {why}{OFF}")
        print()
        if fails:
            print(f"{RED}ПРОБА ИНВАРИАНТОВ: {len(fails)} из {len(self.rows)} "
                  f"механизмов не активировались.{OFF}")
            print("Любой FAIL означает: замер по этому механизму — измерение нуля,")
            print("а не измерение эффекта. Абляции по нему бессмысленны.")
        else:
            print(f"{GREEN}ПРОБА ИНВАРИАНТОВ: все механизмы активны.{OFF}")
        return 1 if fails else 0


def load(path, condition=None):
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if condition and r.get('condition') != condition:
                continue
            rows.append(r)
    return rows


def _vals(rows, key):
    out = []
    for r in rows:
        v = r.get(key)
        if isinstance(v, (int, float)) and v is not None:
            out.append(float(v))
    return out


def run_probe(rows, k_act_default=6):
    p = Probe()
    n = len(rows)

    # 1. K_ACT — PID вообще управляет числом агентов?
    ks = _vals(rows, 'k_act')
    uniq = sorted(set(ks))
    p.check('K_ACT', len(uniq) > 1,
            f"значений: {uniq if len(uniq) <= 6 else f'{len(uniq)} разных'}  (n={len(ks)})",
            "PID вызывается раз в chaos.pid_interval шагов; на коротком прогоне "
            "срабатывает один раз и ставит k_act_center. См. R6, Ш2.2.")

    # 2. Квоты каст — гиперцикл двигает alloc?
    devs = []
    for r in rows:
        x = r.get('hypercycle_x_after') or r.get('hypercycle_x_before')
        if isinstance(x, dict) and x:
            devs.append(max(abs(v - 1 / 3) for v in x.values()))
    k_for_thresh = int(st.median(ks)) if ks else k_act_default
    thresh = 1.0 / (2 * max(1, k_for_thresh))
    dev_max = max(devs) if devs else 0.0
    p.check('квоты каст', dev_max >= thresh,
            f"max|x-1/3| = {dev_max:.5f}, порог сдвига round(x*{k_for_thresh}) = {thresh:.4f}",
            "Гиперцикл структурно инертен: мутация 0.03/шаг сильнее "
            "каталитического дифференциала. См. R5, lab/sim_hypercycle.py.")

    allocs = collections.Counter()
    for r in rows:
        a = r.get('alloc_log')
        if isinstance(a, dict):
            allocs[tuple(sorted(a.items()))] += 1
        elif isinstance(a, (list, tuple)) and a:
            allocs[tuple(a)] += 1
        elif isinstance(a, str) and a:
            # харнесс кладёт сюда сырую строку лога:
            # "swarm:   Hypercycle alloc: G=2 C=2 S=2"
            m = re.findall(r'([GCS])\s*=\s*(\d+)', a)
            if m:
                allocs[tuple(sorted(m))] += 1
    if allocs:
        top = ', '.join(
            '/'.join(f'{c}={v}' for c, v in k) for k in list(allocs)[:2])
        # ВАЖНО: «alloc меняется» само по себе ничего не значит. Первая версия
        # этой проверки требовала только len(allocs) > 1 и дала PASS на
        # прогоне sh2_30, где Multinomial при x ~ 1/3 рандомизировал касты:
        # G принимал 1..4, и все решённые задачи пришли из шагов с G=2, а
        # solved упал 26.7% -> 10.0%. То есть проба одобрила инъекцию шума.
        # Правильный вопрос — меняется ли alloc В СТОРОНУ x: корреляция
        # (alloc_c - K/3) с (x_c - 1/3) по всем шагам и кастам.
        num, dx2 = 0.0, 0.0
        pairs = 0
        for r in rows:
            a = r.get('alloc_log')
            x = r.get('hypercycle_x_after') or r.get('hypercycle_x_before')
            if not isinstance(x, dict) or not isinstance(a, str):
                continue
            m = dict((c, int(v)) for c, v in re.findall(r'([GCS])\s*=\s*(\d+)', a))
            if not m:
                continue
            k_tot = sum(m.values())
            for c in m:
                if c not in x:
                    continue
                num += (m[c] - k_tot / 3.0) * (x[c] - 1 / 3.0)
                dx2 += (x[c] - 1 / 3.0) ** 2
                pairs += 1
        aligned = (num / dx2) if dx2 > 1e-12 else 0.0
        p.check('alloc следует за x', len(allocs) > 1 and aligned > 0,
                f"раскладок: {len(allocs)}  наклон alloc по x: {aligned:+.1f} "
                f"({pairs} пар)  ({top})",
                "Если раскладок много, а наклон около нуля или отрицателен — "
                "это не гиперцикл, а случайное перераспределение каст. При "
                "x ~ 1/3 Multinomial даёт именно шум.")

    # 3. Популяционная динамика — кто-нибудь умирает или делится?
    alive = _vals(rows, 'alive_agents')
    if alive:
        p.check('death/split', len(set(alive)) > 1,
                f"alive_agents: {sorted(set(int(a) for a in alive))}",
                "Награда одинакова для всех агентов шага, поэтому энергии идут "
                "по одной траектории и сходятся к точке между death и split. "
                "См. R1, Ш1.3.")

    # 4. Разброс mu — Thompson различает модели?
    #    В results_full.jsonl mu не пишется; читаем состояние бандита, если есть.
    mu_note = 'нет данных в записях'
    # ВАЖНО: bandit_state.json оркестратор сохраняет только каждые 50 шагов
    # (orchestrator: if self.step % 50 == 0). На прогоне в 10-30 шагов файл
    # остаётся начальным, и spread(mu) по нему бессмыслен. Поэтому сначала
    # пробуем снимок mu из самих записей прогона (bandit_mu), и только если
    # его нет — падаем на файл.
    mu_ok = None
    snaps = [r.get('bandit_mu') for r in rows if isinstance(r.get('bandit_mu'), dict)]
    if snaps:
        last = snaps[-1]
        vals = [v for v in last.values() if isinstance(v, (int, float))]
        if len(vals) > 1:
            spread = max(vals) - min(vals)
            mu_ok = spread > 0.05
            mu_note = (f"spread(mu) = {spread:.6f} по {len(vals)} агентам "
                       f"(снимок из записи прогона)")
    st_path = 'data/bandit_state.json'
    if mu_ok is None and os.path.exists(st_path):
        try:
            with open(st_path, encoding='utf-8') as f:
                state = json.load(f)
            mus = [v['mu'] for v in state.values() if isinstance(v, dict) and 'mu' in v]
            called = [v['mu'] for v in state.values()
                      if isinstance(v, dict) and v.get('calls', 0) > 0]
            pool = called or mus
            if len(pool) > 1:
                spread = max(pool) - min(pool)
                mu_ok = spread > 0.05
                mu_note = (f"spread(mu) = {spread:.6f} по {len(pool)} агентам "
                           f"с calls>0")
        except Exception as exc:
            mu_note = f"не прочитано: {exc}"
    if mu_ok is not None:
        p.check('spread(mu)', mu_ok, mu_note,
                "Все агенты получают одну и ту же награду, поэтому mu сходятся "
                "к общему значению и селектор не может выбрать. Это R1 — "
                "главная причина. См. Ш1.3.")

    # 5. cost/budget — штраф за расточительность несёт информацию?
    toks = _vals(rows, 'tokens_completion') or _vals(rows, 'tokens')
    if toks:
        # Бюджет берём ИЗ КОНФИГА, а не хардкодом. Первая версия считала по
        # старой формуле tokens_per_hour/3600 и показывала «на клипе 10/10»
        # уже ПОСЛЕ включения fix.budget_per_step — то есть сама проба
        # измеряла не то, что делает код.
        budget = 800000 / 3600
        try:
            import yaml
            _c = yaml.safe_load(open('config/settings.yaml', encoding='utf-8'))
            _fx = _c.get('fix', {}) or {}
            if _fx.get('budget_per_step'):
                budget = float(_fx.get('budget_tokens_per_step', 18000))
            else:
                budget = _c['budget']['tokens_per_hour'] / 3600
        except Exception:
            pass
        ratios = [min(t / budget, 1.0) for t in toks]
        sat = sum(1 for x in ratios if x >= 0.999)
        var = st.pvariance(ratios) if len(ratios) > 1 else 0.0
        p.check('cost/budget', var > 1e-6,
                f"дисперсия {var:.2e}, на клипе 1.0: {sat}/{len(ratios)} шагов",
                "budget = tokens_per_hour/3600 — это токенов в СЕКУНДУ, "
                "подставлено как бюджет ШАГА. Член вырождается в константу "
                "-0.182*(K/9). См. R3, Ш1.1.")

    # 6. Термодинамика — Phi жива?
    phis, dhs = [], []
    for r in rows:
        t = r.get('thermo')
        if isinstance(t, dict):
            if isinstance(t.get('Phi'), (int, float)):
                phis.append(float(t['Phi']))
            if isinstance(t.get('delta_H'), (int, float)):
                dhs.append(float(t['delta_H']))
    if phis:
        nz = sum(1 for v in phis if abs(v) > 1e-12)
        p.check('Phi', nz > 0, f"ненулевых: {nz}/{len(phis)}",
                "depth loop кладёт final_answer в блок до compute_thermodynamics, "
                "поэтому H_old == H_new и dH == 0. Флаг depth.thermo_causal_fix. "
                "См. R7.")
    if dhs:
        nz = sum(1 for v in dhs if abs(v) > 1e-12)
        p.check('delta_H', nz > 0, f"ненулевых: {nz}/{len(dhs)}")

    # 7. Разнообразие кандидатов — есть ли вообще ансамбль?
    ident, tot, ncand = 0, 0, []
    for r in rows:
        c = r.get('q_env_candidates')
        if isinstance(c, list) and len(c) > 1:
            tot += 1
            ncand.append(len(c))
            if len(set(c)) == 1:
                ident += 1
    if tot:
        frac = ident / tot
        p.check('различимость кандидатов', frac < 0.30,
                f"кандидатов/шаг {st.mean(ncand):.1f}; с одинаковой q_env: "
                f"{ident}/{tot} ({frac:.0%})",
                "Реплики одной модели с одной температурой дают неразличимые "
                "ответы: ниши, fitness sharing и дискриминация Thompson "
                "остаются без предмета. См. Ф1.2, Ш0.1.")

    ds = _vals(rows, 'd_sem')
    if ds:
        p.check('D_sem', st.mean(ds) > 0.15,
                f"среднее {st.mean(ds):.4f}, макс {max(ds):.4f}",
                "Низкий D_sem означает, что ответы почти совпадают — "
                "разнообразие мерить нечем.")

    # 8. lambda_L двигается (в отличие от K_ACT — это разные вопросы)
    ll = _vals(rows, 'lambda_L')
    if ll:
        outside = sum(1 for v in ll if abs(v) > 0.1)
        p.check('lambda_L', len(set(round(v, 6) for v in ll)) > 1,
                f"диапазон [{min(ll):.3f}, {max(ll):.3f}], вне +-0.1: "
                f"{outside}/{len(ll)}")

    print(f"\n  записей: {n}")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='full')
    ap.add_argument('--results', default=None)
    ap.add_argument('--condition', default=None,
                    help='например M; по умолчанию все условия вместе')
    args = ap.parse_args()

    path = args.results or f'lab/data/results_{args.tag}.jsonl'
    if not os.path.exists(path):
        print(f"нет файла {path}")
        return 2

    rows = load(path, args.condition)
    if not rows:
        print(f"в {path} нет записей"
              + (f" для условия {args.condition}" if args.condition else ''))
        return 2

    scope = f"условие {args.condition}" if args.condition else 'все условия'
    print(f"\nПРОБА ИНВАРИАНТОВ — {path}, {scope}\n")
    p = run_probe(rows)
    print()
    return p.report()


if __name__ == '__main__':
    sys.exit(main())
