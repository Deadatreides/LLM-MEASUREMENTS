"""
lab/sim_hypercycle.py — гейт перед правкой core/hypercycle.py (Ш2.3).

Вопрос: способен ли гиперцикл сдвинуть alloc, и какая комбинация правок это
даёт. Офлайн, без LLM, секунды. Это надо было прогнать ДО реализации.

Разрыв каталитических констант берётся ИЗ ПРОГОНА (lab/data/results_full.jsonl,
поле hypercycle_k), а не выдумывается. Плюс проекция на то, что будет после
Ш1.3, когда награда пойдёт от q_env вместо q_ext.

Порог: round(x_i*K_ACT) меняется при |x_i - 1/3| >= 1/(2*K_ACT).
  K_ACT=6 -> 0.0833;  K_ACT=9 -> 0.0556

--- НАЙДЕНО ЭТИМ СКРИПТОМ -------------------------------------------------
Phi (и phi_min) в уравнении гиперцикла НЕ ВЛИЯЮТ НИ НА ЧТО. Доказательство:

    x_new[c] = x[c] * (1 + a*k[c]*x_prev[c]) / (1 + a*Phi)

Phi — один и тот же скаляр для всех каст, то есть знаменатель общий. Дальше
код делает x_new[c] /= sum(x_new), и общий множитель сокращается ТОЧНО.
Клип [x_min, x_max] применяется уже после деления, так что и он не спасает.

Следствия:
  * "Ошибка 3: деление на нулевое Phi" из аудита Sigma_v8.5 и введённая под
    неё защита max(Phi, 0.05) — защита от деления, которое не может ни на что
    повлиять;
  * правка "убрать alpha из Phi" (форма v8.7) даёт РОВНО НОЛЬ. Проверено ниже
    численно: |dx| совпадает до 5 знаков;
  * в уравнениях Эйгена Phi — это и есть член нормировки, удерживающий сумму
    в единице. Явная нормировка x/sum(x) делает ту же работу второй раз,
    поэтому Phi вырождается. Это не баг, а избыточность — но она означает,
    что весь тюнинг Phi в спецификациях декоративен.

Реальных рычагов два: коэффициент мутации и способ квантования alloc.

Запуск:
    python lab/sim_hypercycle.py
    python lab/sim_hypercycle.py --k-act 9 --steps 50000
"""

import argparse
import json
import os
import statistics as st
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CASTES = ['G', 'C', 'S']
RESULTS = 'lab/data/results_full.jsonl'


def observed_k_gap(path=RESULTS, condition='M'):
    """Разрыв max(k)-min(k), фактически наблюдавшийся в прогоне."""
    if not os.path.exists(path):
        return None
    ks = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get('condition') != condition:
                continue
            k = r.get('hypercycle_k')
            if isinstance(k, dict) and all(c in k for c in CASTES):
                ks.append(k)
    if not ks:
        return None
    gaps = [max(k.values()) - min(k.values()) for k in ks]
    means = {c: st.mean([k[c] for k in ks]) for c in CASTES}
    return {'gap_mean': st.mean(gaps), 'gap_max': max(gaps),
            'k_mean': means, 'n': len(ks)}


def spread_ratio(path=RESULTS, condition='M'):
    """Во сколько раз шире динамический диапазон q_env против q_ext.

    k_i — EMA от mean_e_by_caste, поэтому в равновесии k_i ~= mean_e_i.
    Замена сигнала q_ext -> q_env (Ш1.3) масштабирует разрыв k примерно
    в этом отношении.
    """
    if not os.path.exists(path):
        return None
    qe, qx = [], []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get('condition') != condition:
                continue
            if r.get('q_env_final') is not None:
                qe.append(float(r['q_env_final']))
            if r.get('q_ext_final') is not None:
                qx.append(float(r['q_ext_final']))
    if not qe or not qx:
        return None
    s_env = max(qe) - min(qe)
    s_ext = max(qx) - min(qx)
    return {'q_env_spread': s_env, 'q_ext_spread': s_ext,
            'ratio': (s_env / s_ext) if s_ext > 1e-9 else float('inf')}


def run(mean_e, steps, alpha, kappa_decay, kappa_gain,
        x_min, x_max, phi_min, mutation,
        alpha_in_phi, multinomial_alloc, k_act, lambda_L=0.0, seed=0):
    rng = np.random.default_rng(seed)
    x = {c: 1 / 3 for c in CASTES}
    k = {c: 0.5 for c in CASTES}
    alpha_eff = float(np.clip(alpha * (1.0 + 0.5 * abs(lambda_L)), 0.001, 0.5))

    for _ in range(steps):
        for c in CASTES:
            if c in mean_e:
                k[c] = kappa_decay * k[c] + kappa_gain * max(0.0, mean_e[c])

        if alpha_in_phi:
            phi = sum(alpha_eff * k[CASTES[j]] * x[CASTES[j]] * x[CASTES[j - 1]]
                      for j in range(3))
        else:
            phi = sum(k[CASTES[j]] * x[CASTES[j]] * x[CASTES[j - 1]]
                      for j in range(3))
        phi = max(phi, phi_min)

        x_new = {}
        for j, c in enumerate(CASTES):
            x_prev = x[CASTES[j - 1]]
            x_new[c] = x[c] * (1 + alpha_eff * k[c] * x_prev) / (1 + alpha_eff * phi)

        total = sum(x_new.values())
        if total > 0:
            for c in CASTES:
                x_new[c] = float(np.clip(x_new[c] / total, x_min, x_max))
        else:
            x_new = {c: 1 / 3 for c in CASTES}

        for c in CASTES:
            x_new[c] = (1.0 - mutation) * x_new[c] + mutation / 3.0

        total = sum(x_new.values())
        x = {c: v / total for c, v in x_new.items()}

    dev = max(abs(v - 1 / 3) for v in x.values())

    if multinomial_alloc:
        p = np.array([x[c] for c in CASTES])
        draws = rng.multinomial(k_act, p, size=8000)
        top_x = int(np.argmax(p))
        hit = float(np.mean(np.argmax(draws, axis=1) == top_x))
        # Сигнал считается прошедшим, если сильнейшая каста получает больше
        # агентов заметно чаще случайного. 0.333 — случайно, 0.50 — порог.
        moves = hit >= 0.50
        note = f"argmax hit={hit:.3f} (случайно 0.333, порог 0.50)"
    else:
        alloc = {c: max(1, round(x[c] * k_act)) for c in CASTES}
        moves = len(set(alloc.values())) > 1
        note = f"alloc={alloc}"

    return dev, moves, note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--k-act', type=int, default=6)
    ap.add_argument('--cfg', default='config/settings.yaml')
    args = ap.parse_args()

    with open(args.cfg, encoding='utf-8') as f:
        hc = yaml.safe_load(f)['hypercycle']

    thresh = 1.0 / (2 * args.k_act)
    print(f"K_ACT={args.k_act}   порог round(x*K): |dx| >= {thresh:.4f}")
    print(f"steps={args.steps}   alpha={hc['alpha']}   "
          f"kappa={hc['kappa_decay']}/{hc['kappa_gain']}   "
          f"мутация в коде={0.03}")

    obs = observed_k_gap()
    sr = spread_ratio()
    print()
    if obs:
        print(f"ИЗ ПРОГОНА (условие M, n={obs['n']}): разрыв max(k)-min(k) "
              f"= {obs['gap_mean']:.4f} в среднем, {obs['gap_max']:.4f} макс")
        print(f"  k_mean: " + "  ".join(f"{c}={obs['k_mean'][c]:.4f}" for c in CASTES))
    else:
        print("results_full.jsonl не найден — сценарии только модельные")
    if sr:
        print(f"  динамический диапазон: q_ext={sr['q_ext_spread']:.3f}  "
              f"q_env={sr['q_env_spread']:.3f}  отношение {sr['ratio']:.1f}x")
    print()

    # Сценарии: наблюдённый, проекция после Ш1.3, и предельный как контроль.
    gap_now = obs['gap_mean'] if obs else 0.084
    gap_after = min(0.9, gap_now * sr['ratio']) if sr else 0.5
    scenarios = [
        (f"НАБЛЮДЁННЫЙ сейчас  разрыв k={gap_now:.3f}  (сигнал q_ext)",
         {'G': 0.5 + gap_now, 'C': 0.5, 'S': 0.5 + gap_now / 2}),
        (f"ПРОЕКЦИЯ после Ш1.3 разрыв k={gap_after:.3f}  (сигнал q_env)",
         {'G': 0.5 + gap_after, 'C': 0.5, 'S': 0.5 + gap_after / 2}),
        ("ПРЕДЕЛЬНЫЙ контроль разрыв k=0.900",
         {'G': 1.0, 'C': 0.1, 'S': 0.1}),
    ]

    variants = [
        ('как сейчас                    ', True,  0.03,  False),
        ('alpha из Phi убран (v8.7)     ', False, 0.03,  False),
        ('мутация 0.03 -> 0.005         ', True,  0.005, False),
        ('Multinomial alloc             ', True,  0.03,  True),
        ('мутация + Multinomial         ', True,  0.005, True),
        ('ВСЕ ТРИ                       ', False, 0.005, True),
    ]

    verdict = {}
    for sc_name, mean_e in scenarios:
        print(f"--- {sc_name}")
        base_dev = None
        for v_name, a_in_phi, mut, multi in variants:
            dev, moves, note = run(
                mean_e=mean_e, steps=args.steps,
                alpha=hc['alpha'], kappa_decay=hc['kappa_decay'],
                kappa_gain=hc['kappa_gain'], x_min=hc['x_min'],
                x_max=hc['x_max'], phi_min=hc['phi_min'],
                mutation=mut, alpha_in_phi=a_in_phi,
                multinomial_alloc=multi, k_act=args.k_act,
            )
            if base_dev is None:
                base_dev = dev
            same = 'ТОЖДЕСТВЕННО как сейчас' if abs(dev - base_dev) < 1e-9 and v_name.strip() != 'как сейчас' else ''
            print(f"  {v_name} |dx|={dev:.5f}  сдвиг: {'ДА ' if moves else 'нет'} "
                  f"| {note} {same}")
            verdict[(sc_name[:12], v_name.strip())] = moves
        print()

    print("=" * 78)
    obs_ok = verdict.get(('НАБЛЮДЁННЫЙ', 'ВСЕ ТРИ'), False)
    proj_ok = verdict.get(('ПРОЕКЦИЯ пос', 'ВСЕ ТРИ'), False)
    lim_ok = verdict.get(('ПРЕДЕЛЬНЫЙ к', 'ВСЕ ТРИ'), False)

    print("ГЕЙТ Ш2.3")
    print(f"  на наблюдённом разрыве (сигнал q_ext):  {'проходит' if obs_ok else 'НЕ ПРОХОДИТ'}")
    print(f"  на проекции после Ш1.3 (сигнал q_env):  {'проходит' if proj_ok else 'НЕ ПРОХОДИТ'}")
    print(f"  на предельном контроле:                 {'проходит' if lim_ok else 'НЕ ПРОХОДИТ'}")
    print()
    if lim_ok and not obs_ok:
        print("ВЫВОД: механика правки рабочая (предельный контроль проходит), но при")
        print("нынешнем разрыве каталитических констант её не хватает. Разрыв задаётся")
        print("сигналом награды, а не константами гиперцикла. Значит Ш2.3 БЕСПОЛЕЗЕН")
        print("до Ш1.3 и должен оцениваться на фактическом разрыве ПОСЛЕ него, а не")
        print("тюниться в отрыве.")
    elif obs_ok:
        print("ВЫВОД: правка достаточна уже на нынешнем сигнале. Можно трогать код.")
    else:
        print("ВЫВОД: правка не работает даже на предельном контроле — пересмотреть.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
