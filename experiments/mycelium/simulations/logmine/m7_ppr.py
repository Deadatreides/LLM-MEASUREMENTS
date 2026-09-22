"""Ж12. PPR: сошёлся в 42.5 % блоков — насколько это на самом деле дефект.

В LOG_MINING.md первой редакции я записал это как «вся термодинамика стоит на
шуме». Здесь величина ошибки считается, а не оценивается на глаз, потому что
формулировка «не сошёлся» ничего не говорит о том, НА СКОЛЬКО не сошёлся.

Три независимых способа:
  1. теория: степенная итерация PPR сходится как alpha^k;
  2. невязки, записанные в лог самим предупреждением;
  3. прямой пересчёт: те же тексты, тот же граф, PPR со снятым лимитом.

Вызовов LLM нет. Сгенерированный код не исполняется — только ast.parse внутри
BlockGraph, ровно как в самом прогоне.
"""
from __future__ import annotations

import json
import math
import os
import re
import statistics as st
import sys

import numpy as np

from common import ROOT, iter_records

sys.path.insert(0, ROOT)

from core.block_graph import BlockGraph            # noqa: E402
from core.ppr import run_ppr, shannon_entropy      # noqa: E402

ALPHA = 0.85
TOL = 1e-8
MAX_ITER_PROD = 100
LOG = os.path.join(ROOT, 'logs', 'runtime.log')
TAG = 'sh2_200'


def main():
    # ── 1. теория ──────────────────────────────────────────────────────────
    need = math.log(TOL) / math.log(ALPHA)
    print("── 1. почему не сходится: арифметика, а не патология графа ──")
    print(f"   степенная итерация π_{{k+1}} = α·Pᵀπ_k + (1−α)v сходится как α^k")
    print(f"   α = {ALPHA}, tol = {TOL:.0e} → нужно {need:.1f} итераций")
    print(f"   в core/ppr.py DEFAULT_MAX_ITER = {MAX_ITER_PROD}")
    print(f"   α^{MAX_ITER_PROD} = {ALPHA**MAX_ITER_PROD:.3e} — это и есть "
          f"невязка, до которой лимит успевает дойти")
    print(f"   ВЫВОД: лимит не дотягивает до собственного tol на "
          f"{need - MAX_ITER_PROD:.0f} итераций. Порог и лимит рассогласованы")
    print(f"   между собой; свойство графа тут ни при чём.")

    # ── 2. невязки из лога ─────────────────────────────────────────────────
    P = re.compile(r'PPR did not converge in (\d+) iterations '
                   r'\(residual=([\de.+-]+), tol=([\de.+-]+)\)')
    res = [float(m.group(2)) for line in open(LOG, encoding='utf-8',
                                              errors='replace')
           for m in [P.search(line)] if m]
    res.sort()
    print("\n── 2. насколько именно не сошёлся: невязки из самого лога ──")
    print(f"   предупреждений: {len(res)}")
    print(f"   ‖π_k − π_{{k−1}}‖₁: мин {res[0]:.2e}  медиана "
          f"{st.median(res):.2e}  q95 {res[int(0.95*len(res))]:.2e}  "
          f"макс {res[-1]:.2e}")
    bound = res[-1] * ALPHA / (1 - ALPHA)
    print(f"   отсюда расстояние до истинной точки: "
          f"‖π_k − π*‖₁ ≤ residual·α/(1−α) ≤ {bound:.2e}")
    print(f"   для сравнения: сама π имеет масштаб 1/n ≈ {1/13:.3f} при "
          f"медианных 13 узлах")
    print(f"   то есть π искажена в {1/13/bound:.0e} раз меньше собственного "
          f"масштаба")

    # ── 3. прямой пересчёт на настоящих текстах прогона ────────────────────
    corpus = json.load(open(os.path.join(ROOT, 'lab', 'data',
                                         'corpus200.json'), encoding='utf-8'))
    prompts = {t['task_id']: t['prompt'] for t in corpus['tasks']}

    texts = []
    for r in iter_records():
        if r['_tag'] != TAG or r['condition'] != 'M' or r.get('status') != 'ok':
            continue
        p = prompts.get(r['task_id'], '')
        for t in (r.get('gen_texts') or []):
            texts.append((p, t))
        for t in (r.get('syn_texts') or []):
            texts.append((p, t))
        if r.get('answer'):
            texts.append((p, r['answer']))
    print(f"\n── 3. пересчёт: те же тексты, лимит снят ──")
    print(f"   текстов прогона: {len(texts)} (черновики + синтезы + итоги)")

    dH, dHn, ns, n_nc, n_deg = [], [], [], 0, 0
    it_needed = []
    for prompt, text in texts:
        try:
            g = BlockGraph().build(text, 'code')
        except Exception:
            continue
        if g.n <= 1:
            n_deg += 1
            continue
        try:
            Pm = g.transition_matrix()
            v = g.personalization_vector(prompt, 'code')
        except Exception:
            continue
        a = run_ppr(Pm, v, alpha=ALPHA, tol=TOL, max_iter=MAX_ITER_PROD)
        b = run_ppr(Pm, v, alpha=ALPHA, tol=1e-15, max_iter=20000)
        if not a.converged:
            n_nc += 1
        dH.append(abs(a.entropy - b.entropy))
        dHn.append(abs(a.entropy_norm - b.entropy_norm))
        ns.append(g.n)
        it_needed.append(b.iterations)

    print(f"   пересчитано блоков: {len(dH)}  "
          f"(вырожденных n<=1, ушедших в fallback: {n_deg})")
    print(f"   из них не сошлись при лимите 100: {n_nc} = "
          f"{n_nc/max(len(dH),1):.1%}  (в прогоне было 57.5 %)")
    print(f"   итераций до сходимости при снятом лимите: медиана "
          f"{st.median(it_needed):.0f}, макс {max(it_needed)}")
    print(f"   узлов в блоке: медиана {st.median(ns):.0f}, макс {max(ns)}")
    print(f"\n   |H(лимит 100) − H(сошёлся)|:      макс {max(dH):.3e}, "
          f"медиана {st.median(dH):.3e}")
    print(f"   |H_norm(100) − H_norm(сошёлся)|:  макс {max(dHn):.3e}, "
          f"медиана {st.median(dHn):.3e}")

    # ── 4. с чем это сравнивать ────────────────────────────────────────────
    M = [r for r in iter_records()
         if r['_tag'] == TAG and r['condition'] == 'M'
         and r.get('status') == 'ok']
    dh_obs = [abs(r['thermo']['delta_H']) for r in M]
    nz = [d for d in dh_obs if d > 1e-12]
    phi = [abs(r['thermo']['Phi']) for r in M if abs(r['thermo']['Phi']) > 1e-12]
    dis = [r['thermo']['d_iS'] for r in M if r['thermo']['d_iS'] > 1e-12]
    print("\n── 4. сопоставление масштабов ──")
    print(f"   |ΔH| в прогоне: нулевых {len(dh_obs)-len(nz)}/{len(dh_obs)}; "
          f"среди ненулевых медиана {st.median(nz):.4f}, "
          f"минимум {min(nz):.4f}")
    print(f"   |Φ| среди ненулевых: медиана {st.median(phi):.5f}, "
          f"минимум {min(phi):.2e}")
    print(f"   d_iS среди ненулевых: медиана {st.median(dis):.5f}")
    ratio = min(nz) / max(max(dHn), 1e-300)
    print(f"\n   САМЫЙ МАЛЫЙ ненулевой |ΔH| прогона больше САМОЙ БОЛЬШОЙ "
          f"ошибки от лимита в {ratio:.1e} раз")
    live = [r for r in M if abs(r['thermo']['delta_H']) > 1e-12]
    flips = sum(1 for r in live
                if abs(abs(r['thermo']['d_eS']) - r['thermo']['d_iS'])
                < max(dHn) * 2)
    print(f"   шагов с ненулевым ΔH: {len(live)}; из них таких, где |d_eS| и "
          f"d_iS ближе\n   друг к другу, чем удвоенная ошибка PPR (гейт alive "
          f"мог бы перевернуться): {flips}")
    print(f"   остальные {len(M)-len(live)} шагов вырождены не из-за PPR, "
          f"а из-за каузального порядка:\n   ΔH ≡ 0, потому что депт-луп "
          f"положил final_answer в блок до замера H_old")

    print("\n── вывод ──")
    print("   Предупреждение «PPR did not converge» — рассогласование tol и")
    print("   max_iter в core/ppr.py, а не дефект данных. Ошибка в H на "
          f"{max(dHn):.0e},")
    print("   на 4-5 порядков ниже самой мелкой величины, которую "
          "термодинамика различает.")
    print("   Формулировку «вся термодинамика стоит на шуме» из первой "
          "редакции")
    print("   LOG_MINING.md снимаю: она не подтверждается замером.")
    print("   Настоящие дефекты термодинамики — другие и уже описаны: ΔH ≡ 0 "
          "в 157/200")
    print("   шагов из-за каузального порядка, и H как статистика ОДНОГО "
          "текста.")
    print("   Правка всё равно нужна, но она косметическая: max_iter 100 → "
          "150,")
    print("   иначе 1891 ложное предупреждение маскирует настоящие.")

    out = {'iters_needed_theory': need, 'max_iter': MAX_ITER_PROD,
           'residual_median': st.median(res), 'residual_max': res[-1],
           'pi_distance_bound': bound,
           'dH_max': max(dH), 'dHn_max': max(dHn),
           'dH_median': st.median(dH),
           'blocks_recomputed': len(dH), 'not_converged': n_nc,
           'obs_dH_min_nonzero': min(nz), 'obs_dH_zero': len(dh_obs) - len(nz),
           'ratio_signal_to_error': ratio}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'out_ppr.json')
    json.dump(out, open(path, 'w', encoding='utf-8'), ensure_ascii=False,
              indent=1)
    print(f"\nсохранено: {path}")


if __name__ == '__main__':
    main()
