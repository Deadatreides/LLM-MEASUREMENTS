# -*- coding: utf-8 -*-
"""Набор горячих экспертов для MYC_HOT_EXPERTS из трассы маршрутизатора (MYCSIG0, ffn_moe_probs).

Частоты выбора top-k считаются по трассе КАЛИБРОВКИ; срезы (слой, эксперт) берутся по всей модели
в порядке частоты, пока не исчерпан бюджет видеопамяти. Слой с острым маржиналом берёт больше —
по sim_vram_experts.py это чуть лучше, чем поровну.

Если дана проверочная трасса (другой текст), печатается, сколько экспертных умножений на токен
останется процессору: при раскладке целыми слоями на тот же бюджет и при этом наборе.

Запуск: python hotset.py <трасса калибровки> <top_k> <бюджет МиБ> <срез МиБ> <выход.txt> [проверочная трасса]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_shaping import load_values, topk              # noqa: E402


def counts(path, k):
    Z, E = load_values(path)                             # [T, L, E]
    sel = topk(Z, k)                                     # [T, L, k]
    T, L, _ = sel.shape
    c = np.zeros((L, E))
    for li in range(L):
        c[li] = np.bincount(sel[:, li].ravel(), minlength=E)
    return sel, c, E


def main():
    calib, k, budget_mib, unit_mib, out = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
    check = sys.argv[6] if len(sys.argv) > 6 else None

    sel_c, c, E = counts(calib, k)
    T, L, _ = sel_c.shape
    n_units = int(budget_mib // unit_mib)
    n_units = max(0, min(n_units, L * E))
    order = np.argsort(-c.ravel(), kind="stable")[:n_units]
    hot = np.zeros((L, E), bool)
    hot.ravel()[order] = True

    with open(out, "w", encoding="utf-8") as f:
        f.write("# hotset.py: калибровка %s, top-%d, бюджет %.0f МиБ, срез %.2f МиБ, срезов %d\n"
                % (os.path.basename(calib), k, budget_mib, unit_mib, n_units))
        for li in range(L):
            es = np.flatnonzero(hot[li])
            if len(es):
                f.write("%d %s\n" % (li, " ".join(str(int(e)) for e in es)))

    per_layer = hot.sum(1)
    print("калибровка %s: токенов %d, слоёв %d, экспертов %d, top-%d" % (os.path.basename(calib), T, L, E, k))
    print("бюджет %.0f МиБ / срез %.2f МиБ = %d срезов (%.1f %% модели); по слоям: %s"
          % (budget_mib, unit_mib, n_units, 100.0 * n_units / (L * E), " ".join(str(int(x)) for x in per_layer)))

    def cpu_work(sel, hot_mask):
        on = hot_mask[np.arange(sel.shape[1])[None, :, None], sel]
        cold = (~on).sum(-1)
        return float(cold.sum(1).mean()), float((cold > 0).sum(1).mean())

    cov_c, _ = cpu_work(sel_c, hot)
    print("на самой калибровке: умножений на CPU %.1f из %d на токен" % (cov_c, L * k))
    if check:
        sel_v, _, E2 = counts(check, k)
        if E2 != E or sel_v.shape[1] != L:
            print("проверочная трасса другой формы — пропуск")
        else:
            m = n_units // E
            layers = np.zeros((L, E), bool)
            if m > 0:
                layers[L - m:] = True
            a_exp, a_lay = cpu_work(sel_v, layers)
            b_exp, b_lay = cpu_work(sel_v, hot)
            print("проверка %s (%d токенов): целые слои (%d) — %.1f умножений на CPU, слоёв с CPU %.1f; "
                  "горячие — %.1f, слоёв с CPU %.1f; отношение %.2fx"
                  % (os.path.basename(check), sel_v.shape[0], m, a_exp, a_lay, b_exp, b_lay, b_exp / max(a_exp, 1e-9)))
    print("записано: %s" % out)
    print("RUN-COMPLETE")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
