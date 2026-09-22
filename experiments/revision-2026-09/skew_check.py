# -*- coding: utf-8 -*-
"""Проверка посылки DOC-SIGMA-FULL §1.2 / §16.2: «aux loss выравнивает маржинал экспертов,
поэтому кэш самых горячих экспертов даёт попадание ровно N_cached/n_expert — головы нет».

На этой посылке в спецификации отброшены нативные эксперты как единица резидентности (и
построены атомы и псевдоэксперты). Проверяется на тех же трассах, что и симуляции, без утечки:
частоты — по первой половине, покрытие — по второй.

  покрытие(c) = доля выборов второй половины, попавших в top-c долю экспертов слоя по частоте
                первой половины; «головы нет» означает покрытие(c) ≈ c
  эфф. доля   = exp(энтропия маржинала второй половины) / n_expert — 1.0 при равномерном

Запуск: python skew_check.py <трасса MYCSIG0> <top_k>
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim_clusters as sc                                    # noqa: E402

FRACS = (0.125, 0.28, 0.45)


def main():
    path, k = sys.argv[1], int(sys.argv[2])
    sel, E, layers, dropped = sc.load(path, k)
    T, L, _ = sel.shape
    t0 = T // 2
    cov = {c: [] for c in FRACS}
    cov_same = {c: [] for c in FRACS}
    eff = []
    for li in range(L):
        tr = np.bincount(sel[:t0, li].ravel(), minlength=E).astype(float)
        te = np.bincount(sel[t0:, li].ravel(), minlength=E).astype(float)
        p = te / te.sum()
        nz = p[p > 0]
        eff.append(float(np.exp(-(nz * np.log(nz)).sum())) / E)
        for c in FRACS:
            n = max(1, int(round(c * E)))
            top_tr = np.argsort(-tr)[:n]
            top_te = np.argsort(-te)[:n]
            cov[c].append(te[top_tr].sum() / te.sum())
            cov_same[c].append(te[top_te].sum() / te.sum())
    print("трасса %s: токенов %d, слоёв %d, экспертов %d, top-%d" % (os.path.basename(path), T, L, E, k))
    print("эфф. доля экспертов по энтропии маржинала (вторая половина): медиана %.3f, мин %.3f, макс %.3f"
          % (np.median(eff), min(eff), max(eff)))
    print("%-8s %-28s %-28s" % ("доля c", "покрытие, частоты 1-й пол.", "потолок: частоты 2-й пол."))
    for c in FRACS:
        a, b = np.array(cov[c]), np.array(cov_same[c])
        print("%-8.3f медиана %.3f (%.3f..%.3f) = %.2fc   медиана %.3f = %.2fc"
              % (c, np.median(a), a.min(), a.max(), np.median(a) / c, np.median(b), np.median(b) / c))
    print("RUN-COMPLETE")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
