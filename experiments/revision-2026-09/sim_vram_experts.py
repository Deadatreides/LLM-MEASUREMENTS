# -*- coding: utf-8 -*-
"""Эксперты в видеопамять поштучно, а не слоями: сколько счёта уходит с процессора при том же бюджете.

В целевом режиме (глубина 16–65 тыс., --fit) машина упирается в процессорное ядро MXFP4, а --fit
заполняет карту ЦЕЛЫМИ слоями экспертов: 6 слоёв из 24 на 16 тыс., 3 на 65 тыс. Каждый такой слой
снимает с процессора ровно свою долю выборов, m/L. Если класть на карту те же байты, но по
горячим экспертам каждого слоя (кластеры и отбор под бюджет — поверх -ot), процессору останется
меньше экспертных умножений. Здесь считается, насколько меньше, — по трассе маршрутизатора,
без утечки: частоты по первой половине, счёт по второй.

Размещения при бюджете c (доля всех срезов (слой, эксперт)):
  layers   целые слои, round(c*L) штук — так делает --fit
  equal    в каждом слое top-round(c*E) экспертов по частоте
  global   top-round(c*L*E) срезов по частоте во всей модели (слои с острым маржиналом берут больше)

Меры на токен (вторая половина трассы):
  cpu_exp     экспертных умножений на процессоре (из L*k)
  cpu_layers  слоёв, где процессор нужен хотя бы для одного выбора (переходы карта-хост)
  gpu_full    слоёв, где все k выборов на карте

Запуск: python sim_vram_experts.py <трасса MYCSIG0> <top_k> [доли через запятую]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim_clusters as sc                                    # noqa: E402


def measure(sel_te, hot):
    """hot: [L, E] bool — что на карте."""
    T, L, k = sel_te.shape
    on_gpu = hot[np.arange(L)[None, :, None], sel_te]         # [T, L, k]
    cold = (~on_gpu).sum(-1)                                  # [T, L]
    return dict(cpu_exp=float(cold.sum(1).mean()),
                cpu_layers=float((cold > 0).sum(1).mean()),
                gpu_full=float((cold == 0).sum(1).mean()))


def main():
    path, k = sys.argv[1], int(sys.argv[2])
    fracs = [float(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "0.125,0.25,0.4167").split(",")]
    sel, E, layers, dropped = sc.load(path, k)
    T, L, _ = sel.shape
    t0 = T // 2
    f = np.zeros((L, E))
    for li in range(L):
        f[li] = np.bincount(sel[:t0, li].ravel(), minlength=E)
    sel_te = sel[t0:]
    print("трасса %s: токенов %d (замер по второй половине), слоёв %d, экспертов %d, top-%d"
          % (os.path.basename(path), T, L, E, k))
    print("%-7s %-8s %9s %11s %9s %14s" % ("бюджет", "размещ.", "cpu_exp", "cpu_layers", "gpu_full", "cpu_exp к layers"))
    for c in fracs:
        base = None
        for name in ("layers", "equal", "global"):
            hot = np.zeros((L, E), bool)
            if name == "layers":
                hot[L - int(round(c * L)):] = True
            elif name == "equal":
                n = int(round(c * E))
                for li in range(L):
                    hot[li, np.argsort(-f[li])[:n]] = True
            else:
                n = int(round(c * L * E))
                idx = np.argsort(-f.ravel())[:n]
                hot.ravel()[idx] = True
            r = measure(sel_te, hot)
            if base is None:
                base = r
            print("%-7.3f %-8s %9.2f %11.2f %9.2f %13.2fx"
                  % (c, name, r["cpu_exp"], r["cpu_layers"], r["gpu_full"], r["cpu_exp"] / base["cpu_exp"]))
    print("RUN-COMPLETE")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
