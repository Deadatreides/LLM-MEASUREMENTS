# -*- coding: utf-8 -*-
"""Спиновое стекло как поле на маршрутизатор: офлайн-фильтр по полным логитам роутера.

Симуляции упреждения закрыли за стеклом и кластерами роль предсказателя: связи не лучше
теплоты, кластеры несут мало информации, лишнее чтение вредит. Остаётся роль, в которой
стекло не угадывает спрос, а МЕНЯЕТ его: смещение логитов роутера к резидентному набору
(то, что в форке делает MYC_ROUTE_BIAS). Здесь связи могут решать другое — сколько стоит
смещение: набор, внутренне соактивированный, может перехватывать выбор с меньшей потерей
массы маршрутизации, чем набор горячих поодиночке.

Смещение — ровно как в форке: selection += b * RMS(строки selection) * [срез в наборе R].
У gpt-oss ffn_moe_probs — сырые логиты (SOFTMAX_WEIGHT), и трасса хранит их для всех экспертов.

Наборы R (обновляются на границе EPOCH токенов, кроме cache):
  cache    текущее содержимое кэша LRU — смещение к тому, что уже в памяти
  heat     top-K по теплоте (EMA спроса), K = ёмкость
  ising    основное состояние стекла приближённо: теплота + lam * связи J (положительная PMI
           со-обращений в окне, выучена на первой половине трассы)

Меры на второй половине трассы: промахи кэша, доля изменённых выборов, потеря массы
маршрутизации = доля вероятности исходных top-k, не попавшая в смещённые top-k (softmax по
всем экспертам). Первое приближение: скрытые состояния не пересчитываются, то есть влияние
изменённого выбора на следующие слои и токены не учтено. Это фильтр, а не замер качества.

Запуск: python sim_shaping.py <трасса MYCSIG0> [top_k=4]
"""
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim_clusters as sc                                    # noqa: E402

UNIT_MB = 13.2
DISK_MBS = 350.0
LAYER_MS = 10.0
WIN = 4
EPOCH = 16
CAPS = [float(x) for x in os.environ.get("CAPS", "0.28,0.45").split(",")]
BIASES = [float(x) for x in os.environ.get("BIASES", "0,0.25,0.5,1,2").split(",")]


def load_values(path):
    per = {}
    with open(path, "rb") as f:
        assert f.read(8)[:7] == b"MYCSIG0"
        width, _ = struct.unpack("<ii", f.read(8))
        while True:
            h = f.read(8)
            if len(h) < 8:
                break
            il, nt = struct.unpack("<ii", h)
            raw = f.read(nt * width * 4)
            if len(raw) < nt * width * 4:
                break
            per.setdefault(il, []).append(np.frombuffer(raw, dtype=np.float32).reshape(nt, width))
    per = {il: np.concatenate(v, 0) for il, v in per.items()}
    full = max(v.shape[0] for v in per.values())
    layers = [il for il in sorted(per) if per[il].shape[0] >= 0.9 * full]
    T = min(per[il].shape[0] for il in layers)
    return np.stack([per[il][:T] for il in layers], axis=1).astype(np.float64), width   # [T, L, E]


def topk(z, k):
    return np.argpartition(-z, k - 1, axis=-1)[..., :k]


def run(Z, t0, k, cap, variant, b, J=None, lam=0.0):
    T, L, E = Z.shape
    U = L * E
    rms = np.sqrt((Z ** 2).mean(-1, keepdims=True))
    if Z.min() >= 0 and abs(Z[:8].sum(-1).mean() - 1.0) < 1e-3:
        Pfull = Z / Z.sum(-1, keepdims=True)              # трасса уже хранит вероятности (SOFTMAX)
    else:
        Pfull = np.exp(Z - Z.max(-1, keepdims=True))      # сырые логиты (SOFTMAX_WEIGHT)
        Pfull /= Pfull.sum(-1, keepdims=True)
    in_cache = np.zeros(U, bool)
    last = np.full(U, -(1 << 40), np.int64)
    heat = np.zeros(U)
    R = np.zeros(U, bool)
    miss = refs = changed = 0
    lost = 0.0
    n_lost = 0
    start = max(t0 - 200, 0)
    offs = (np.arange(L) * E)[:, None]
    for t in range(start, T):
        counted = t >= t0
        if variant == "cache":
            R = in_cache.copy()
        elif (t - start) % EPOCH == 0:
            s = np.zeros(U, bool)
            s[np.argpartition(-heat, cap - 1)[:cap]] = True
            if variant == "ising" and lam > 0:
                for _ in range(5):
                    score = heat + lam * (J @ (s * heat)) / cap
                    s[:] = False
                    s[np.argpartition(-score, cap - 1)[:cap]] = True
            R = s
        z = Z[t] + b * rms[t] * R.reshape(L, E)
        sel = topk(z, k)
        need = (offs + sel).ravel()
        if counted:
            orig = topk(Z[t], k)
            refs += len(need)
            hit = in_cache[need]
            miss += int((~hit).sum())
            po = np.take_along_axis(Pfull[t], orig, -1).sum(-1)
            pb = np.take_along_axis(Pfull[t], sel, -1).sum(-1)
            lost += float(((po - np.minimum(pb, po)) / po).sum())
            n_lost += L
            same = (np.sort(orig, -1) == np.sort(sel, -1)).all(-1)
            changed += int((~same).sum())
        new = need[~in_cache[need]]
        if len(new):
            free = cap - int(in_cache.sum())
            over = len(new) - free
            if over > 0:
                cand = np.flatnonzero(in_cache)
                cand = cand[~np.isin(cand, need)]
                victims = cand[np.argpartition(last[cand], over - 1)[:over]]
                in_cache[victims] = False
            in_cache[new] = True
        last[need] = t
        heat *= 0.9
        heat[need] += 0.1
    n_tok = T - t0
    miss_mb = miss * UNIT_MB / n_tok
    return dict(hit=1 - miss / refs, miss_mb=miss_mb, changed=changed / (n_tok * L),
                lost=lost / n_lost, ms=L * LAYER_MS + miss_mb / DISK_MBS * 1000)


def main():
    path = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    t_begin = time.time()
    Z, E = load_values(path)
    T, L, _ = Z.shape
    U = L * E
    t0 = T // 2
    rows = Z[:64].reshape(-1, E)
    print("трасса %s: токенов %d, слоёв %d, экспертов %d, top-%d; значения: мин %.3f макс %.3f, сумма строки %.3f"
          % (os.path.basename(path), T, L, E, k, rows.min(), rows.max(), rows.sum(1).mean()))
    sel_tr = topk(Z[:t0], k)
    J, _ = sc.couplings(sc.incidence(sel_tr, E), WIN)

    for frac in CAPS:
        cap = int(frac * U)
        print("\n=== ёмкость %.0f %% (%d срезов) ===" % (100 * frac, cap))
        print("%-14s %6s %8s %11s %10s %11s %9s %7s"
              % ("набор R", "сила", "попад.", "промах МБ", "изменено", "потеря массы", "мс/токен", "к b=0"))
        base = run(Z, t0, k, cap, "cache", 0.0)
        print("%-14s %6.2f %7.1f%% %11.1f %9.1f%% %10.2f%% %9.1f %6.2fx"
              % ("без смещения", 0, 100 * base["hit"], base["miss_mb"], 0, 0, base["ms"], 1))
        for variant, lam in (("cache", 0.0), ("heat", 0.0), ("ising", 1.0), ("ising", 4.0)):
            name = variant if variant != "ising" else "ising lam=%.0f" % lam
            for b in BIASES:
                if b == 0:
                    continue
                r = run(Z, t0, k, cap, variant, b, J, lam)
                print("%-14s %6.2f %7.1f%% %11.1f %9.1f%% %10.2f%% %9.1f %6.2fx"
                      % (name, b, 100 * r["hit"], r["miss_mb"], 100 * r["changed"], 100 * r["lost"],
                         r["ms"], base["ms"] / r["ms"]))
    print("\nготово за %.0f с" % (time.time() - t_begin))
    print("RUN-COMPLETE")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
