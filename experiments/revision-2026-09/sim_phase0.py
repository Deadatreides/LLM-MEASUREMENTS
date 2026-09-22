# -*- coding: utf-8 -*-
"""Этап 0.3: офлайн-расчёты по полным логитам роутера до всякого кода.

Трасса MYCSIG0 с полными значениями роутера (gx-prb.bin — gpt-oss-20b, 23 слоя × 32, top-4, сырые
логиты; gr-prb.bin — granite, вероятности, top-8). Первая половина — «заполнение», по ней учится набор;
замер — на второй.

  hot    смещение к набору на карте (Б2): в каждом слое H экспертов, горячих по первой половине;
         selection += b·RMS(строки)·[горячий]. Меры: доля выборов на карте, умножений на CPU на токен,
         изменённых решений, потеря массы маршрутизации (как в sim_shaping.py).
  ser    урезание экспертов: всегда top-Kmin, прочие выбранные — если p_i > t·p_0 (веса — softmax
         по выбранным логитам у gpt-oss, перенормированные вероятности у granite). Меры: средний k,
         доля веса отброшенных; и распределение бюджета по слоям жадным отбором по минимальной потере.
  union  заполнение ниже обрыва: объединение экспертов слоя на убатч для убатчей 128/512/1024 при
         смещении к набору из прошлых убатчей (LRU-модель кэша). Доля экспертов слоя, которую
         пришлось бы дочитать на убатч.

Запуск: python sim_phase0.py <трасса> <top_k>
"""
import os
import struct
import sys

import numpy as np


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
    return np.stack([per[il][:T] for il in layers], axis=1).astype(np.float64)   # [T, L, E]


def topk(z, k):
    return np.argpartition(-z, k - 1, axis=-1)[..., :k]


def is_prob(Z):
    return Z.min() >= 0 and abs(Z[:8].sum(-1).mean() - 1.0) < 1e-3


def full_probs(Z):
    if is_prob(Z):
        return Z / Z.sum(-1, keepdims=True)
    P = np.exp(Z - Z.max(-1, keepdims=True))
    return P / P.sum(-1, keepdims=True)


def sel_weights(Z, sel):
    """Веса выбранных экспертов так, как их считает модель."""
    v = np.take_along_axis(Z, sel, -1)
    if is_prob(Z):
        return v / v.sum(-1, keepdims=True)
    e = np.exp(v - v.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def sim_hot(Z, k):
    T, L, E = Z.shape
    t0 = T // 2
    sel_tr = topk(Z[:t0], k)
    freq = np.zeros((L, E))
    for li in range(L):
        freq[li] = np.bincount(sel_tr[:, li].ravel(), minlength=E)
    Pfull = full_probs(Z[t0:])
    rms = np.sqrt((Z[t0:] ** 2).mean(-1, keepdims=True))
    orig = topk(Z[t0:], k)
    po = np.take_along_axis(Pfull, orig, -1).sum(-1)
    print("\n[hot] смещение к набору на карте; замер по второй половине")
    print("%-6s %-5s %9s %12s %10s %10s" % ("доля", "сила", "на карте", "умнож. CPU", "изменено", "потеря"))
    for frac in (0.125, 0.25):
        H = max(1, int(round(frac * E)))
        hot = np.zeros((L, E), bool)
        for li in range(L):
            hot[li, np.argsort(-freq[li])[:H]] = True
        for b in (0.0, 0.25, 0.5, 1.0):
            z = Z[t0:] + b * rms * hot[None, :, :]
            sel = topk(z, k)
            on = np.take_along_axis(np.broadcast_to(hot, z.shape), sel, -1)
            cpu = (~on).sum(-1).sum(-1).mean()
            pb = np.take_along_axis(Pfull, sel, -1).sum(-1)
            lost = float(((po - np.minimum(pb, po)) / po).mean())
            same = (np.sort(orig, -1) == np.sort(sel, -1)).all(-1)
            print("%-6.3f %-5.2f %8.1f%% %12.2f %9.1f%% %9.2f%%"
                  % (frac, b, 100 * on.mean(), cpu, 100 * (1 - same.mean()), 100 * lost))


def sim_ser(Z, k):
    T, L, E = Z.shape
    t0 = T // 2
    sel = topk(Z[t0:], k)
    w = sel_weights(Z[t0:], sel)                                   # [T', L, k]
    ws = -np.sort(-w, -1)                                          # по убыванию
    print("\n[ser] урезание: всегда top-Kmin, прочие при p_i > t·p_0; вес отброшенных — доля от 1")
    print("%-5s %-5s %8s %10s %10s" % ("Kmin", "t", "ср. k", "отброшено", "худш. слой"))
    for kmin in range(1, k):
        for t in (0.1, 0.2, 0.3, 0.4, 0.6):
            keep = np.ones_like(ws, bool)
            keep[..., kmin:] = ws[..., kmin:] > t * ws[..., :1]
            kk = keep.sum(-1)
            dropped = (ws * ~keep).sum(-1)
            per_layer = dropped.mean(0)
            print("%-5d %-5.2f %8.2f %9.2f%% %9.2f%%" % (kmin, t, kk.mean(), 100 * dropped.mean(), 100 * per_layer.max()))
    # бюджет по слоям: жадно снимаем по одному эксперту (с конца по весу) там, где потеря меньше
    print("\n[ser] бюджет по слоям (жадно по средней потере веса), средний k против отброшенного веса:")
    kl = np.full(L, k)
    loss_if = lambda li, kk: float(ws[:, li, kk:].sum(-1).mean()) if kk < k else 0.0
    total = 0.0
    for step in range(L * (k - 1)):
        best, best_li = None, -1
        for li in range(L):
            if kl[li] > 1:
                d = loss_if(li, kl[li] - 1) - loss_if(li, kl[li])
                if best is None or d < best:
                    best, best_li = d, li
        kl[best_li] -= 1
        total = sum(loss_if(li, kl[li]) for li in range(L)) / L
        if step % max(1, L // 2) == 0 or kl.mean() <= 2.0:
            print("  ср. k %.2f  отброшено %.2f%%  k по слоям %s" % (kl.mean(), 100 * total, "".join(str(x) for x in kl)))
        if kl.mean() <= 2.0:
            break


def sim_union(Z, k):
    T, L, E = Z.shape
    rms = np.sqrt((Z ** 2).mean(-1, keepdims=True))
    Pfull = full_probs(Z)
    orig_all = topk(Z, k)
    po_all = np.take_along_axis(Pfull, orig_all, -1).sum(-1)
    print("\n[union] доля экспертов слоя в объединении убатча (что дочитывать при пустом кэше)")
    print("и дочитка при кэше в C экспертов слоя (LRU по прошлым убатчам), смещение к содержимому кэша")
    print("%-6s %-5s %-5s %10s %12s %10s" % ("убатч", "C", "сила", "объедин.", "дочитка", "потеря"))
    for ub in (128, 512, 1024):
        n_ub = T // ub
        if n_ub < 2:
            continue
        for C in (8, 16):
            for b in (0.0, 0.5, 1.0, 2.0):
                cache = [list() for _ in range(L)]        # LRU: список номеров, последний — свежий
                unions, misses, lost = [], [], []
                for u in range(n_ub):
                    zz = Z[u * ub:(u + 1) * ub]
                    inC = np.zeros((L, E), bool)
                    for li in range(L):
                        inC[li, cache[li]] = True
                    z = zz + b * rms[u * ub:(u + 1) * ub] * inC[None]
                    sel = topk(z, k)
                    pb = np.take_along_axis(Pfull[u * ub:(u + 1) * ub], sel, -1).sum(-1)
                    po = po_all[u * ub:(u + 1) * ub]
                    lost.append(float(((po - np.minimum(pb, po)) / po).mean()))
                    for li in range(L):
                        used = np.unique(sel[:, li])
                        unions.append(len(used) / E)
                        if u > 0:
                            misses.append(float(np.sum(~inC[li, used])) / E)
                        lst = [e for e in cache[li] if e not in set(used.tolist())] + used.tolist()
                        cache[li] = lst[-C:]
                print("%-6d %-5d %-5.1f %9.1f%% %11.1f%% %9.2f%%"
                      % (ub, C, b, 100 * np.mean(unions), 100 * np.mean(misses), 100 * np.mean(lost)))


def main():
    path, k = sys.argv[1], int(sys.argv[2])
    Z = load_values(path)
    T, L, E = Z.shape
    print(f"трасса {os.path.basename(path)}: токенов {T}, слоёв {L}, экспертов {E}, top-{k}, "
          f"{'вероятности' if is_prob(Z) else 'логиты'}")
    sim_hot(Z, k)
    sim_ser(Z, k)
    sim_union(Z, k)
    print("RUN-COMPLETE")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
