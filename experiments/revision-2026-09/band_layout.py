# -*- coding: utf-8 -*-
r"""band_layout.py — цена раскладки в секундах на токен, а не в МБ/с.

ЗАЧЕМ. `io_curve.py` дал кривую устройства по размеру порции. Здесь она переводится в
величину, ради которой всё делается: **сколько секунд на токен стоит чтение разреженного
FFN** при трёх раскладках одного и того же объёма.

Геометрия Qwen3.8-27B в Q8_0 (hidden 5120, FFN 17408, блок 34 байта на 32 веса):
  строка нейрона gate/up = 5 440 байт
  полоса 64 нейрона      = 348 160 байт
  полос на слое          = 272

ТРИ РАСКЛАДКИ, одинаковый объём чтения:
  1. `rows`     — строки нейронов вразброс: так лежит файл БЕЗ перестановки;
  2. `bands`    — полосы по 64 нейрона вразброс: перестановка собрала нейроны в полосы,
                  но сами полосы разбросаны;
  3. `grouped`  — отобранные полосы лежат подряд: перестановка ещё и упорядочила их.

Каждая раскладка меряется при очереди 1 и 8. Чтение небуферизованное, мимо кэша.

Запуск: python band_layout.py <файл.gguf> [доля_активных] [слоёв_в_замере]
"""
from __future__ import annotations

import ctypes
import os
import random
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from io_curve import SECTOR, k32, open_unbuffered, read_at, MEM_COMMIT_RESERVE, PAGE_READWRITE

ROW = 5440                 # строка нейрона gate/up в Q8_0
BAND_NEURONS = 64
BAND = ROW * BAND_NEURONS  # 348 160
N_BANDS_PER_LAYER = 17408 // BAND_NEURONS   # 272
N_LAYERS = 64


def align_down(x):
    return (x // SECTOR) * SECTOR


def run(path, reqs, depth):
    """reqs — список (offset, size), уже выровненных. Возвращает (секунды, байт)."""
    fsize = os.path.getsize(path)
    stop = threading.Event()
    idx = [0]
    lock = threading.Lock()
    got = [0] * depth

    def worker(w):
        h = open_unbuffered(path)
        buf = k32.VirtualAlloc(None, ctypes.c_size_t(8 << 20), MEM_COMMIT_RESERVE, PAGE_READWRITE)
        try:
            while True:
                with lock:
                    i = idx[0]
                    idx[0] += 1
                if i >= len(reqs) or stop.is_set():
                    return
                off, size = reqs[i]
                if off + size > fsize:
                    continue
                got[w] += read_at(h, buf, size, off)
        finally:
            k32.CloseHandle(h)

    ts = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(depth)]
    t0 = time.time()
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=300)
    return time.time() - t0, sum(got)


def build(path, kind, frac, layers, seed=7):
    """Список запросов для одной раскладки."""
    fsize = align_down(os.path.getsize(path))
    rnd = random.Random(seed)
    n_sel = max(1, int(N_BANDS_PER_LAYER * frac))
    reqs = []
    for _ in range(layers):
        if kind == "rows":
            # строки вразброс: тот же объём, что n_sel полос, но запросами по строке
            for _ in range(n_sel * BAND_NEURONS):
                off = align_down(rnd.randrange(0, fsize - 8192))
                reqs.append((off, 8192))        # 5 440 -> ближайший законный размер 8 КиБ
        elif kind == "bands":
            for _ in range(n_sel):
                off = align_down(rnd.randrange(0, fsize - BAND))
                reqs.append((off, BAND))
        elif kind == "grouped":
            # отобранные полосы подряд: один пробег, нарезанный на порции по 64 КиБ
            total = n_sel * BAND
            start = align_down(rnd.randrange(0, max(1, fsize - total)))
            chunk = 64 * 1024
            done = 0
            while done < total:
                sz = min(chunk, total - done)
                sz = align_down(sz) or SECTOR
                reqs.append((start + done, sz))
                done += sz
        else:
            raise ValueError(kind)
    return reqs


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    path = argv[1]
    frac = float(argv[2]) if len(argv) > 2 else 0.35
    layers = int(argv[3]) if len(argv) > 3 else 4

    n_sel = max(1, int(N_BANDS_PER_LAYER * frac))
    per_layer = n_sel * BAND
    print("файл %s" % os.path.basename(path))
    print("доля активных %.0f %%: %d полос из %d на слой, %.1f МиБ на слой, %.2f ГиБ на токен (64 слоя)"
          % (frac * 100, n_sel, N_BANDS_PER_LAYER, per_layer / 2**20, per_layer * N_LAYERS / 2**30))
    print("в замере %d слоёв\n" % layers)

    print("%-38s %10s %10s %14s" % ("раскладка", "очередь 1", "очередь 8", "с/токен при оч.8"))
    print("-" * 76)
    out = {}
    for kind, label in (("rows", "строки вразброс (без перестановки)"),
                        ("bands", "полосы вразброс"),
                        ("grouped", "полосы подряд, порции 64 КиБ")):
        row = "%-38s" % label
        for depth in (1, 8):
            reqs = build(path, kind, frac, layers)
            dt, nb = run(path, reqs, depth)
            mbs = nb / dt / 1e6
            out[(kind, depth)] = mbs
            row += "%10.1f" % mbs
        # секунды на токен: весь FFN-трафик за токен при этой скорости
        per_token = per_layer * N_LAYERS
        row += "%14.2f" % (per_token / (out[(kind, 8)] * 1e6))
        print(row, flush=True)

    print()
    b = out[("grouped", 8)] / out[("rows", 8)]
    print("подряд против строк вразброс при очереди 8: %.2fx" % b)
    print("полосы вразброс против строк вразброс, очередь 8: %.2fx"
          % (out[("bands", 8)] / out[("rows", 8)]))
    print("очередь 8 против 1: строки %.2fx, полосы %.2fx, подряд %.2fx"
          % (out[("rows", 8)] / out[("rows", 1)],
             out[("bands", 8)] / out[("bands", 1)],
             out[("grouped", 8)] / out[("grouped", 1)]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
