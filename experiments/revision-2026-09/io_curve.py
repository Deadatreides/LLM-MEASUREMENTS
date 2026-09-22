# -*- coding: utf-8 -*-
r"""io_curve.py — кривая накопителя по размеру порции и глубине очереди.

ЗАЧЕМ. Два переоткрытых вопроса, оба без сборки:

1. **Раскладка на полосах.** Прежний вердикт «раскладка файла ничего не даёт» снят на срезах
   эксперта ~12.6 МиБ — там кривая уже плоская, и эффект проявиться не мог. Единица полосы
   в 64 канала это ~340 КиБ, строка нейрона — 5.4 КиБ. Надо знать, что даёт накопитель на
   этих размерах.
2. **Глубина очереди.** Не пробовалась ни разу: подкачка мерилась одиночным потоком
   `PrefetchVirtualMemory`. SATA умеет NCQ, и одна операция в полёте шину не насыщает.

КАК. Небуферизованное чтение (`FILE_FLAG_NO_BUFFERING`) — мимо страничного кэша, поэтому
числа не зависят от того, что успело прогреться. Требования формата соблюдены: размер,
смещение и адрес буфера кратны сектору. Глубина очереди задаётся числом потоков, у каждого
свой дескриптор (общий с `SetFilePointerEx` был бы гонкой).

Запуск: python io_curve.py <файл> [секунд_на_точку]
"""
from __future__ import annotations

import ctypes
import os
import random
import sys
import threading
import time
from ctypes import wintypes

GENERIC_READ = 0x80000000
FILE_SHARE_READ = 0x00000001
OPEN_EXISTING = 3
FILE_FLAG_NO_BUFFERING = 0x20000000
FILE_FLAG_SEQUENTIAL_SCAN = 0x08000000
MEM_COMMIT_RESERVE = 0x3000
PAGE_READWRITE = 0x04
SECTOR = 4096

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.CreateFileW.restype = wintypes.HANDLE
k32.VirtualAlloc.restype = ctypes.c_void_p
k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wintypes.DWORD, wintypes.DWORD]
k32.SetFilePointerEx.argtypes = [wintypes.HANDLE, ctypes.c_longlong,
                                 ctypes.POINTER(ctypes.c_longlong), wintypes.DWORD]
k32.SetFilePointerEx.restype = wintypes.BOOL
k32.ReadFile.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                         ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
k32.ReadFile.restype = wintypes.BOOL


def open_unbuffered(path):
    h = k32.CreateFileW(path, GENERIC_READ, FILE_SHARE_READ, None, OPEN_EXISTING,
                        FILE_FLAG_NO_BUFFERING, None)
    if h == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    return h


def read_at(h, buf, size, offset):
    # SetFilePointerEx, а не SetFilePointer: файл больше 2 ГиБ, и 32-битное смещение
    # переполняется знаковым LONG (поймано прогоном).
    new_pos = ctypes.c_longlong(0)
    if not k32.SetFilePointerEx(h, ctypes.c_longlong(offset), ctypes.byref(new_pos), 0):
        raise ctypes.WinError(ctypes.get_last_error())
    got = wintypes.DWORD(0)
    if not k32.ReadFile(h, buf, wintypes.DWORD(size), ctypes.byref(got), None):
        raise ctypes.WinError(ctypes.get_last_error())
    return got.value


def point(path, size, depth, seconds, seq=False, seed=0):
    """Вернуть МБ/с для (размер порции, глубина очереди)."""
    fsize = os.path.getsize(path)
    span = (fsize // SECTOR) * SECTOR - size
    stop = threading.Event()
    counts = [0] * depth
    cursor = [0]
    lock = threading.Lock()

    def worker(idx):
        h = open_unbuffered(path)
        buf = k32.VirtualAlloc(None, ctypes.c_size_t(size), MEM_COMMIT_RESERVE, PAGE_READWRITE)
        rnd = random.Random(seed * 1000 + idx)
        try:
            while not stop.is_set():
                if seq:
                    with lock:
                        off = cursor[0]
                        cursor[0] += size
                        if off > span:
                            cursor[0] = 0
                            off = 0
                else:
                    off = (rnd.randrange(0, span // SECTOR)) * SECTOR
                n = read_at(h, buf, size, off)
                if n <= 0:
                    break
                counts[idx] += n
        finally:
            k32.CloseHandle(h)

    ts = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(depth)]
    t0 = time.time()
    for t in ts:
        t.start()
    time.sleep(seconds)
    stop.set()
    for t in ts:
        t.join(timeout=30)
    dt = time.time() - t0
    return sum(counts) / dt / 1e6


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    path = argv[1]
    secs = float(argv[2]) if len(argv) > 2 else 3.0
    sizes = [
        (4 * 1024, "4 КиБ"),
        (8 * 1024, "8 КиБ  ~строка нейрона 5.4"),
        (64 * 1024, "64 КиБ"),
        (348160, "340 КиБ  полоса 64 канала"),
        (1024 * 1024, "1 МиБ"),
        (4 * 1024 * 1024, "4 МиБ"),
    ]
    depths = [1, 2, 4, 8]
    print("файл %s (%.1f ГБ), по %.0f с на точку, небуферизованное случайное чтение"
          % (os.path.basename(path), os.path.getsize(path) / 1e9, secs))
    print("точек: %d" % (len(sizes) * len(depths) + len(sizes)))
    print()
    hdr = "%-28s" % "размер порции"
    for d in depths:
        hdr += "%10s" % ("очередь %d" % d)
    hdr += "%12s" % "послед. 1"
    print(hdr)
    print("-" * len(hdr))
    results = {}
    for size, label in sizes:
        row = "%-28s" % label
        for d in depths:
            v = point(path, size, d, secs)
            results[(size, d)] = v
            row += "%10.1f" % v
        vs = point(path, size, 1, secs, seq=True)
        results[(size, "seq")] = vs
        row += "%12.1f" % vs
        print(row, flush=True)
    print()
    best = max(results.values())
    b1 = results[(4096, 1)]
    print("разброс по таблице: %.1f — %.1f МБ/с, то есть %.1fx" % (b1, best, best / b1 if b1 else 0))
    band = results[(348160, 1)]
    row_ = results[(8192, 1)]
    print("полоса 340 КиБ против строки 8 КиБ при очереди 1: %.1f против %.1f — %.1fx"
          % (band, row_, band / row_ if row_ else 0))
    for size, label in sizes:
        v1, v8 = results[(size, 1)], results[(size, 8)]
        print("  очередь 8 против 1 на %-26s %.2fx" % (label + ":", v8 / v1 if v1 else 0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
