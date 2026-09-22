# -*- coding: utf-8 -*-
"""Цена промаха: страничное чтение отображённого файла против явного крупного чтения.

Обе симуляции (sim_clusters.py, sim_timeline.py) показали: главный множитель дают не
кластеры и не упреждение, а перевод промаха из страничного чтения (в деле 128 МБ/с, §4.105 —
выведено из разности времён, а не замерено) в явный крупный запрос (367 МБ/с — лабораторный
профиль §4.103 при небуферизованном чтении). Это допущение. Здесь оно меряется прямо, на
холодном файле модели, срезами размера эксперта gpt-oss (12.6 МиБ):

  fault      копия отображённого участка — чтение страничными промахами, как у ядра MXFP4
  read       один ReadFile на весь срез через кэш ОС
  read1m     ReadFile кусками по 1 МиБ через кэш ОС
  prefetch   PrefetchVirtualMemory на отображённый участок, затем копия

Смещения случайные и неперекрывающиеся, методы чередуются. Если медиана какого-то метода
выше 1500 МБ/с — участки уже лежали в кэше, и замер негоден (печатается предупреждение).
Запуск: python bench_fault_vs_read.py <файл> [срезов на метод, 16]
"""
import ctypes
import ctypes.wintypes as wt
import mmap
import os
import random
import statistics
import sys
import time

PATH = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 16
SLICE = int(12.6 * 1024 * 1024) // 4096 * 4096
MIB = 1024 * 1024


class RANGE(ctypes.Structure):
    _fields_ = [("VirtualAddress", ctypes.c_void_p), ("NumberOfBytes", ctypes.c_size_t)]


k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.GetCurrentProcess.restype = wt.HANDLE
k32.PrefetchVirtualMemory.argtypes = [wt.HANDLE, ctypes.c_size_t, ctypes.POINTER(RANGE), wt.ULONG]
k32.PrefetchVirtualMemory.restype = wt.BOOL


def main():
    size = os.path.getsize(PATH)
    rng = random.Random(time.time_ns())
    head = 256 * MIB
    slots = (size - head - SLICE) // SLICE
    methods = ["fault", "read", "read1m", "prefetch"]
    picks = rng.sample(range(slots), N * len(methods) + 4)
    offs = [head + p * SLICE for p in picks]

    f = open(PATH, "rb", buffering=0)
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY)
    anchor = ctypes.c_char.from_buffer(mm)
    base = ctypes.addressof(anchor)
    buf = bytearray(SLICE)
    mv = memoryview(buf)
    proc = k32.GetCurrentProcess()

    def run(method, off):
        t = time.perf_counter()
        if method == "fault":
            _ = mm[off:off + SLICE]
        elif method == "read":
            f.seek(off)
            got = f.readinto(buf)
            assert got == SLICE
        elif method == "read1m":
            f.seek(off)
            pos = 0
            while pos < SLICE:
                got = f.readinto(mv[pos:min(pos + MIB, SLICE)])
                assert got > 0
                pos += got
        else:
            r = RANGE(base + off, SLICE)
            if not k32.PrefetchVirtualMemory(proc, 1, ctypes.byref(r), 0):
                raise OSError(ctypes.get_last_error(), "PrefetchVirtualMemory")
            _ = mm[off:off + SLICE]
        return time.perf_counter() - t

    plan = [(methods[i % len(methods)], offs[i]) for i in range(N * len(methods))]
    rng.shuffle(plan)
    res = {m: [] for m in methods}
    for method, off in plan:
        dt = run(method, off)
        res[method].append(SLICE / MIB / dt)

    hot = [SLICE / MIB / run("fault", off) for _, off in plan[:4]]

    print("файл %s, %.2f ГиБ; срез %.1f МиБ; по %d срезов на метод, порядок перемешан"
          % (os.path.basename(PATH), size / 2**30, SLICE / MIB, N))
    print("%-10s %9s %9s %9s %9s" % ("метод", "медиана", "среднее", "мин", "макс"))
    bad = False
    for m in methods:
        v = res[m]
        med = statistics.median(v)
        bad |= med > 1500
        print("%-10s %6.0f МиБ/с %6.0f %9.0f %9.0f" % (m, med, statistics.mean(v), min(v), max(v)))
    print("повторное чтение тех же срезов (кэш): медиана %.0f МиБ/с" % statistics.median(hot))
    if bad:
        print("ВНИМАНИЕ: медиана выше 1500 МиБ/с — участки были в кэше, замер негоден")
    del anchor
    mm.close()
    f.close()
    print("RUN-COMPLETE")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
