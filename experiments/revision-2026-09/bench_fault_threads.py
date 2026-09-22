# -*- coding: utf-8 -*-
"""Страничное чтение среза эксперта шестью потоками — как его читает ядро умножения.

bench_fault_vs_read.py показал на холодном файле: однопоточная копия отображённого среза идёт
335 МиБ/с, явное чтение 366–380 — разница около 10 %, а не втрое, как выводил §4.105. Но там
копия шла одним потоком. Ядро llama.cpp делит строки матрицы между потоками, и страничные
промахи приходят одновременно из нескольких мест среза. Если одновременные промахи обслуживаются
хуже одного последовательного потока, вывод §4.105 мог быть верен для настоящего доступа.

  fault1   ctypes.memmove всего среза одним потоком (GIL на время копии отпущен)
  fault6   шесть потоков одновременно, каждый — свой непрерывный шестой кусок среза
  read1m   явное чтение кусками по 1 МиБ через кэш ОС

Запуск: python bench_fault_threads.py <холодный файл> [срезов на метод, 16]
"""
import ctypes
import mmap
import os
import random
import statistics
import sys
import threading
import time

PATH = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 16
SLICE = int(12.6 * 1024 * 1024) // 4096 * 4096
MIB = 1024 * 1024
THREADS = 6


def main():
    size = os.path.getsize(PATH)
    rng = random.Random(time.time_ns())
    head = 256 * MIB
    slots = (size - head - SLICE) // SLICE
    methods = ["fault1", "fault6", "read1m"]
    picks = rng.sample(range(slots), N * len(methods) + 4)
    offs = [head + p * SLICE for p in picks]

    f = open(PATH, "rb", buffering=0)
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY)
    anchor = ctypes.c_char.from_buffer(mm)
    base = ctypes.addressof(anchor)
    dst = (ctypes.c_char * SLICE)()
    dst_addr = ctypes.addressof(dst)
    buf = bytearray(SLICE)
    mv = memoryview(buf)
    part = SLICE // THREADS // 4096 * 4096

    def run(method, off):
        t = time.perf_counter()
        if method == "fault1":
            ctypes.memmove(dst_addr, base + off, SLICE)
        elif method == "fault6":
            ths = []
            for i in range(THREADS):
                a = i * part
                b = SLICE if i == THREADS - 1 else a + part
                ths.append(threading.Thread(target=ctypes.memmove,
                                            args=(dst_addr + a, base + off + a, b - a)))
            for th in ths:
                th.start()
            for th in ths:
                th.join()
        else:
            f.seek(off)
            pos = 0
            while pos < SLICE:
                got = f.readinto(mv[pos:min(pos + MIB, SLICE)])
                assert got > 0
                pos += got
        return time.perf_counter() - t

    plan = [(methods[i % len(methods)], offs[i]) for i in range(N * len(methods))]
    rng.shuffle(plan)
    res = {m: [] for m in methods}
    for method, off in plan:
        res[method].append(SLICE / MIB / run(method, off))
    hot = [SLICE / MIB / run("fault1", off) for _, off in plan[:4]]

    print("файл %s, %.2f ГиБ; срез %.1f МиБ; по %d срезов на метод; потоков %d"
          % (os.path.basename(PATH), size / 2**30, SLICE / MIB, N, THREADS))
    print("%-8s %9s %9s %9s %9s" % ("метод", "медиана", "среднее", "мин", "макс"))
    bad = False
    for m in methods:
        v = res[m]
        med = statistics.median(v)
        bad |= med > 1500
        print("%-8s %6.0f МиБ/с %6.0f %9.0f %9.0f" % (m, med, statistics.mean(v), min(v), max(v)))
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
