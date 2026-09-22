# -*- coding: utf-8 -*-
"""Циклический проход по «слоям» файла при кэше меньше цикла: что вытесняет Windows.

Вопрос поставлен плотной Qwen3.8-27B Q8_0. На каждый токен она проходит ствол слой за слоем: хосту
нужно ~21.7 ГиБ, а под страничный кэш остаётся ~18. Порядок обращений циклический: 0, 1, …, L−1,
и снова. Для такого порядка LRU — худшая политика. В кэше лежат только что пройденные слои, а нужны
следующие, и при кэше меньше цикла промахивается ВСЁ: перечитывается весь цикл. Оптимум (Белади)
перечитывает только дефицит, «цикл − кэш». Для 27B это 21.7 против ~4 ГиБ на токен.

Косвенный признак уже есть. PREFILL-DEFICIT.md: ниже обрыва каждый убатч gpt-oss-20b перечитывал
практически весь набор экспертов хоста (7.7 ГиБ при 7.6 ГиБ вне баллона). Но там доступ не чисто
циклический, и кэш был оценён грубо. Здесь проверяется в чистом виде.

Цикл — gpt-oss-20b-MXFP4.gguf, 11.28 ГиБ, «слои» по 384 МиБ. Кэш ограничен баллоном
(VirtualLock, как balloon_lock.py): вне баллона остаётся --outside ГиБ ОЗУ.

Режимы:
  stock    система сама решает, что вытеснять;
  pin      первые K слоёв закреплены VirtualLock, остальные идут потоком через промахи;
  pinread  как pin, плюс поток-читатель подкачивает PrefetchVirtualMemory следующий потоковый
           слой, пока «считается» текущий, а пройденный потоковый слой сразу выводится из рабочего
           набора (VirtualUnlock по незакреплённому диапазону).

Каждый слой читают шесть потоков memmove, по непрерывной шестой части — так ядро умножения делит
строки. На круге: байты с физического диска файла, время, минимум доступной памяти.

Запуск: python -u cyclic_scan.py [--outside 13.0] [--cycles 4] [--pin-k N]
Итог — строки CYCLE и SUMMARY, в конце RUN-COMPLETE. Без BALLOON-OK замер недействителен.
"""
import argparse
import ctypes
import ctypes.wintypes as wt
import os
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import psutil

GIB = 1 << 30
MIB = 1 << 20
PAGE = 4096

k32 = ctypes.WinDLL("kernel32", use_last_error=True)

GENERIC_READ = 0x80000000
FILE_SHARE_ALL = 0x1 | 0x2 | 0x4
OPEN_EXISTING = 3
FILE_ATTRIBUTE_NORMAL = 0x80
PAGE_READONLY = 0x02
FILE_MAP_READ = 0x0004
MEM_COMMIT = 0x1000
MEM_RESERVE = 0x2000
PAGE_READWRITE = 0x04
INVALID = ctypes.c_void_p(-1).value

k32.CreateFileW.restype = ctypes.c_void_p
k32.CreateFileW.argtypes = [wt.LPCWSTR, wt.DWORD, wt.DWORD, ctypes.c_void_p, wt.DWORD, wt.DWORD, ctypes.c_void_p]
k32.CreateFileMappingW.restype = ctypes.c_void_p
k32.CreateFileMappingW.argtypes = [ctypes.c_void_p, ctypes.c_void_p, wt.DWORD, wt.DWORD, wt.DWORD, wt.LPCWSTR]
k32.MapViewOfFile.restype = ctypes.c_void_p
k32.MapViewOfFile.argtypes = [ctypes.c_void_p, wt.DWORD, wt.DWORD, wt.DWORD, ctypes.c_size_t]
k32.VirtualAlloc.restype = ctypes.c_void_p
k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wt.DWORD, wt.DWORD]
k32.VirtualFree.restype = wt.BOOL
k32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wt.DWORD]
k32.VirtualLock.restype = wt.BOOL
k32.VirtualLock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
k32.VirtualUnlock.restype = wt.BOOL
k32.VirtualUnlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
k32.SetProcessWorkingSetSize.restype = wt.BOOL
k32.SetProcessWorkingSetSize.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
k32.GetCurrentProcess.restype = ctypes.c_void_p


class RANGE(ctypes.Structure):
    _fields_ = [("VirtualAddress", ctypes.c_void_p), ("NumberOfBytes", ctypes.c_size_t)]


k32.PrefetchVirtualMemory.restype = wt.BOOL
k32.PrefetchVirtualMemory.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(RANGE), wt.ULONG]

HPROC = k32.GetCurrentProcess()


def avail_gib():
    return psutil.virtual_memory().available / GIB


def disk_read(drive):
    return psutil.disk_io_counters(perdisk=True)[drive].read_bytes


def set_ws(min_bytes):
    for i in range(10):
        if k32.SetProcessWorkingSetSize(HPROC, min_bytes, min_bytes + 2 * GIB):
            return True
        print("SetProcessWorkingSetSize otkaz %d, popytka %d" % (ctypes.get_last_error(), i + 1), flush=True)
        time.sleep(3)
    return False


def map_file(path):
    h = k32.CreateFileW(path, GENERIC_READ, FILE_SHARE_ALL, None, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, None)
    if h in (None, INVALID):
        raise OSError("CreateFileW %d" % ctypes.get_last_error())
    m = k32.CreateFileMappingW(h, None, PAGE_READONLY, 0, 0, None)
    if not m:
        raise OSError("CreateFileMappingW %d" % ctypes.get_last_error())
    p = k32.MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0)
    if not p:
        raise OSError("MapViewOfFile %d" % ctypes.get_last_error())
    return p


def physical_drive_of(path):
    # E: на этой машине — PhysicalDrive3 (ADATA SU650); проверяем по росту счётчика при чтении
    return os.environ.get("SCAN_DRIVE", "PhysicalDrive3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=r"<MODELS>\gpt-oss-20b-MXFP4.gguf")
    ap.add_argument("--outside", type=float, default=13.0, help="ОЗУ вне баллона, ГиБ")
    ap.add_argument("--pin-k", type=int, default=-1, help="сколько слоёв закрепить; −1 — по оценке кэша")
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--chunk-mib", type=int, default=384)
    ap.add_argument("--order", default="stock,pin,pinread,stock")
    ap.add_argument("--reserve-mib", type=int, default=1300,
                    help="сколько доступной памяти оставить незакреплённой в режимах pin")
    a = ap.parse_args()

    drive = physical_drive_of(a.file)
    size = os.path.getsize(a.file)
    chunk = a.chunk_mib * MIB
    chunks = [(off, min(chunk, size - off)) for off in range(0, size, chunk)]
    base = map_file(a.file)
    print("fail %s: %.2f GiB, sloev %d po %d MiB, disk %s" % (a.file, size / GIB, len(chunks), a.chunk_mib, drive), flush=True)

    # ---- баллон: частная память под VirtualLock, «ОЗУ вне баллона» = --outside ----
    # Первая попытка подстраивала баллон под доступную память и провалилась: под давлением система
    # выдавила соседей в файл подкачки, файл целиком поместился в кэш, и дефицита не возникло.
    # Поэтому, как в bench_v2, задаётся объём ОЗУ вне баллона, а не доступная память.
    total = psutil.virtual_memory().total
    want = max(0, total - int(a.outside * GIB))
    a0 = avail_gib()
    print("do ballona dostupno %.2f GiB; vsego %.2f GiB; ballon %.2f GiB (vne ballona %.2f)"
          % (a0, total / GIB, want / GIB, a.outside), flush=True)
    STEP = 256 * MIB
    locked_balloon = 0
    ptrs = []
    fails = 0
    while locked_balloon < want and fails < 40:
        n = min(STEP, want - locked_balloon)
        if not set_ws(locked_balloon + n + GIB):
            print("BALLOON-FAIL rabochiy nabor", flush=True)
            return 1
        p = k32.VirtualAlloc(None, n, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)
        if not p or not k32.VirtualLock(p, n):
            fails += 1
            print("VirtualLock otkaz %d pri %.2f GiB" % (ctypes.get_last_error(), locked_balloon / GIB), flush=True)
            time.sleep(2)
            continue
        ctypes.memset(p, 1, n)
        ptrs.append(p)
        locked_balloon += n
    # стабилизация: доступная память перестала меняться
    prev = avail_gib()
    stable = 0
    t_end = time.time() + 90
    while stable < 5 and time.time() < t_end:
        time.sleep(2)
        cur = avail_gib()
        stable = stable + 1 if abs(cur - prev) < 0.15 else 0
        prev = cur
    av = avail_gib()
    wset = psutil.Process().memory_info().wset
    ok = locked_balloon >= want * 0.97 and wset >= locked_balloon * 0.97
    print("%s zablokirovano %.2f GiB iz %.2f, dostupno %.2f GiB, rabochiy nabor %.2f GiB"
          % ("BALLOON-OK" if ok else "BALLOON-FAIL", locked_balloon / GIB, want / GIB, av, wset / GIB), flush=True)
    if not ok:
        return 1

    buf_n = (chunk + 5) // 6 + PAGE
    bufs = [ctypes.create_string_buffer(buf_n) for _ in range(6)]
    pool = ThreadPoolExecutor(6)

    def touch(i):
        off, ln = chunks[i]
        part = (ln + 5) // 6
        jobs = []
        for t in range(6):
            s = t * part
            n = min(part, ln - s)
            if n > 0:
                jobs.append(pool.submit(ctypes.memmove, bufs[t], base + off + s, n))
        for j in jobs:
            j.result()

    def prefetch(i):
        off, ln = chunks[i]
        r = RANGE(base + off, ln)
        k32.PrefetchVirtualMemory(HPROC, 1, ctypes.byref(r), 0)

    def trim(i):
        off, ln = chunks[i]
        k32.VirtualUnlock(base + off, ln)  # не закреплён: ERROR_NOT_LOCKED, но страницы уходят из набора

    results = {}
    pinned = []
    ws_scan = [0.0]  # максимум рабочего набора под файл на последнем круге stock, ГиБ

    def pin_first(k):
        need = locked_balloon + k * chunk + 512 * MIB
        if not set_ws(need):
            return 0
        got = 0
        for i in range(k):
            off, ln = chunks[i]
            if k32.VirtualLock(base + off, ln):
                pinned.append(i)
                got += 1
            else:
                print("pin otkaz %d na sloe %d" % (ctypes.get_last_error(), i), flush=True)
                break
        return got

    def unpin_all():
        for i in pinned:
            off, ln = chunks[i]
            k32.VirtualUnlock(base + off, ln)
        pinned.clear()
        set_ws(locked_balloon + GIB)

    for vi, mode in enumerate(a.order.split(",")):
        tag = "%s#%d" % (mode, vi)
        if mode in ("pin", "pinread"):
            if not pinned:
                av = avail_gib()
                if a.pin_k >= 0:
                    k = a.pin_k
                else:
                    k = int((ws_scan[0] * GIB - a.reserve_mib * MIB) // chunk)
                k = max(0, min(k, len(chunks) - 2))
                print("PIN-PLAN k=%d: kesh po rabochemu naboru %.2f GiB, rezerv %d MiB" % (k, ws_scan[0], a.reserve_mib), flush=True)
                d0 = disk_read(drive)
                t0 = time.time()
                got = pin_first(k)
                print("PIN %s: zakrepleno %d sloev (%.2f GiB) za %.1f s, prochitano %.2f GiB; dostupno bylo %.2f, stalo %.2f"
                      % (tag, got, got * chunk / GIB, time.time() - t0, (disk_read(drive) - d0) / GIB, av, avail_gib()),
                      flush=True)
        elif pinned:
            unpin_all()

        streamed = [i for i in range(len(chunks)) if i not in set(pinned)]
        reader = None
        q = []
        qcv = threading.Condition()
        stop = [False]
        if mode == "pinread":
            def reader_loop():
                while True:
                    with qcv:
                        while not q and not stop[0]:
                            qcv.wait()
                        if stop[0] and not q:
                            return
                        j = q.pop(0)
                    prefetch(j)
            reader = threading.Thread(target=reader_loop, daemon=True)
            reader.start()
        pset = set(pinned)

        for c in range(a.cycles + 1):  # круг 0 — прогрев, в среднее не входит
            d0 = disk_read(drive)
            t0 = time.time()
            amin = avail_gib()
            wmax = 0
            per = []
            for i in range(len(chunks)):
                if reader is not None:
                    nxt = [(i + d) % len(chunks) for d in (1, 2)]
                    with qcv:
                        for j in nxt:
                            if j not in pset and j not in q:
                                q.append(j)
                        qcv.notify()
                di = disk_read(drive)
                touch(i)
                per.append((disk_read(drive) - di) / MIB)
                if reader is not None and i not in pset:
                    trim(i)
                amin = min(amin, avail_gib())
                wmax = max(wmax, psutil.Process().memory_info().wset)
            dt = time.time() - t0
            if mode == "stock" and c > 0:
                ws_scan[0] = (wmax - locked_balloon - 6 * buf_n) / GIB
            rd = (disk_read(drive) - d0) / GIB
            print("CYCLE %s krug %d: %.1f s, chtenie %.2f GiB (%.0f MiB/s), min dostupno %.2f GiB, zakrepleno %d sloev, rabochiy nabor %.2f GiB"
                  % (tag, c, dt, rd, rd * 1024 / dt if dt > 0 else 0, amin, len(pinned),
                     psutil.Process().memory_info().wset / GIB), flush=True)
            print("  po sloyam MiB: " + " ".join("%.0f" % x for x in per), flush=True)
            if c > 0:
                results.setdefault(tag, []).append((dt, rd))
        if reader is not None:
            with qcv:
                stop[0] = True
                qcv.notify()
            reader.join()

    unpin_all()
    cyc = size / GIB
    print("SUMMARY cikl %.2f GiB; ballon %.2f GiB; vne ballona %.2f GiB; kesh po rabochemu naboru %.2f GiB"
          % (cyc, locked_balloon / GIB, a.outside, ws_scan[0]), flush=True)
    for tag, rows in results.items():
        ts = [r[0] for r in rows]
        rs = [r[1] for r in rows]
        print("SUMMARY %-10s krugov %d: %.1f s na krug (razbros %.1f), chtenie %.2f GiB na krug (%.0f %% cikla)"
              % (tag, len(rows), statistics.mean(ts), (max(ts) - min(ts)), statistics.mean(rs),
                 100 * statistics.mean(rs) / cyc), flush=True)
    print("RUN-COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
