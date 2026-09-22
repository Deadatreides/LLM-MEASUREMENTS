# -*- coding: utf-8 -*-
"""Баллон ОЗУ с ЗАБЛОКИРОВАННЫМИ страницами.

ЗАЧЕМ ИМЕННО ТАКОЙ. Прежний баллон занимал память обычным bytearray и периодически её трогал.
Под давлением этого мало: Windows выдавливает его в файл подкачки, освобождает физическую
память под файловый кэш, и модель постепенно кэшируется. Признак виден прямо в замере — чтение
с диска затухает по кругам: 12.17, 4.66, 1.61 ГиБ. Режим уплывает во время самого замера, и
сравнение начинает мерить не схему, а то, насколько успел прогреться кэш.

Здесь страницы фиксируются VirtualLock, то есть выгрузить их система не может. Рабочий набор
процесса предварительно расширяется SetProcessWorkingSetSize — иначе VirtualLock упрётся в
лимит и вернёт отказ.

Проверка встроена: после блокировки печатается доступная память. Если она не упала, баллон
мнимый и об этом надо знать сразу, а не после замера.
"""
import ctypes
import ctypes.wintypes as wt
import sys
import time

k32 = ctypes.WinDLL("kernel32", use_last_error=True)

MEM_COMMIT  = 0x1000
MEM_RESERVE = 0x2000
MEM_RELEASE = 0x8000
PAGE_READWRITE = 0x04

k32.VirtualAlloc.restype  = ctypes.c_void_p
k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wt.DWORD, wt.DWORD]
k32.VirtualLock.restype   = wt.BOOL
k32.VirtualLock.argtypes  = [ctypes.c_void_p, ctypes.c_size_t]
k32.SetProcessWorkingSetSize.restype  = wt.BOOL
k32.SetProcessWorkingSetSize.argtypes = [wt.HANDLE, ctypes.c_size_t, ctypes.c_size_t]
k32.GetCurrentProcess.restype = wt.HANDLE


def avail_mib():
    class MEMSTAT(ctypes.Structure):
        _fields_ = [("dwLength", wt.DWORD), ("dwMemoryLoad", wt.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MEMSTAT()
    m.dwLength = ctypes.sizeof(MEMSTAT)
    k32.GlobalMemoryStatusEx(ctypes.byref(m))
    return int(m.ullAvailPhys // (1024 * 1024))


def main():
    gib = float(sys.argv[1]) if len(sys.argv) > 1 else 13.5
    n = int(gib * (1 << 30))
    print("do ballona dostupno: %d MiB" % avail_mib(), flush=True)

    # Рабочий набор надо расширить ДО блокировки, с запасом.
    #
    # С ПОВТОРАМИ, и это не перестраховка. Отказ 1450 (ERROR_NO_SYSTEM_RESOURCES) наступает,
    # когда память предыдущего баллона ещё не возвращена системе. Один раз он уже стоил замера:
    # 9 сентября баллон отчитался «zablokirovano 0.00 GiB», прогон продолжился, и половина
    # результатов оказалась снята вне целевого режима.
    extra = n + (1 << 30)
    ws_ok = False
    for i in range(10):
        if k32.SetProcessWorkingSetSize(k32.GetCurrentProcess(), extra, extra + (1 << 30)):
            ws_ok = True
            break
        print("SetProcessWorkingSetSize otkaz: %d (popytka %d, dostupno %d MiB)"
              % (ctypes.get_last_error(), i + 1, avail_mib()), flush=True)
        time.sleep(4)
    if not ws_ok:
        print("BALLOON-FAIL rabochiy nabor ne rasshiren", flush=True)
        return 1

    p = k32.VirtualAlloc(None, n, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)
    if not p:
        print("VirtualAlloc otkaz: %d" % ctypes.get_last_error(), flush=True)
        print("BALLOON-FAIL", flush=True)
        return 1

    # блокировать кусками: одна огромная просьба отвергается чаще, чем много мелких
    CHUNK = 64 << 20
    locked = 0
    for off in range(0, n, CHUNK):
        ln = min(CHUNK, n - off)
        if k32.VirtualLock(ctypes.c_void_p(p + off), ln):
            locked += ln
        else:
            # не удалось — всё равно касаемся, чтобы страница была физической
            ctypes.memset(ctypes.c_void_p(p + off), 1, ln)

    ctypes.memset(ctypes.c_void_p(p), 1, n)
    a = avail_mib()
    print("zablokirovano %.2f GiB iz %.2f; posle ballona dostupno: %d MiB"
          % (locked / (1 << 30), gib, a), flush=True)

    # Приговор в одной машиночитаемой строке: обёртка обязана отказаться считать без неё.
    if locked < n * 0.9:
        print("BALLOON-FAIL zablokirovano %.2f iz %.2f GiB" % (locked / (1 << 30), gib), flush=True)
        k32.VirtualFree(ctypes.c_void_p(p), 0, MEM_RELEASE)
        return 1
    print("BALLOON-OK %.2f %d" % (locked / (1 << 30), a), flush=True)

    while True:
        time.sleep(5)


if __name__ == "__main__":
    sys.exit(main())
