# -*- coding: utf-8 -*-
r"""q8_entropy.py — сколько на самом деле можно снять с Q8_0 архиваторными средствами.

ЗАЧЕМ. Прежний вердикт «веса несжимаемы» снят на MXFP4 (коэффициент 0.919). Модель
пользователя — Q8_0, формат другой: блок это 32 целых восьмибитных значения плюс масштаб
FP16. Утверждение «сжимать нечего» надо не переносить, а мерить, и мерить РАЗДЕЛЬНО:
масштабы и кванты имеют разную природу.

ЧТО СЧИТАЕТ, для каждого Q8_0-тензора и суммарно:
  * энтропия квантов (бит на значение при 8 возможных);
  * коэффициент zlib и lzma на потоке квантов;
  * то же на потоке масштабов, отдельно и с дельта-фильтром (приём архиваторов: перед
    энтропийным кодером ставится обратимое преобразование);
  * доля повторяющихся блоков по 4 КиБ — проверка «длинных совпадений» (rzip/lrzip/zstd --long);
  * что даёт сжатие файла целиком, для сверки.

Запуск: python q8_entropy.py <model.gguf> [предел_МиБ]
"""
from __future__ import annotations

import collections
import hashlib
import lzma
import math
import os
import struct
import sys
import zlib

import numpy as np

GGUF_MAGIC = b"GGUF"
# коды типов ggml, нужные здесь
T_F32, T_F16, T_Q8_0 = 0, 1, 8
BLOCK_Q8_0 = 34          # 2 байта масштаба fp16 + 32 int8
QK8_0 = 32


def _read(f, fmt):
    sz = struct.calcsize(fmt)
    return struct.unpack(fmt, f.read(sz))


def _read_str(f):
    (n,) = _read(f, "<Q")
    return f.read(n).decode("utf-8", "replace")


def _skip_value(f, vtype):
    simple = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if vtype in simple:
        f.read(simple[vtype])
    elif vtype == 8:                       # строка
        _read_str(f)
    elif vtype == 9:                       # массив
        (et,) = _read(f, "<I")
        (cnt,) = _read(f, "<Q")
        for _ in range(cnt):
            _skip_value(f, et)
    else:
        raise ValueError("неизвестный тип значения %d" % vtype)


def parse_gguf(path):
    """Минимальный разбор GGUF: возвращает (список тензоров, смещение данных, выравнивание)."""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != GGUF_MAGIC:
            raise ValueError("не GGUF: %r" % magic)
        ver, n_tensors, n_kv = _read(f, "<IQQ")
        alignment = 32
        for _ in range(n_kv):
            key = _read_str(f)
            (vtype,) = _read(f, "<I")
            if key.endswith("general.alignment") and vtype in (4, 5):
                (alignment,) = _read(f, "<I")
                continue
            _skip_value(f, vtype)
        tensors = []
        for _ in range(n_tensors):
            name = _read_str(f)
            (nd,) = _read(f, "<I")
            dims = [_read(f, "<Q")[0] for _ in range(nd)]
            (ttype,) = _read(f, "<I")
            (offset,) = _read(f, "<Q")
            tensors.append({"name": name, "dims": dims, "type": ttype, "offset": offset})
        data_start = f.tell()
    pad = (alignment - (data_start % alignment)) % alignment
    return tensors, data_start + pad, alignment


def q8_nbytes(dims):
    n = 1
    for d in dims:
        n *= d
    return (n // QK8_0) * BLOCK_Q8_0


def ratio(buf, fn):
    if not len(buf):
        return float("nan")
    return len(fn(buf)) / len(buf)


def entropy_bits(arr: np.ndarray) -> float:
    cnt = np.bincount(arr.view(np.uint8).ravel(), minlength=256).astype(np.float64)
    p = cnt[cnt > 0] / cnt.sum()
    return float(-(p * np.log2(p)).sum())


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    path = argv[1]
    limit_mib = int(argv[2]) if len(argv) > 2 else 512
    tensors, data_start, align = parse_gguf(path)
    q8 = [t for t in tensors if t["type"] == T_Q8_0]
    print("файл            %s (%.2f ГБ)" % (os.path.basename(path), os.path.getsize(path) / 1e9))
    print("тензоров        %d, из них Q8_0 %d" % (len(tensors), len(q8)))
    if not q8:
        print("Q8_0-тензоров нет — мерить нечего")
        return 1

    quants = bytearray()
    scales = bytearray()
    budget = limit_mib * 1024 * 1024
    taken = 0
    with open(path, "rb") as f:
        for t in q8:
            nb = q8_nbytes(t["dims"])
            if taken >= budget:
                break
            nb = min(nb, budget - taken)
            nb -= nb % BLOCK_Q8_0
            if nb <= 0:
                continue
            f.seek(data_start + t["offset"])
            raw = np.frombuffer(f.read(nb), dtype=np.uint8).reshape(-1, BLOCK_Q8_0)
            scales += raw[:, :2].tobytes()
            quants += raw[:, 2:].tobytes()
            taken += nb

    qa = np.frombuffer(bytes(quants), dtype=np.uint8)
    sa = np.frombuffer(bytes(scales), dtype=np.uint8)
    print("взято           %.1f МиБ блоков Q8_0 (%d блоков)" % (taken / 2**20, len(qa) // QK8_0))
    print()
    print("-- кванты (32 байта из 34, %.1f %% объёма) --" % (100 * 32 / 34))
    print("  энтропия          %.3f бит из 8  -> предел сжатия %.3f" %
          (entropy_bits(qa), entropy_bits(qa) / 8))
    print("  zlib -9           %.4f" % ratio(bytes(quants), lambda b: zlib.compress(b, 9)))
    print("  lzma              %.4f" % ratio(bytes(quants), lambda b: lzma.compress(b, preset=6)))
    print()
    print("-- масштабы fp16 (2 байта из 34, %.1f %% объёма) --" % (100 * 2 / 34))
    print("  энтропия          %.3f бит из 8" % entropy_bits(sa))
    print("  zlib -9           %.4f" % ratio(bytes(scales), lambda b: zlib.compress(b, 9)))
    print("  lzma              %.4f" % ratio(bytes(scales), lambda b: lzma.compress(b, preset=6)))
    s16 = np.frombuffer(bytes(scales), dtype=np.uint16)
    delta = np.ascontiguousarray(np.diff(s16.astype(np.int32)).astype(np.int16)).tobytes()
    print("  lzma + дельта     %.4f   <- приём архиваторов: фильтр перед кодером" %
          ratio(delta, lambda b: lzma.compress(b, preset=6)))
    print()
    both = bytes(quants) + bytes(scales)
    r_q = ratio(bytes(quants), lambda b: lzma.compress(b, preset=6))
    r_s = ratio(delta, lambda b: lzma.compress(b, preset=6))
    total = (32 * r_q + 2 * r_s) / 34
    print("-- итог по блоку --")
    print("  раздельно, лучшие коэффициенты: (32*%.4f + 2*%.4f)/34 = %.4f" % (r_q, r_s, total))
    print("  экономия на весах              %.2f %%" % (100 * (1 - total)))
    print()
    print("-- длинные совпадения (rzip / zstd --long / дедупликация) --")
    blk = 4096
    h = collections.Counter()
    mv = memoryview(both)
    for i in range(0, len(mv) - blk, blk):
        h[hashlib.blake2b(mv[i:i + blk], digest_size=8).digest()] += 1
    tot = sum(h.values())
    dup = tot - len(h)
    print("  блоков по 4 КиБ   %d, повторов %d (%.3f %%)" % (tot, dup, 100 * dup / tot if tot else 0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
