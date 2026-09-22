# -*- coding: utf-8 -*-
"""Сколько байтов экспертов MXFP4 можно снять сжатием без потерь — энтропия нулевого порядка.

Блок MXFP4 — 17 байт на 32 значения: байт общего масштаба E8M0 и 16 байт кодов e2m1 по 4 бита.
Энтропийный кодер (rANS) с таблицей на тензор приближается к энтропии нулевого порядка отдельно для
потока масштабов и потока полубайтов. Здесь она и считается — на экспертах gpt-oss-20b, по выборке
срезов из разных слоёв, — плюс для сверки lzma на тех же байтах.

Запуск: python mxfp4_entropy.py [модель.gguf]
"""
import lzma
import math
import sys

import numpy as np

sys.path.insert(0, r"H:\ftree\gguf-py")
from gguf import GGUFReader  # noqa: E402

PATH = sys.argv[1] if len(sys.argv) > 1 else r"<MODELS>\gpt-oss-20b-MXFP4.gguf"


def entropy(counts):
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def main():
    rd = GGUFReader(PATH, "r")
    exps = [t for t in rd.tensors if t.name.endswith("_exps.weight") and t.tensor_type.name == "MXFP4"]
    print(f"тензоров экспертов MXFP4: {len(exps)}")
    tot_bits = tot_bits_ent = 0.0
    sample_bytes = bytearray()
    rows = []
    for t in exps:
        il = int(t.name.split(".")[1])
        if il % 4 != 0:                                  # выборка: каждый четвёртый слой
            continue
        raw = np.asarray(t.data, dtype=np.uint8).reshape(-1)
        n_blk = raw.size // 17
        blk = raw[: n_blk * 17].reshape(n_blk, 17)
        scales = blk[:, 0]
        nib = np.concatenate([blk[:, 1:] & 0x0F, blk[:, 1:] >> 4], axis=None)
        h_s = entropy(np.bincount(scales, minlength=256))
        h_n = entropy(np.bincount(nib, minlength=16))
        bits = n_blk * 136
        bits_ent = n_blk * (h_s + 32 * h_n)
        tot_bits += bits
        tot_bits_ent += bits_ent
        if len(sample_bytes) < 64 * 2**20:
            sample_bytes += raw[: 8 * 2**20].tobytes()
        rows.append((t.name, h_s, h_n, bits_ent / bits))
    for name, h_s, h_n, r in rows:
        print(f"{name:36s} масштаб {h_s:4.2f} бит из 8, код {h_n:5.3f} бит из 4, r = {r:.4f}")
    print(f"\nитого по выборке: r = {tot_bits_ent / tot_bits:.4f} (снимается {100 * (1 - tot_bits_ent / tot_bits):.1f} %)")
    comp = lzma.compress(bytes(sample_bytes), preset=6)
    print(f"lzma-6 на {len(sample_bytes) / 2**20:.0f} МиБ сырых срезов: r = {len(comp) / len(sample_bytes):.4f}")
    print("RUN-COMPLETE")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
