# -*- coding: utf-8 -*-
"""Этап 0: окна корпуса модели для ворот качества и порции беседы для замера скорости.

Корпус gpt-oss-own.plain.txt — 40 ответов самой модели, каждый начинается строкой своего запроса
из corpus-prompts.txt. Окна берутся по разным запросам, чтобы быть независимыми (как у myc-autotune):
  набор  — ответы 1–20 (для рычагов, которым нужен выученный набор);
  окно A — ответы 21–30, окно B — ответы 31–40 (цена).
Порции беседы: весь корпус подряд, нарезанный по ~4600 символов (около 1 тыс. токенов).

Пишет в H:/tools/smoke/phase0/.
"""
import io
import os
import sys

SRC = r"H:\tools\smoke\gpt-oss-own.plain.txt"
PROMPTS = r"H:\tools\smoke\corpus-prompts.txt"
OUT = r"H:\tools\smoke\phase0"


def main():
    text = io.open(SRC, encoding="utf-8").read()
    prompts = [p.strip() for p in io.open(PROMPTS, encoding="utf-8") if p.strip()]
    pos = []
    cursor = 0
    for p in prompts:
        i = text.find(p, cursor)
        if i < 0:
            raise SystemExit(f"запрос не найден в корпусе: {p[:60]}")
        pos.append(i)
        cursor = i + len(p)
    pos.append(len(text))
    answers = [text[pos[k]:pos[k + 1]].strip() for k in range(len(prompts))]
    os.makedirs(OUT, exist_ok=True)

    def write(name, parts):
        body = "\n\n".join(parts) + "\n"
        io.open(os.path.join(OUT, name), "w", encoding="utf-8", newline="\n").write(body)
        print(f"{name}: ответов {len(parts)}, символов {len(body)}")

    write("set_01_20.txt", answers[0:20])
    write("win_a_21_30.txt", answers[20:30])
    write("win_b_31_40.txt", answers[30:40])

    step = 4600
    chunks = [text[i:i + step] for i in range(0, len(text), step)]
    for k, ch in enumerate(chunks[:5]):
        io.open(os.path.join(OUT, f"chat_turn_{k + 1}.txt"), "w", encoding="utf-8", newline="\n").write(ch)
    print(f"порций беседы: 5 из {len(chunks)} по {step} символов")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
