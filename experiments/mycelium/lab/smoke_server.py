"""
lab/smoke_server.py — быстрая проверка сервера: все модели отвечают, LRU не
ломает выдачу, видно цену перезагрузки.

Гоняет пул по кругу N раундов. Если MAX_RESIDENT меньше числа моделей, часть
будет вытесняться и подгружаться заново — здесь и видно, сколько это стоит.

    python lab/smoke_server.py --rounds 2
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

SERVER = 'http://127.0.0.1:8077/v1'
Q = "Write a python function is_prime(n). Reply with a single ```python block only."


def get(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def chat(model, prompt, max_tokens=128, timeout=600):
    body = json.dumps({
        'model': model,
        'messages': [{'role': 'system', 'content': 'You are a Generator.'},
                     {'role': 'user', 'content': prompt}],
        'temperature': 0.7, 'max_tokens': max_tokens}).encode('utf-8')
    req = urllib.request.Request(SERVER + '/chat/completions', data=body,
                                 headers={'Content-Type': 'application/json'})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return d, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rounds', type=int, default=2)
    ap.add_argument('--models', default='')
    args = ap.parse_args()

    try:
        models = [m['id'] for m in get(SERVER + '/models')['data']]
    except urllib.error.URLError as exc:
        print(f"сервер недоступен: {exc}")
        return 2
    if args.models:
        models = [m.strip() for m in args.models.split(',') if m.strip()]

    print(f"моделей: {len(models)}, раундов: {args.rounds}\n")
    bad = []
    for rnd in range(args.rounds):
        for m in models:
            try:
                d, dt = chat(m, Q)
            except Exception as exc:
                print(f"r{rnd} {m:<36} ОШИБКА {exc}")
                bad.append((m, str(exc)))
                continue
            txt = d['choices'][0]['message']['content'] or ''
            ok = 'def ' in txt
            if not ok:
                bad.append((m, 'нет def в ответе'))
            print(f"r{rnd} {m:<36} {dt:>6.1f}s  "
                  f"tok={d['usage']['completion_tokens']:>3}  "
                  f"def={'да' if ok else 'НЕТ'}"
                  + ('' if ok else f"  <<< {txt.strip()[:60]!r}"))
        print()

    s = get(SERVER + '/stats')
    print(f"вытеснений: {s['evictions']}  загрузок: {s['reloads']}  "
          f"VRAM: {s['vram_gib']} GiB  резидентно: {s['loaded']}")
    if s['gen_seconds']:
        print(f"средняя скорость: "
              f"{s['completion_tokens']/s['gen_seconds']:.0f} ток/с "
              f"по {s['calls']} вызовам")
    if bad:
        print(f"\nПРОБЛЕМЫ ({len(bad)}):")
        for m, why in bad:
            print(f"  {m}: {why}")
        return 1
    print("\nвсе модели отвечают кодом")
    return 0


if __name__ == '__main__':
    sys.exit(main())
