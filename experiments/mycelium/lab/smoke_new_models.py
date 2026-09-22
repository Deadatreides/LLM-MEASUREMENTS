"""lab/smoke_new_models.py — MODELS_ONBOARDING §5: смоук перед основным прогоном.

Три задачи из corpus30, по одному вызову на модель. Печатает ровно то, что
§5 предписывает смотреть ГЛАЗАМИ:

  1. сырой промпт после применения chat-шаблона — инструкция роли G на месте?
     спецтокены не задвоены? у Gemma инструкция внутри пользовательского хода?
  2. сырой ответ — блок кода или продолжение инструкции / второй ход диалога?
  3. метаданные GGUF — tokenizer.chat_template присутствует и применяется.

Работает на CPU (n_gpu_layers=0) и не поднимает llm_server: для проверки
шаблона и стоп-токенов ускоритель не нужен, а VRAM в это время может быть
занята. Стоп-поведение от устройства не зависит.

Запуск:
    python lab/smoke_new_models.py
    python lab/smoke_new_models.py --n-tasks 3 --max-tokens 512
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS_DIR = os.path.join(os.path.dirname(ROOT), 'models')
sys.path.insert(0, ROOT)

# Тот же системный промпт, что породил lab/data/baseline_pool.json
# (lab/bench_pool.py). Менять нельзя: иначе baseline несравним (§5.2).
SYSTEM = ("You are a Generator. Solve the task in Python. "
          "Reply with a single ```python code block and nothing else.")

CANDIDATES = [
    ('gemma-3-it-1B-Q5_K_S', 'gemma-3-it-1B-Q5_K_S/gemma-3-it-1B-Q5_K_S.gguf'),
    ('Llama-3.2-1B-Instruct-Q4_0',
     'Llama-3.2-1B-Instruct-Q4_0/Llama-3.2-1B-Instruct-Q4_0.gguf'),
]

BAD_SIGNS = [
    ('повтор системной инструкции', SYSTEM[:40]),
    ('второй ход диалога (user)', '<start_of_turn>user'),
    ('второй ход диалога (header)', '<|start_header_id|>'),
    ('пересказ задачи', 'Your code should pass these tests'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-tasks', type=int, default=3)
    ap.add_argument('--max-tokens', type=int, default=512)
    ap.add_argument('--temperature', type=float, default=0.0)
    a = ap.parse_args()

    from lab.gguf_boot import ensure_llama_cpp
    ensure_llama_cpp()
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Jinja2ChatFormatter

    tasks = json.load(open(os.path.join(HERE, 'data', 'corpus30.json'),
                           encoding='utf-8'))['tasks'][:a.n_tasks]
    report = []

    for name, rel in CANDIDATES:
        path = os.path.join(MODELS_DIR, rel)
        key = os.path.splitext(os.path.basename(path))[0].lower()
        print('=' * 78)
        print(f"МОДЕЛЬ {name}")
        print(f"  ключ, под которым её увидит llm_server: {key!r}")
        print(f"  (llm_server.discover: splitext(basename).lower(), "
              f"строка 135 — не имя каталога и не имя из карточки)")
        if not os.path.exists(path):
            print(f"  ФАЙЛА НЕТ: {path}")
            continue

        m = Llama(model_path=path, n_gpu_layers=0, n_ctx=2048, verbose=False)
        md = m.metadata
        tmpl = md.get('tokenizer.chat_template', '') or ''
        bos_id = int(md.get('tokenizer.ggml.bos_token_id', -1))
        eos_id = int(md.get('tokenizer.ggml.eos_token_id', -1))

        print(f"\n  [3] МЕТАДАННЫЕ")
        print(f"      general.name         : {md.get('general.name')}")
        print(f"      general.architecture : {md.get('general.architecture')}")
        print(f"      chat_template        : "
              f"{'ЕСТЬ' if tmpl.strip() else 'НЕТ — модель повторит промпт'}"
              f" ({len(tmpl)} символов)")
        print(f"      bos_token_id={bos_id}  eos_token_id={eos_id}  "
              f"add_bos_token={md.get('tokenizer.ggml.add_bos_token')}")

        bos = m.detokenize([bos_id]).decode('utf-8', 'replace') if bos_id >= 0 else ''
        eos = m.detokenize([eos_id]).decode('utf-8', 'replace') if eos_id >= 0 else ''
        fmt = Jinja2ChatFormatter(template=tmpl, bos_token=bos, eos_token=eos)

        msgs = [{'role': 'system', 'content': SYSTEM},
                {'role': 'user', 'content': tasks[0]['prompt']}]
        rendered = fmt(messages=msgs).prompt

        # BOS: шаблон подставляет bos_token как строку, llama.cpp добавляет
        # свой при токенизации. Двойной BOS = два одинаковых id подряд в начале.
        toks = m.tokenize(rendered.encode('utf-8'), add_bos=True, special=True)
        double_bos = len(toks) > 1 and toks[0] == bos_id and toks[1] == bos_id

        print(f"\n  [1] СЫРОЙ ПРОМПТ ПОСЛЕ ШАБЛОНА (первые 700 символов)")
        print('      ' + rendered[:700].replace('\n', '\n      '))
        print(f"\n      системная инструкция видна в промпте: "
              f"{'ДА' if SYSTEM[:40] in rendered else 'НЕТ — ОСТАНОВИТЬСЯ'}")
        if md.get('general.architecture') == 'gemma3':
            inside = ('<start_of_turn>user' in rendered
                      and rendered.find('<start_of_turn>user')
                      < rendered.find(SYSTEM[:40]))
            print(f"      Gemma: инструкция внутри пользовательского хода: "
                  f"{'ДА' if inside else 'НЕТ — ОСТАНОВИТЬСЯ'}")
        print(f"      первые 4 токена: {toks[:4]}   двойной BOS: "
              f"{'ДА — ОСТАНОВИТЬСЯ' if double_bos else 'нет'}")

        print(f"\n  [2] СЫРЫЕ ОТВЕТЫ на {len(tasks)} задачи")
        rows = []
        for t in tasks:
            msgs = [{'role': 'system', 'content': SYSTEM},
                    {'role': 'user', 'content': t['prompt']}]
            t0 = time.time()
            out = m.create_chat_completion(messages=msgs,
                                           max_tokens=a.max_tokens,
                                           temperature=a.temperature)
            dt = time.time() - t0
            raw = out['choices'][0]['message']['content'] or ''
            fin = out['choices'][0].get('finish_reason')
            n_out = out['usage']['completion_tokens']
            hits = [label for label, marker in BAD_SIGNS if marker in raw]
            print(f"\n    ── задача {t['task_id']}  "
                  f"({n_out} ток, {dt:.0f}s, finish={fin}) ──")
            print('    ' + raw[:600].replace('\n', '\n    '))
            print(f"    ┗ код-фенс: {'есть' if '```' in raw else 'НЕТ'}  |  "
                  f"признаки непримененного шаблона: "
                  f"{hits if hits else 'нет'}")
            if fin == 'length':
                print("    ┗ ВНИМАНИЕ finish_reason=length: упёрлось в лимит, "
                      "а не в стоп-токен.")
                print("      Если так на всех трёх — стоп-токен не объявлен, "
                      "см. ONBOARDING §3/§4.")
            rows.append({'task_id': t['task_id'], 'finish': fin,
                         'tokens': n_out, 'has_fence': '```' in raw,
                         'bad_signs': hits, 'seconds': round(dt, 1)})

        stop_ok = sum(1 for r in rows if r['finish'] == 'stop')
        report.append({
            'model': name, 'server_key': key,
            'has_template': bool(tmpl.strip()),
            'bos_id': bos_id, 'eos_id': eos_id, 'double_bos': double_bos,
            'system_visible': SYSTEM[:40] in rendered,
            'stopped_on_stop_token': f'{stop_ok}/{len(rows)}',
            'tasks': rows,
        })
        del m

    path = os.path.join(HERE, 'data', 'smoke_new_models.json')
    json.dump(report, open(path, 'w', encoding='utf-8'), ensure_ascii=False,
              indent=1)
    print('\n' + '=' * 78)
    print(f"сохранено: {path}")
    print("Чек-лист ONBOARDING: пока все три пункта §5 не пройдены глазами "
          "на обеих моделях, основной прогон не запускать.")
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
