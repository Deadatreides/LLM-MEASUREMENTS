"""lab/build_corpus.py — 30 задач MBPP, случайные из первых 200, список заморожен.

Пишет lab/data/corpus30.json:
  task_id, prompt (текст + эталонные тесты, канонический MBPP-промпт),
  tests (готовый pytest-файл), reference (эталонное решение MBPP).
"""
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
FULL = os.path.join(HERE, 'data', 'mbpp_full.jsonl')
OUT = os.path.join(HERE, 'data', 'corpus30.json')

SEED = 20260808
N = 30
POOL_MAX_TASK_ID = 200


def make_test_file(row: dict) -> str:
    lines = [
        "import sys, os",
        "sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))",
        "from solution import *",
        "",
    ]
    setup = (row.get('test_setup_code') or '').strip()
    if setup:
        lines += [setup, ""]
    for i, assertion in enumerate(row['test_list']):
        lines.append(f"def test_{i}():")
        lines.append("    " + assertion.strip().replace('\r', ''))
        lines.append("")
    return '\n'.join(lines)


def make_prompt(row: dict) -> str:
    tests = '\n'.join(t.strip() for t in row['test_list'])
    return (
        f"{row['text'].strip()}\n\n"
        f"Your code should pass these tests:\n{tests}\n\n"
        "Return a single Python code block with the complete solution."
    )


def main():
    rows = [json.loads(l) for l in open(FULL, encoding='utf-8') if l.strip()]
    pool = [r for r in rows if r['task_id'] <= POOL_MAX_TASK_ID]
    rnd = random.Random(SEED)
    picked = sorted(rnd.sample(pool, N), key=lambda r: r['task_id'])

    corpus = []
    for r in picked:
        corpus.append({
            'task_id': r['task_id'],
            'text': r['text'].strip(),
            'prompt': make_prompt(r),
            'tests': make_test_file(r),
            'reference': r['code'],
            'test_list': r['test_list'],
        })

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump({'seed': SEED, 'pool_max_task_id': POOL_MAX_TASK_ID,
                   'n': N, 'source': 'mbpp.jsonl (google-research)',
                   'tasks': corpus}, f, ensure_ascii=False, indent=1)
    print(f"wrote {OUT}")
    print("task_ids:", [c['task_id'] for c in corpus])


if __name__ == '__main__':
    main()
