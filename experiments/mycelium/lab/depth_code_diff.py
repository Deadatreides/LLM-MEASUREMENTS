"""
lab/depth_code_diff.py — что именно меняет депт-луп.

Вопрос: депт-луп переписывает ИЗВЛЕЧЁННЫЙ код или только обрамление вокруг
него (преамбулу, комментарии, объяснение)? Если только обрамление, то
q_env не может измениться в принципе — она считается по коду, — и наблюдаемое
«ответ изменён 21/30, Δq_env = 0.000» объясняется тривиально.

Сравнение идёт ровно тем извлекателем, который использует scorer:
QualityScorer._extract_python_code (все ```-блоки, склеенные через \\n\\n).

    python lab/depth_code_diff.py --tag full
"""
import argparse
import difflib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import yaml                                   # noqa: E402
from core.scorer import QualityScorer         # noqa: E402


def norm(code: str) -> str:
    """Убрать различия, невидимые для интерпретатора: пустые строки, хвостовые
    пробелы, комментарии целой строкой."""
    out = []
    for line in code.replace('\r', '').split('\n'):
        s = line.rstrip()
        if not s.strip():
            continue
        if s.strip().startswith('#'):
            continue
        out.append(s)
    return '\n'.join(out)


def main(tag, condition, show):
    with open(os.path.join(os.path.dirname(HERE), 'config', 'settings.yaml'),
              encoding='utf-8') as f:
        sc = QualityScorer(yaml.safe_load(f))

    rows = [json.loads(l) for l in
            open(os.path.join(HERE, 'data', f'results_{tag}.jsonl'),
                 encoding='utf-8') if l.strip()]
    rows = [r for r in rows if r.get('status') == 'ok'
            and r.get('condition') == condition and r.get('syn_texts')]
    if not rows:
        print(f"нет подходящих записей для условия {condition}")
        return

    stats = {'total': 0, 'depth_ran': 0, 'text_changed': 0,
             'code_changed': 0, 'code_same_text_changed': 0,
             'code_empty_before': 0, 'code_empty_after': 0,
             'q_changed': 0}
    examples = []

    for r in rows:
        stats['total'] += 1
        d = r.get('depth') or {}
        if not d.get('started') or not d.get('steps'):
            continue
        stats['depth_ran'] += 1

        before_full = r['syn_texts'][0]
        after_full = r['answer']
        text_changed = before_full.strip() != after_full.strip()

        cb = norm(sc._extract_python_code(before_full))
        ca = norm(sc._extract_python_code(after_full))
        code_changed = cb != ca

        if text_changed:
            stats['text_changed'] += 1
        if code_changed:
            stats['code_changed'] += 1
        if text_changed and not code_changed:
            stats['code_same_text_changed'] += 1
        if not cb:
            stats['code_empty_before'] += 1
        if not ca:
            stats['code_empty_after'] += 1
        if abs(r.get('q_env_final', 0) - r.get('q_env_pre_depth', 0)) > 1e-9:
            stats['q_changed'] += 1

        if code_changed and len(examples) < show:
            diff = list(difflib.unified_diff(
                cb.split('\n'), ca.split('\n'),
                fromfile=f"task{r['task_id']} до депта",
                tofile=f"task{r['task_id']} после депта", lineterm=''))
            examples.append((r['task_id'],
                             r.get('q_env_pre_depth'), r.get('q_env_final'),
                             diff[:40]))

    n = stats['depth_ran']
    print(f"условие {condition}, записей {stats['total']}, "
          f"депт-луп сделал ≥1 шаг в {n}")
    if not n:
        return
    print()
    print(f"  текст ответа изменился            {stats['text_changed']:>3}/{n}")
    print(f"  извлечённый КОД изменился         {stats['code_changed']:>3}/{n}")
    print(f"  текст изменился, а код — нет      "
          f"{stats['code_same_text_changed']:>3}/{n}   ← правка обрамления")
    print(f"  q_env изменилась                  {stats['q_changed']:>3}/{n}")
    print(f"  код не извлёкся до депта          {stats['code_empty_before']:>3}/{n}")
    print(f"  код не извлёкся после депта       {stats['code_empty_after']:>3}/{n}")

    for tid, qb, qa, diff in examples:
        print(f"\n--- task {tid}: q_env {qb} → {qa} ---")
        for line in diff:
            print('   ' + line)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='full')
    ap.add_argument('--condition', default='M')
    ap.add_argument('--show', type=int, default=3)
    a = ap.parse_args()
    main(a.tag, a.condition, a.show)
