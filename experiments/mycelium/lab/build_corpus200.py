"""lab/build_corpus200.py — корпус на 200 задач MBPP как НАДМНОЖЕСТВО прежних 30.

Зачем 200, а не 30. Механизмы Ш2 — медленные контуры по построению, это прямо
записано в спецификации: eta_DPO : eta_bandit : eta_PID = 1 : 20 : 200.
Замерено на этом коде:
  * гиперциклу нужно ~1000 шагов до равновесия (при мутации 0.005 равновесие
    |dx| = 0.0149, на 30-м шаге видно 0.0018);
  * PID при |lambda_L| = 0.6 требует 4 срабатываний, при 0.3 — около 15;
  * пер-ролевая mu на 30 шагах получает 4-7 наблюдений на ячейку при
    ema_alpha = 0.047, то есть окно EMA в 21 наблюдение не заполняется.
На 30 задачах всё это измеряет переходный процесс, а не поведение.

Плюс шум: BoN-G на одинаковой конфигурации давал 36.7 / 36.7 / 46.7 / 36.7 %
solved. При n=30 разрыв M - BoN-G (SEM 0.05-0.075) не разрешается.

Почему надмножество. Первые 30 задач — те же самые, с тем же порядком, что в
corpus30.json. Значит все прежние прогоны (base_30, sh1_30, role2_30, sh2_30)
остаются сравнимыми на общей подвыборке, и «стало лучше» можно проверить и на
200 задачах, и на исходных 30.

Пул расширен с task_id <= 200 до диапазона тест-сплита MBPP (11..510):
из 200 первых нельзя набрать 200 задач, а выходить в train-сплит незачем.

    python lab/build_corpus200.py
"""
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
FULL = os.path.join(HERE, 'data', 'mbpp_full.jsonl')
BASE30 = os.path.join(HERE, 'data', 'corpus30.json')
OUT = os.path.join(HERE, 'data', 'corpus200.json')

SEED = 20260810
N = 200
POOL_LO, POOL_HI = 11, 510          # тест-сплит MBPP


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


def entry(row: dict) -> dict:
    return {
        'task_id': row['task_id'],
        'text': row['text'].strip(),
        'prompt': make_prompt(row),
        'tests': make_test_file(row),
        'reference': row['code'],
        'test_list': row['test_list'],
    }


def main():
    rows = {r['task_id']: r for r in
            (json.loads(l) for l in open(FULL, encoding='utf-8') if l.strip())}

    with open(BASE30, encoding='utf-8') as f:
        base = json.load(f)
    head = base['tasks']                       # порядок сохраняем как есть
    head_ids = {t['task_id'] for t in head}

    pool = [tid for tid in sorted(rows)
            if POOL_LO <= tid <= POOL_HI and tid not in head_ids]
    need = N - len(head)
    if need > len(pool):
        raise SystemExit(f"в пуле {len(pool)} задач, нужно {need}")
    rnd = random.Random(SEED)
    extra_ids = sorted(rnd.sample(pool, need))
    tasks = head + [entry(rows[tid]) for tid in extra_ids]

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump({'seed': SEED, 'n': len(tasks),
                   'source': 'mbpp.jsonl (google-research), тест-сплит 11..510',
                   'superset_of': 'corpus30.json (первые 30 задач, тот же порядок)',
                   'pool_range': [POOL_LO, POOL_HI],
                   'tasks': tasks}, f, ensure_ascii=False, indent=1)

    print(f"записано {OUT}: {len(tasks)} задач")
    print(f"первые 30 совпадают с corpus30: "
          f"{[t['task_id'] for t in tasks[:30]] == [t['task_id'] for t in head]}")
    print(f"добавлено {len(extra_ids)}, диапазон task_id "
          f"{min(t['task_id'] for t in tasks)}..{max(t['task_id'] for t in tasks)}")
    n_tests = [len(t['test_list']) for t in tasks]
    print(f"тестов на задачу: min {min(n_tests)} медиана "
          f"{sorted(n_tests)[len(n_tests)//2]} max {max(n_tests)}")


if __name__ == '__main__':
    main()
