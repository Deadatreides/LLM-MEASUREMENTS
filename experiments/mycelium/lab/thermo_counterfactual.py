"""
lab/thermo_counterfactual.py — почему Φ ≡ 0 (диагностика §7.1).

Гипотеза, вычитанная из кода:
  _run_depth_loop() ПЕРВЫМ делом апсертит в блок step-N все черновики,
  включая final_answer (orchestrator.py:829-833), и делает это ДО того,
  как run_step вызовет compute_thermodynamics для того же блока.
  Значит H_old = H(final_answer), а затем H_new = H(final_answer).
  ΔH ≡ 0 → π_r = 0 → d_eS = 0, d_iS = 0 → Φ = |d_eS| − d_iS ≡ 0.

Скрипт прогоняет ту же blackboard-механику на реальных текстах прогона
в двух вариантах и печатает разницу:
  A. «как в Mycelium»: апсерт черновиков + final, потом compute_thermodynamics
  B. «контрфактик»:    апсерт только черновиков,   потом compute_thermodynamics

    python lab/thermo_counterfactual.py --tag full
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import yaml                                    # noqa: E402
from core.blackboard import SemanticBlackboard  # noqa: E402


def cfg():
    with open(os.path.join(os.path.dirname(HERE), 'config', 'settings.yaml'),
              encoding='utf-8') as f:
        return yaml.safe_load(f)


def run_variant(c, drafts, final, task, q_env, tokens, include_final_in_pre):
    bb = SemanticBlackboard(c)
    bid = 'step-0'
    pre = list(drafts) + ([final] if include_final_in_pre else [])
    for text in pre[:bb.max_versions]:
        bb.upsert(bid, text, task=task, task_type='code')
    return bb.compute_thermodynamics(bid, final, task, 'code', q_env, tokens)


def main(tag):
    import logging
    logging.disable(logging.INFO)
    c = cfg()
    path = os.path.join(HERE, 'data', f'results_{tag}.jsonl')
    rows = [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]
    rows = [r for r in rows if r.get('status') == 'ok' and r.get('gen_texts')
            and r['condition'] == 'M']
    if not rows:
        print("нет записей с сохранёнными gen_texts (нужен прогон после правки)")
        return

    print(f"{'task':>5} | {'A: как в Mycelium':^34} | {'B: без пред-апсерта final':^34}")
    print(f"{'':>5} | {'ΔH':>9}{'Φ':>11}{'alive':>8}{'':>6} | "
          f"{'ΔH':>9}{'Φ':>11}{'alive':>8}")
    print('-' * 82)
    a_pos = b_pos = 0
    for r in rows:
        drafts = r['gen_texts'][:3]
        final = r['answer']
        tokens = max(1, r.get('tokens_completion', 1))
        qe = r.get('q_env_final', 0.5)
        A = run_variant(c, drafts, final, r.get('prompt', ''), qe, tokens, True)
        B = run_variant(c, drafts, final, r.get('prompt', ''), qe, tokens, False)
        a_pos += A.Phi > 0
        b_pos += B.Phi > 0
        print(f"{r['task_id']:>5} | {A.delta_H:>9.4f}{A.Phi:>11.5f}"
              f"{str(A.alive):>8}{'':>6} | {B.delta_H:>9.4f}{B.Phi:>11.5f}"
              f"{str(B.alive):>8}")
    n = len(rows)
    print('-' * 82)
    print(f"Φ > 0:  вариант A (как в Mycelium) {a_pos}/{n}   "
          f"вариант B (контрфактик) {b_pos}/{n}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='full')
    main(ap.parse_args().tag)
