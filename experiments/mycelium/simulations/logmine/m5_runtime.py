"""Ж10. То, что есть ТОЛЬКО в logs/runtime.log и не попало в results_*.jsonl.

В jsonl пишется срез состояния на конец шага. В runtime.log остались события
внутри шага: вырожденность пер-агентной награды (по ТОНКОМУ сигналу
q_env_fine, а не по грубой q_env), сходимость PPR, LOO-награды критиков,
фатальные вызовы. Часть из них — прямые ответы на вопросы отчётов:

  SUMMARY.md п.6 «борьба со срывами тестов» — сколько вызовов вообще не
      состоялось и по какой причине;
  SIM_REPORT_4 §1 «шум съедает половину эффекта» — доля шагов, где награда
      не различает черновиков;
  RESULTS.md §4 «H_norm медиана 0.93» — распределение энтропии блока и
      сходимость PPR, из которого она считается.

Прогоны в логе разделяются по строке «Mycelium vX initialized». Прогон
sh2_200 — последний, 200 шагов, 2026-08-10 19:42..21:51.
"""
from __future__ import annotations

import collections
import json
import os
import re
import statistics as st

from common import ROOT, wilson

LOG = os.path.join(ROOT, 'logs', 'runtime.log')

P_INIT = re.compile(r'\[swarm\] Mycelium v[\d.]+ initialized')
P_STEP = re.compile(r'\[swarm\] Step (\d+) \| task_type')
P_WARN = re.compile(r'per_agent_reward: сигнал одинаков у всех (\d+) '
                    r'черновиков \(([\d.]+)\)')
P_PPR = re.compile(r'PPR did not converge in (\d+) iterations')
P_BLOCK = re.compile(r'Block \[step-(\d+)\] type=(\S+) n=(\d+) edges=(\d+) '
                     r'fallback=(\w+) PPR_conv=(\w+) iters=(\d+) '
                     r'H=([\d.]+) H_norm=([\d.]+) score=([-\d.]+)')
P_LOO = re.compile(r'critic LOO: (.+)$')
P_FATAL = re.compile(r'Missing API key for provider (\S+)\. Marking call as FATAL')
P_DEPTH = re.compile(r'Depth step (\d+): H ([\d.]+)->([\d.]+) q=([\d.]+) '
                     r'E_approx=([-\d.]+) tokens=(\d+)/(\d+)')
P_DSTART = re.compile(r'Depth loop start: block=step(\d+) H=([\d.]+) '
                      r'E_total_ref=([-\d.]+)')
P_TRANS = re.compile(r'Transient server error \((\d+)\) on (\S+)')


def segment():
    runs, cur = [], None
    for line in open(LOG, encoding='utf-8', errors='replace'):
        if P_INIT.search(line):
            cur = {'start': line[:19], 'end': line[:19], 'lines': []}
            runs.append(cur)
        if cur is None:
            continue
        cur['lines'].append(line)
        if line[:4].isdigit():
            cur['end'] = line[:19]
    for r in runs:
        r['steps'] = [int(m.group(1)) for l in r['lines']
                      for m in [P_STEP.search(l)] if m]
    return [r for r in runs if r['steps']]


def main():
    runs = segment()
    run = max(runs, key=lambda r: len(r['steps']))
    print(f"прогонов в логе: {len(runs)}")
    print(f"выбран самый длинный: {run['start']} .. {run['end']}, "
          f"шагов {len(run['steps'])} (это M-условие sh2_200)\n")
    L = run['lines']
    n = len(run['steps'])

    # ── вырожденность ТОНКОГО сигнала награды ──────────────────────────────
    vals = [float(m.group(2)) for l in L for m in [P_WARN.search(l)] if m]
    cnt = collections.Counter(vals)
    p, lo, hi = wilson(len(vals), n)
    print("── Ж10.1. пер-агентная награда: различает ли она черновики ──")
    print(f"   шагов, где q_env_fine одинакова у ВСЕХ черновиков: "
          f"{len(vals)}/{n} = {p:.1%} [{lo:.1%}, {hi:.1%}]")
    print("   на каком значении схлопнулось (q_env_fine = 0.15 + 0.85·ratio):")
    for v, c in sorted(cnt.items()):
        if abs(v - 1.0) < 1e-9:
            what = 'честная ничья: оба решили всё'
        elif abs(v - 0.15) < 1e-9:
            what = 'оба прошли 0 тестов'
        elif abs(v - 0.10) < 1e-9:
            what = 'оба не импортировались'
        elif abs(v) < 1e-9:
            what = 'у обоих синтаксическая ошибка'
        else:
            what = f'оба прошли ratio={(v-0.15)/0.85:.2f}'
        print(f"      {v:.3f}  ×{c:<4} {what}")
    honest = cnt.get(1.0, 0)
    print(f"   честных ничьих {honest}, вырожденных {len(vals)-honest} "
          f"= {(len(vals)-honest)/n:.1%} шагов прогона")

    # ── PPR, из которого считается энтропия блока ──────────────────────────
    blocks = [P_BLOCK.search(l) for l in L]
    blocks = [b for b in blocks if b]
    conv = sum(1 for b in blocks if b.group(6) == 'True')
    fb = sum(1 for b in blocks if b.group(5) == 'True')
    types = collections.Counter(b.group(2) for b in blocks)
    print("\n── Ж10.2. чем считается энтропия: сходимость PPR ──")
    print(f"   блоков построено: {len(blocks)}, типы: {dict(types)}")
    print(f"   PPR сошёлся: {conv}/{len(blocks)} = {conv/max(len(blocks),1):.1%}")
    print(f"   fallback на text_paragraph (AST не разобрался): "
          f"{fb}/{len(blocks)} = {fb/max(len(blocks),1):.1%}")
    print("   несошедшийся PPR означает, что H и ΔH, на которых стоит вся "
          "термодинамика, посчитаны по недосошедшемуся вектору")
    ns = [int(b.group(3)) for b in blocks]
    es = [int(b.group(4)) for b in blocks]
    print(f"   узлов в блоке: медиана {st.median(ns)}, "
          f"рёбер: медиана {st.median(es)}")

    hnorm = sorted(float(b.group(9)) for b in blocks)
    hnat = sorted(float(b.group(8)) for b in blocks)
    thr = 0.08
    print(f"   H в натах:  медиана {st.median(hnat):.4f} "
          f"[{hnat[0]:.4f}, {hnat[-1]:.4f}]")
    print(f"   H_norm:     медиана {st.median(hnorm):.4f} "
          f"[{hnorm[0]:.4f}, {hnorm[-1]:.4f}]")
    print(f"   порог depth.entropy_threshold={thr}: ниже него "
          f"{sum(1 for h in hnorm if h < thr)}/{len(hnorm)} блоков")
    print("   квантили H_norm: " + "  ".join(
        f"{q:.2f}→{hnorm[int(q*(len(hnorm)-1))]:.4f}"
        for q in (0.01, 0.05, 0.10, 0.25, 0.50)))
    print(f"   чтобы порог отсекал нижние 10 %, он должен быть "
          f"{hnorm[int(0.10*(len(hnorm)-1))]:.3f}, а не {thr}")

    # ── депт-луп: что реально происходило ──────────────────────────────────
    ds = [P_DEPTH.search(l) for l in L]
    ds = [d for d in ds if d]
    dh = [(float(d.group(2)), float(d.group(3))) for d in ds]
    print("\n── Ж10.3. депт-луп ──")
    print(f"   шагов депт-лупа записано: {len(ds)}")
    if dh:
        down = sum(1 for a, b in dh if b < a - 1e-9)
        up = sum(1 for a, b in dh if b > a + 1e-9)
        print(f"   H снизилась: {down}, выросла: {up}, не изменилась: "
              f"{len(dh)-down-up}")
        print(f"   медиана H до {st.median([a for a, _ in dh]):.4f} → "
              f"после {st.median([b for _, b in dh]):.4f}")

    # ── критики LOO ────────────────────────────────────────────────────────
    loo = [P_LOO.search(l).group(1).strip() for l in L if P_LOO.search(l)]
    print("\n── Ж10.4. LOO-награды критиков (пишутся только когда LOO сработал) ──")
    print(f"   строк LOO в этом прогоне: {len(loo)} на {n} шагов "
          f"= {len(loo)/n:.1%}")
    if loo:
        vals2 = [float(x.split('=')[1]) for s in loo for x in s.split()
                 if '=' in x]
        good = sum(1 for v in vals2 if v > 0.5 + 1e-9)
        bad = sum(1 for v in vals2 if v < 0.5 - 1e-9)
        print(f"   оценок критиков: {len(vals2)}, помогла {good}, "
              f"помешала {bad}, не изменила {len(vals2)-good-bad}")
        print(f"   медиана {st.median(vals2):.4f} (0.5 = критика ничего "
              f"не изменила)")
    else:
        print("   LOO в этом прогоне не срабатывал ни разу — "
              "проверить условие len(c_ok) > 1 в orchestrator.py:729")

    # ── сорванные вызовы ───────────────────────────────────────────────────
    fatal = collections.Counter(m.group(1) for l in L
                                for m in [P_FATAL.search(l)] if m)
    trans = collections.Counter(m.group(2) for l in L
                                for m in [P_TRANS.search(l)] if m)
    print("\n── Ж10.5. сорванные вызовы ──")
    print(f"   FATAL «Missing API key» по провайдерам: {dict(fatal)} "
          f"(всего {sum(fatal.values())})")
    print(f"   Transient server error: {dict(trans)} "
          f"(всего {sum(trans.values())})")

    # то же по всему логу — где именно это было
    allfatal = collections.Counter()
    for line in open(LOG, encoding='utf-8', errors='replace'):
        m = P_FATAL.search(line)
        if m:
            allfatal[(line[:10], m.group(1))] += 1
    print(f"   по всему логу, по датам: {dict(allfatal)}")

    out = {'run_start': run['start'], 'run_end': run['end'], 'steps': n,
           'degenerate_fine_reward': len(vals),
           'degenerate_honest_ties': honest,
           'ppr_converged': conv, 'ppr_total': len(blocks),
           'fallback_blocks': fb,
           'loo_lines': len(loo),
           'fatal': dict(fatal), 'transient': dict(trans)}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'out_runtime.json')
    json.dump(out, open(path, 'w', encoding='utf-8'), ensure_ascii=False,
              indent=1)
    print(f"\nсохранено: {path}")


if __name__ == '__main__':
    main()
