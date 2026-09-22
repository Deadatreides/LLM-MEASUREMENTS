"""lab/acceptance_offline.py — часть C задания TASK_ACCEPTANCE: офлайн-пересчёт
приёмки на сохранённых текстах прогона sh2_200. Ноль вызовов LLM.

Что делает:
  1. прогоняет каждый сохранённый текст через песочницу ПОАССЕРТНО
     (тесты корпуса уже разбиты на test_0/test_1/test_2, поэтому достаточно
     распарсить `pytest -v` — расщеплять файл не нужно);
  2. проверяет, что восстановленная q_env совпадает с записанной в прогоне —
     без этого доверять пересчёту нельзя;
  3. расщепляет ассерты: test_0 → приёмка, test_1+test_2 → оценка (§B.2);
  4. применяет трёхступенчатый гейт §B.3 и считает, что было бы выбрано;
  5. сравнивает с фактическим выбором в обе стороны.

Про исполнение кода. Тексты — решения MBPP от локальных моделей 0.5-1.7B,
уже исполнявшиеся в самом прогоне sh2_200 тем же харнессом. Каждый запуск в
своей временной папке, с таймаутом, как в core/scorer.py::_q_code_safe.

Запуск:
    python lab/acceptance_offline.py --run        # песочница, ~1400 текстов
    python lab/acceptance_offline.py --analyze    # разбор, мгновенно
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import os
import re
import statistics as st
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

RESULTS = os.path.join(HERE, 'data', 'results_sh2_200.jsonl')
CORPUS = os.path.join(HERE, 'data', 'corpus200.json')
CACHE = os.path.join(HERE, 'data', 'acceptance_assertlevel.jsonl')
OUT = os.path.join(HERE, 'data', 'acceptance_offline.json')

ACCEPT_TEST = 'test_0'                    # приёмочный ассерт
EVAL_TESTS = ('test_1', 'test_2')         # оценочные

CODE_FENCE = re.compile(r'```(?:python|py)?\s*(.*?)```',
                        re.IGNORECASE | re.DOTALL)
PYTEST_LINE = re.compile(r'::(test_\d+)\s+(PASSED|FAILED|ERROR)')


def extract_code(text: str) -> str:
    """Идентично core/scorer.py::_extract_python_code — иначе пересчёт
    поедет относительно исторических чисел."""
    blocks = CODE_FENCE.findall(text or '')
    return '\n\n'.join(blocks).strip() if blocks else (text or '').strip()


# ── песочница, поассертно ──────────────────────────────────────────────────
def run_per_assert(code_text: str, tests_src: str) -> dict:
    import ast
    try:
        ast.parse(code_text)
    except SyntaxError:
        return {'status': 'syntax_error', 'tests': {}}
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, 'solution.py'), 'w', encoding='utf-8') as f:
            f.write(code_text)
        with open(os.path.join(d, 'test_solution.py'), 'w',
                  encoding='utf-8') as f:
            f.write(tests_src)
        try:
            r = subprocess.run(
                [sys.executable, '-c',
                 'import sys; sys.path.insert(0,"."); import solution'],
                capture_output=True, timeout=5, cwd=d)
            if r.returncode != 0:
                return {'status': 'import_error', 'tests': {}}
            tr = subprocess.run(
                # без -q: он гасит -v, и построчных «::test_0 PASSED» не будет
                [sys.executable, '-m', 'pytest', 'test_solution.py',
                 '-v', '--tb=no', '--no-header', '-p', 'no:cacheprovider'],
                capture_output=True, timeout=20, cwd=d)
            out = tr.stdout.decode('utf-8', errors='ignore')
            tests = {name: (verdict == 'PASSED')
                     for name, verdict in PYTEST_LINE.findall(out)}
            if not tests:
                return {'status': 'no_tests_ran', 'tests': {}}
            return {'status': 'ok', 'tests': tests}
        except subprocess.TimeoutExpired:
            return {'status': 'timeout', 'tests': {}}
        except FileNotFoundError:
            return {'status': 'no_pytest', 'tests': {}}
        except Exception as exc:
            return {'status': f'error:{type(exc).__name__}', 'tests': {}}


def q_env_from(detail: dict, n_total: int = 3) -> float:
    """Историческая шкала core/scorer.py::_q_code_safe — для сверки."""
    s = detail['status']
    if s == 'syntax_error' or s == 'import_error':
        return 0.2
    if s == 'timeout':
        return 0.1
    if s == 'no_tests_ran':
        return 0.5
    if s != 'ok':
        return 0.0
    passed = sum(1 for v in detail['tests'].values() if v)
    total = len(detail['tests'])
    if total == 0:
        return 0.5
    ratio = passed / total
    return 1.0 if ratio == 1.0 else max(0.4, ratio * 0.8)


# ── сбор текстов ───────────────────────────────────────────────────────────
def load():
    recs = [json.loads(l) for l in open(RESULTS, encoding='utf-8') if l.strip()]
    M = [r for r in recs if r['condition'] == 'M' and r.get('status') == 'ok']
    B = [r for r in recs if r['condition'] == 'BoN-G' and r.get('status') == 'ok']
    corpus = json.load(open(CORPUS, encoding='utf-8'))['tasks']
    tests = {t['task_id']: t['tests'] for t in corpus}
    return M, B, tests


def collect_texts(M):
    """[(key, task_id, kind, idx, text)] — все тексты, которые шаг произвёл."""
    items = []
    for r in M:
        tid = r['task_id']
        for i, t in enumerate(r.get('gen_texts') or []):
            items.append((tid, 'G', i, t))
        for i, t in enumerate(r.get('syn_texts') or []):
            items.append((tid, 'S', i, t))
        if r.get('answer'):
            items.append((tid, 'final', 0, r['answer']))
    return items


def key_of(tid, text):
    return hashlib.sha1(f'{tid}\x00{text}'.encode('utf-8')).hexdigest()[:20]


def do_run(args):
    M, B, tests = load()
    items = collect_texts(M)
    cache = {}
    if os.path.exists(CACHE):
        for line in open(CACHE, encoding='utf-8'):
            if line.strip():
                d = json.loads(line)
                cache[d['key']] = d
    todo = [(key_of(tid, txt), tid, kind, idx, txt)
            for tid, kind, idx, txt in items
            if key_of(tid, txt) not in cache]
    print(f"текстов всего {len(items)}, уже в кеше {len(items)-len(todo)}, "
          f"к прогону {len(todo)}")
    if not todo:
        print("нечего делать")
        return 0

    fh = open(CACHE, 'a', encoding='utf-8')
    done = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_per_assert, extract_code(txt), tests[tid]):
                (k, tid, kind, idx)
                for k, tid, kind, idx, txt in todo}
        for fut in cf.as_completed(futs):
            k, tid, kind, idx = futs[fut]
            try:
                det = fut.result()
            except Exception as exc:
                det = {'status': f'harness_error:{type(exc).__name__}',
                       'tests': {}}
            rec = {'key': k, 'task_id': tid, 'kind': kind, 'idx': idx,
                   'status': det['status'], 'tests': det['tests'],
                   'q_env_recomputed': q_env_from(det)}
            fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
            done += 1
            if done % 100 == 0:
                fh.flush()
                os.fsync(fh.fileno())
                print(f"  {done}/{len(todo)}", flush=True)
    fh.close()
    print("готово")
    return 0


# ── статистика ─────────────────────────────────────────────────────────────
def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def solved_old(det):
    return det['status'] == 'ok' and len(det['tests']) > 0 and \
        all(det['tests'].values())


def solved_new(det):
    """Оценка по двум ассертам: test_1 и test_2."""
    if det['status'] != 'ok':
        return False
    t = det['tests']
    return all(t.get(name) is True for name in EVAL_TESTS)


def accepts(det):
    """Ступень 2: приёмочный ассарт test_0 прошёл."""
    return det['status'] == 'ok' and det['tests'].get(ACCEPT_TEST) is True


def importable(det):
    """Ступень 1: компилируется и импортируется."""
    return det['status'] not in ('syntax_error', 'import_error', 'timeout',
                                 'no_pytest') and not det['status'].startswith(
                                     ('error:', 'harness_error'))


def n_passed(det):
    return sum(1 for v in det['tests'].values() if v) \
        if det['status'] == 'ok' else 0


# ── разбор ─────────────────────────────────────────────────────────────────
def do_analyze(args):
    import yaml
    from core.scorer import QualityScorer
    scorer = QualityScorer(yaml.safe_load(
        open(os.path.join(ROOT, 'config', 'settings.yaml'), encoding='utf-8')))

    M, B, tests = load()
    cache = {}
    for line in open(CACHE, encoding='utf-8'):
        if line.strip():
            d = json.loads(line)
            cache[d['key']] = d

    def det(tid, text):
        return cache.get(key_of(tid, text))

    # ── 0. СВЕРКА: восстановленная q_env против записанной в прогоне ───────
    mis, tot = 0, 0
    for r in M:
        tid = r['task_id']
        pairs = list(zip(r.get('gen_texts') or [], r.get('q_env_candidates') or []))
        pairs += list(zip(r.get('syn_texts') or [], r.get('q_env_synth_all') or []))
        if r.get('answer'):
            pairs.append((r['answer'], r.get('q_env_final')))
        for text, q_hist in pairs:
            d = det(tid, text)
            if d is None or q_hist is None:
                continue
            tot += 1
            if abs(d['q_env_recomputed'] - float(q_hist)) > 1e-6:
                mis += 1
    print("── 0. сверка пересчёта с записанным в прогоне ──")
    print(f"   сопоставлено текстов: {tot}, расхождений: {mis} "
          f"({mis/max(tot,1):.2%})")
    if mis > tot * 0.02:
        print("   РАСХОЖДЕНИЕ ВЫШЕ 2 % — дальше считать нельзя, разбираться")
    else:
        print("   песочница воспроизводит исторические числа, пересчёт "
              "достоверен")

    # ── 1. расщепление ассертов: цена по §B.2 ─────────────────────────────
    finals = [(r['task_id'], det(r['task_id'], r.get('answer', '')))
              for r in M if r.get('answer')]
    finals = [(t, d) for t, d in finals if d]
    s3 = sum(1 for _t, d in finals if solved_old(d))
    s2 = sum(1 for _t, d in finals if solved_new(d))
    up = sum(1 for _t, d in finals if solved_new(d) and not solved_old(d))
    dn = sum(1 for _t, d in finals if solved_old(d) and not solved_new(d))
    print("\n── 1. цена расщепления: оценка по двум ассертам вместо трёх ──")
    print(f"   solved по трём ассертам (историческое): {s3}/{len(finals)} = "
          f"{s3/len(finals):.1%}")
    print(f"   solved по двум оценочным:               {s2}/{len(finals)} = "
          f"{s2/len(finals):.1%}")
    print(f"   дрейф {100*(s2-s3)/len(finals):+.1f} п.п.  "
          f"(порог из §B.2: 3-4 п.п.)  вверх {up}, вниз {dn}")
    if abs(s2 - s3) / len(finals) > 0.04:
        print("   дрейф выше порога — сравнение с прошлым прогоном придётся "
              "делать на новой шкале")

    # Вырожденность награды: §D просит проверить, ухудшает ли расщепление.
    # Сравнивать надо ДОЛЮ пройденных (из неё считается q_env_fine =
    # 0.15+0.85·ratio), а не состав пройденных: состав тоньше доли, и на нём
    # расщепление ложно выглядит улучшением.
    def fine(d, names):
        """q_env_fine по подмножеству ассертов; None если сигнала нет."""
        if d['status'] == 'syntax_error':
            return 0.0
        if d['status'] == 'timeout':
            return 0.05
        if d['status'] == 'import_error':
            return 0.10
        if d['status'] != 'ok' or not d['tests']:
            return None
        vals = [d['tests'].get(k) for k in names if k in d['tests']]
        if not vals:
            return None
        return 0.15 + 0.85 * (sum(1 for v in vals if v) / len(vals))

    all3 = (ACCEPT_TEST,) + EVAL_TESTS
    deg3 = deg2 = npair = 0
    for r in M:
        ds = [det(r['task_id'], t) for t in (r.get('gen_texts') or [])]
        ds = [d for d in ds if d]
        if len(ds) < 2:
            continue
        npair += 1
        if len({fine(d, all3) for d in ds}) == 1:
            deg3 += 1
        if len({fine(d, EVAL_TESTS) for d in ds}) == 1:
            deg2 += 1
    print(f"   вырожденность награды по черновикам (одинаковая q_env_fine "
          f"у всех):")
    print(f"      по трём ассертам: {deg3}/{npair} = {deg3/max(npair,1):.1%}"
          f"   (в логе прогона было 58.0 %)")
    print(f"      по двум оценочным: {deg2}/{npair} = {deg2/max(npair,1):.1%}"
          f"   {'— расщепление УХУДШАЕТ различимость' if deg2 > deg3 else ''}")
    print(f"      цена расщепления для награды: "
          f"{100*(deg2-deg3)/max(npair,1):+.1f} п.п. вырожденных шагов")

    # ── 1b. насколько силён сам приёмочный ассерт ─────────────────────────
    # Все три ассерта видны в промпте задачи (проверено: 200 из 200 задач
    # corpus200). Значит кандидат, захардкодивший примеры, проходит приёмку,
    # не решив задачу. Это внутреннее ограничение одноассертного гейта, и его
    # надо измерить, а не предполагать.
    acc_pass = acc_pass_eval_fail = 0
    for d in cache.values():
        if not accepts(d):
            continue
        acc_pass += 1
        if not solved_new(d):
            acc_pass_eval_fail += 1
    p_a, lo_a, hi_a = wilson(acc_pass_eval_fail, acc_pass)
    print("\n── 1b. приёмочный ассарт как фильтр ──")
    print(f"   кандидатов, прошедших приёмку (test_0): {acc_pass}")
    print(f"   из них проваливших оценку (test_1&test_2): {acc_pass_eval_fail}"
          f" = {p_a:.1%} [{lo_a:.1%}, {hi_a:.1%}]")
    print("   это потолок точности гейта: одноассертная приёмка не может"
          " отличить")
    print("   такого кандидата от решившего, потому что все три ассерта видны"
          " в промпте")

    # ── 2. гейт §B.3 ──────────────────────────────────────────────────────
    def q_ext(text):
        try:
            return float(scorer.q_ext(text, 'code'))
        except Exception:
            return 0.0

    rows = []
    for r in M:
        tid = r['task_id']
        pool = []
        for t in (r.get('gen_texts') or []):
            pool.append(('G', t))
        for t in (r.get('syn_texts') or []):
            pool.append(('S', t))
        final_text = r.get('answer') or ''
        pool_full = pool + ([('final', final_text)] if final_text else [])
        cand = [(kind, t, det(tid, t)) for kind, t in pool_full]
        cand = [(k, t, d) for k, t, d in cand if d]
        if not cand:
            continue

        st1 = [c for c in cand if importable(c[2])]
        st2 = [c for c in st1 if accepts(c[2])]
        survivors = st2 or st1 or cand
        chosen = max(survivors, key=lambda c: q_ext(c[1]))

        # §B.3 буквально: ступень 3 выбирает среди черновиков и синтезов,
        # без пост-депт-лупного итога
        cand_b3 = [c for c in cand if c[0] != 'final']
        s1b = [c for c in cand_b3 if importable(c[2])]
        s2b = [c for c in s1b if accepts(c[2])]
        surv_b = s2b or s1b or cand_b3
        chosen_b = max(surv_b, key=lambda c: q_ext(c[1])) if surv_b else chosen

        fd = det(tid, final_text) if final_text else None
        rows.append({
            'task_id': tid,
            'actual_new': bool(fd and solved_new(fd)),
            'actual_old': bool(fd and solved_old(fd)),
            'gate_new': solved_new(chosen[2]),
            'gate_b3_new': solved_new(chosen_b[2]),
            'any_new': any(solved_new(d) for _k, _t, d in cand),
            'any_new_b3': any(solved_new(d) for _k, _t, d in cand_b3),
            'n_cand': len(cand), 'n_st1': len(st1), 'n_st2': len(st2),
            'chosen_kind': chosen[0],
        })

    n = len(rows)
    act = sum(1 for x in rows if x['actual_new'])
    gate = sum(1 for x in rows if x['gate_new'])
    gate_b = sum(1 for x in rows if x['gate_b3_new'])
    orc = sum(1 for x in rows if x['any_new'])
    print("\n── 2. трёхступенчатый гейт на сохранённых текстах ──")
    print(f"   шагов: {n}, кандидатов на шаг: медиана "
          f"{st.median([x['n_cand'] for x in rows]):.0f}")
    print(f"   ступень 1 (импортируется) оставляет: медиана "
          f"{st.median([x['n_st1'] for x in rows]):.0f}")
    print(f"   ступень 2 (test_0 прошёл) оставляет:  медиана "
          f"{st.median([x['n_st2'] for x in rows]):.0f}; "
          f"шагов, где не выжил никто: "
          f"{sum(1 for x in rows if x['n_st2'] == 0)}")
    print(f"\n   solved (шкала из двух ассертов):")
    print(f"      фактический выбор s_ok[0]+депт-луп: {act}/{n} = {act/n:.1%}")
    print(f"      гейт, пул = черновики+синтезы+итог: {gate}/{n} = "
          f"{gate/n:.1%}   ({100*(gate-act)/n:+.1f} п.п.)")
    print(f"      гейт, пул по §B.3 без итога:        {gate_b}/{n} = "
          f"{gate_b/n:.1%}   ({100*(gate_b-act)/n:+.1f} п.п.)")
    print(f"      оракул по всему произведённому:     {orc}/{n} = {orc/n:.1%}")

    # f2 до и после
    f2_old_k = sum(1 for x in rows if x['any_new'] and not x['actual_new'])
    f2_new_k = sum(1 for x in rows if x['any_new'] and not x['gate_new'])
    den = sum(1 for x in rows if x['any_new'])
    p_o, lo_o, hi_o = wilson(f2_old_k, den)
    p_n, lo_n, hi_n = wilson(f2_new_k, den)
    print(f"\n   f2 = P(взятое не проходит | проходящее было), n={den}")
    print(f"      было:  {p_o:.4f} [{lo_o:.4f}, {hi_o:.4f}]")
    print(f"      стало: {p_n:.4f} [{lo_n:.4f}, {hi_n:.4f}]   порог 0.11 "
          f"{'ВЗЯТ' if hi_n < 0.11 or p_n < 0.11 else 'НЕ взят'}")

    # ── 3. изменения в обе стороны (§C) ───────────────────────────────────
    win = [x['task_id'] for x in rows if x['gate_new'] and not x['actual_new']]
    lose = [x['task_id'] for x in rows if x['actual_new'] and not x['gate_new']]
    print("\n── 3. изменения исхода в обе стороны ──")
    print(f"   гейт спас: {len(win)} задач")
    print(f"   гейт сломал: {len(lose)} задач {lose if lose else ''}")
    print(f"   чистый итог: {len(win)-len(lose):+d}")
    print("   (приёмка, спасающая 30 и ломающая 25, — не то же, что "
          "спасающая 30 и ломающая 2)")

    # ── 4. BoN-G: что можно и чего нельзя ─────────────────────────────────
    print("\n── 4. BoN-G: тексты кандидатов не сохранены ──")
    print("   run_experiment.py пишет для BoN-G только q_env_candidates и")
    print("   q_ext_candidates (списки чисел), самих текстов нет. Поассертно")
    print("   пересчитать нельзя. Считаем границы по шкале q_env.")
    # условные вероятности из M: как распределяются провалы по ассертам
    k2_fail_is_accept = k2 = k1_accept = k1 = 0
    for d in cache.values():
        if d['status'] != 'ok' or len(d['tests']) != 3:
            continue
        np_ = sum(1 for v in d['tests'].values() if v)
        if np_ == 2:
            k2 += 1
            if d['tests'].get(ACCEPT_TEST) is False:
                k2_fail_is_accept += 1
        elif np_ == 1:
            k1 += 1
            if d['tests'].get(ACCEPT_TEST) is True:
                k1_accept += 1
    print(f"   из M: при 2 из 3 пройденных провалившийся — приёмочный в "
          f"{k2_fail_is_accept}/{k2} = {k2_fail_is_accept/max(k2,1):.1%}")
    print(f"   из M: при 1 из 3 пройденный — приёмочный в "
          f"{k1_accept}/{k1} = {k1_accept/max(k1,1):.1%}")

    lo_b = hi_b = 0
    n_b = 0
    for r in B:
        qs = r.get('q_env_candidates') or []
        if not qs:
            continue
        n_b += 1
        n_all = sum(1 for q in qs if abs(q - 1.0) < 1e-9)
        n_23 = sum(1 for q in qs if abs(q - 0.5333333333333333) < 1e-6)
        if n_all:
            lo_b += 1
            hi_b += 1
        elif n_23:
            hi_b += 1
    print(f"\n   потолок BoN-G на шкале из двух ассертов: от {lo_b}/{n_b} = "
          f"{lo_b/max(n_b,1):.1%} до {hi_b}/{n_b} = {hi_b/max(n_b,1):.1%}")
    imp = lo_b + (hi_b - lo_b) * (k2_fail_is_accept / max(k2, 1))
    print(f"   точечная оценка при переносе условных долей из M: "
          f"{imp/max(n_b,1):.1%}")
    print("   f1 после гейта из этих записей НЕ считается: нужен ассерт-уровень")
    print("   по каждому кандидату. ТРЕБОВАНИЕ К ЧАСТИ D: сохранять тексты")
    print("   кандидатов BoN-G, иначе ветка снова окажется непроверяемой.")

    out = {'validation_mismatch': mis, 'validation_total': tot,
           'solved3': s3, 'solved2': s2, 'drift_pp': 100*(s2-s3)/len(finals),
           'n_steps': n, 'actual': act, 'gate': gate, 'gate_b3': gate_b,
           'oracle': orc,
           'f2_old': p_o, 'f2_new': p_n, 'f2_new_ci': [lo_n, hi_n],
           'saved': len(win), 'broken': len(lose), 'broken_ids': lose,
           'degenerate_3': deg3, 'degenerate_2': deg2, 'n_pairs': npair,
           'bon_ceiling_lo': lo_b/max(n_b, 1), 'bon_ceiling_hi': hi_b/max(n_b, 1)}
    json.dump(out, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False,
              indent=1)
    print(f"\nсохранено: {OUT}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--analyze', action='store_true')
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    if a.run:
        return do_run(a)
    if a.analyze:
        return do_analyze(a)
    ap.print_help()
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
