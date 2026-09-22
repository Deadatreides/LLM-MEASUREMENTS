"""lab/pool_candidates.py — отбор кандидатов в пул (TASK_POOL_CANDIDATES.md).

Критерий один и зафиксирован ДО прогона: дополнительность
`c = P(кандидат решает | coder15 провалил)` против порога c* = 0.125.
Не сила. В нынешнем пуле модель с худшим baseline (smol17, 0.4822) даёт
лучшую дополнительность (0.182), а smol360 с более высоким — ровно ноль.

Схема (§2):
  часть 1 — baseline на corpus30: 30 задач × S сэмплов × 3 модели;
  часть 2 — дополнительность на задачах sh2_200: 199 задач × S × 3;
  третья модель — smol17, контроль: его историческое c=0.182 получено другой
  процедурой (парами внутри шага) и напрямую несравнимо.

Запуск:
    python lab/pool_candidates.py --preflight
    python lab/pool_candidates.py --run          # ~2061 вызов при S=3
    python lab/pool_candidates.py --analyze

──────────────────────────────────────────────────────────────────────────
РАСХОЖДЕНИЕ В ЗАДАНИИ, РАЗРЕШЕНО ЯВНО

§2 (часть 1): «Тот же харнесс и тот же формат промпта, что породили
              baseline_pool.json» → это lab/bench_pool.py::SYSTEM.
§5.2:         «формат промпта роли G» как в sh2_200 → это
              orchestrator.py:483 system_g.

Это РАЗНЫЕ промпты, и второй невоспроизводим: system_g склеивается с
контекстом из памяти (`context_from_memory[:500]`) и подсказками эволюции,
то есть зависит от шага и от состояния памяти, которого больше нет.

Решение: обе части идут на промпте bench_pool.py. Основания:
  * часть 1 обязана совпадать с baseline_pool.json, иначе baseline несравним;
  * части 1 и 2 обязаны совпадать между собой, иначе c не сопоставим с
    baseline того же прогона;
  * сравнение с smol17 идёт ВНУТРИ прогона — §2 вводит его контролем ровно
    потому, что историческое число несравнимо. Общий сдвиг промпта на вывод
    не влияет: он действует на все три модели одинаково.
Цена: абсолютное c этого прогона не сравнивать с историческим 0.182 напрямую.
Сравнивать с smol17 ЗДЕСЬ.

Температура 0.7: столько было в bench_pool.py, и это центр арсенала LinUCB
[0.6, 0.7, 0.8], которым sh2_200 крутил роль G. max_tokens 512 — как у роли G
в config/models.yaml.

ШКАЛА ОЦЕНКИ — ТРЁХАССЕРТНАЯ, БЕЗ РАСЩЕПЛЕНИЯ. В TASK_ACCEPTANCE ассерты
расщепляются, потому что там идёт ОТБОР по ним и расщепление снимает утечку.
Здесь отбора нет: меряется сырая решаемость каждой модели. Популяции P_hard и
P_soft и пороги (c* = 0.125, smol17 = 0.182) выведены на трёхассертной шкале —
смешивать шкалы нельзя.
"""
from __future__ import annotations

import argparse
import ast
import collections
import json
import math
import os
import random
import statistics as st
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

SERVER = 'http://127.0.0.1:8077/v1'
OUT = os.path.join(HERE, 'data', 'pool_candidates_calls.jsonl')
FAILLOG = os.path.join(HERE, 'data', 'pool_candidates_failures.jsonl')
RESULT = os.path.join(HERE, 'data', 'out_pool_candidates.json')
SH2 = os.path.join(HERE, 'data', 'results_sh2_200.jsonl')

# Промпт и параметры — дословно из lab/bench_pool.py (§2).
SYSTEM = ("You are a Generator. Solve the task in Python. "
          "Reply with a single ```python code block and nothing else.")
TEMPERATURE = 0.7
MAX_TOKENS = 512
CALL_TIMEOUT = 900
MAX_RETRY = 5
BACKOFF = 2.0

BEST = 'qwen2.5-coder-1.5b-instruct-q4_0'
CANDIDATES = {
    'gemma-3-it-1b-q5_k_s': 'gemma1b',
    'llama-3.2-1b-instruct-q4_0': 'llama1b',
    'smollm2-1.7b-instruct-q4_k_m': 'smol17',      # контроль (§2)
}
C_STAR = 0.125
BASELINE_REF = {'coder15': 0.6844, 'qwen3': 0.5689, 'smol17': 0.4822}
GAP_MIN = 0.09          # §4: ниже этого бандит пару не различит

# ── популяции: расхождение с заданием, разрешено явно ──────────────────────
# §3 задаёт P_hard=58 и P_soft=142. Первое воспроизводится, второе — нет,
# и не может: 142 это ДОПОЛНЕНИЕ P_hard (200−58), то есть «coder15 решил хотя
# бы раз», а §3 определяет P_soft как «провалил хотя бы одну попытку». Это
# противоположные условия, и подстановка 142 обратила бы условие.
#
# Пересчитано каноническим фильтром (тем же, что дал c=0.075 и c=0.019):
#     задач с попытками coder15 199, попыток 3474
#     провалил ВСЕ            = 58    ← совпадает с заданием
#     провалил хотя бы одну   = 160   ← это и есть P_soft по смыслу §3
#     решил хотя бы раз       = 141   ← отсюда «142» в задании
# Берётся смысл §3 («между провалила один раз и провалила семнадцать»),
# а не число.
EXPECT_HARD, EXPECT_SOFT = 58, 160

# Прогоны, из которых строятся популяции. acc_* ИСКЛЮЧЕНЫ намеренно: там
# оценка идёт по ДВУМ ассертам (test_0 ушёл в приёмку), q_env на другой шкале,
# и подмешивание сделало бы «coder15 решил эту задачу» несравнимым между
# задачами. Тихое загрязнение, которое ничем бы себя не выдало.
EXCLUDE_TAGS = ('acc_',)


# ── статистика ─────────────────────────────────────────────────────────────
def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def binom_le(k, n, p0):
    """P(X <= k) при X~Bin(n,p0) — односторонний тест H0: c >= p0."""
    return sum(math.comb(n, i) * p0 ** i * (1 - p0) ** (n - i)
               for i in range(0, k + 1))


# ── популяции из sh2_200 ───────────────────────────────────────────────────
GGUF_POOL = {
    'qwen2.5-coder-1.5b-instruct-q4_0': 'coder15',
    'qwen3-1.7b-q4_0': 'qwen3',
    'smollm2-1.7b-instruct-q4_k_m': 'smol17',
    'smollm2-360m-instruct-q5_k_m': 'smol360',
    'qwen2.5-0.5b-instruct-q5_0': 'qwen05',
}


def _is_gguf_run(rec, tag):
    """Прогон на GGUF-пуле? Старый safetensors-пул несравним (HANDOFF §3.2)."""
    for c in rec.get('calls') or []:
        if c.get('model'):
            return c['model'] in GGUF_POOL
    return tag not in ('full', 'smoke', 'cf', 'fixcheck')


def load_populations():
    """P_hard / P_soft по всем попыткам coder15 на GGUF-прогонах.

    Пул прогонов тот же, на котором мерились c=0.075 и c=0.019
    (simulations/logmine/m1_complement.py): иначе пороги 0.125 и 0.182
    относились бы к другой популяции.

    Модель кандидата восстанавливается сопоставлением calls[role=G,ok] с
    q_env_candidates: оба списка строятся одним фильтром в одном порядке
    (run_experiment.py:361 и :408).
    """
    import glob
    attempts = collections.defaultdict(list)
    used = []
    for path in sorted(glob.glob(os.path.join(HERE, 'data',
                                              'results_*.jsonl'))):
        tag = os.path.basename(path)[len('results_'):-len('.jsonl')]
        if any(tag.startswith(x) for x in EXCLUDE_TAGS):
            continue
        used.append(tag)
        for line in open(path, encoding='utf-8'):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get('status') != 'ok' or not _is_gguf_run(r, tag):
                continue
            tid = r['task_id']
            if r['condition'] == 'BoN-G':
                for q in r.get('q_env_candidates') or []:
                    attempts[tid].append(float(q) >= 1.0)
                continue
            if r['condition'] != 'M':
                continue
            g = [c for c in (r.get('calls') or [])
                 if c.get('role') == 'G' and c.get('ok')]
            qs = r.get('q_env_candidates') or []
            if len(g) != len(qs):
                continue
            for c, q in zip(g, qs):
                if GGUF_POOL.get(c.get('model')) == 'coder15':
                    attempts[tid].append(float(q) >= 1.0)

    hard = sorted(t for t, v in attempts.items() if v and not any(v))
    soft = sorted(t for t, v in attempts.items() if v and not all(v))
    return attempts, hard, soft, used


# ── вызов с ретраями ───────────────────────────────────────────────────────
class Transient(Exception):
    pass


def _post(payload, timeout=CALL_TIMEOUT):
    req = urllib.request.Request(
        SERVER + '/chat/completions',
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        if e.code >= 500:
            raise Transient(f'HTTP {e.code}')
        raise
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        raise Transient(str(e))


def call(model, prompt, meta):
    """Возвращает (text, tokens, seconds) или бросает RuntimeError.

    §5.4: сорванный вызов НЕ считается нерешённой попыткой — он выбрасывается
    из знаменателя. Иначе модель с худшей стабильностью получит заниженное c.
    """
    payload = {'model': model,
               'messages': [{'role': 'system', 'content': SYSTEM},
                            {'role': 'user', 'content': prompt}],
               'temperature': TEMPERATURE, 'max_tokens': MAX_TOKENS}
    delay = BACKOFF
    for attempt in range(1, MAX_RETRY + 1):
        t0 = time.time()
        try:
            d = _post(payload)
            return (d['choices'][0]['message']['content'] or '',
                    int(d.get('usage', {}).get('completion_tokens', 0)),
                    round(time.time() - t0, 1))
        except Transient as exc:
            kind = 'transient'
        except Exception as exc:                       # noqa: F841
            kind = 'fatal'
        rec = dict(meta, attempt=attempt, kind=kind,
                   error=str(sys.exc_info()[1])[:200], ts=int(time.time()))
        with open(FAILLOG, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        if kind == 'fatal' or attempt == MAX_RETRY:
            raise RuntimeError(f'{kind}: {sys.exc_info()[1]}')
        time.sleep(delay)
        delay *= 2


# ── прогон ─────────────────────────────────────────────────────────────────
def load_done():
    done = {}
    if not os.path.exists(OUT):
        return done
    for line in open(OUT, encoding='utf-8'):
        if line.strip():
            r = json.loads(line)
            done[(r['model'], r['task_id'], r['sample'])] = r
    return done


def do_run(args):
    import yaml
    from core.scorer import QualityScorer
    scorer = QualityScorer(yaml.safe_load(
        open(os.path.join(ROOT, 'config', 'settings.yaml'), encoding='utf-8')))

    c30 = json.load(open(os.path.join(HERE, 'data', 'corpus30.json'),
                         encoding='utf-8'))['tasks']
    c200 = json.load(open(os.path.join(HERE, 'data', 'corpus200.json'),
                          encoding='utf-8'))['tasks']
    by_id = {t['task_id']: t for t in c200}
    for t in c30:
        by_id.setdefault(t['task_id'], t)

    attempts, hard, soft, _used = load_populations()
    part2_ids = sorted(attempts)                 # все задачи с попытками coder15
    part1_ids = [t['task_id'] for t in c30]

    plan = []
    for m in CANDIDATES:
        for tid in part1_ids:
            for s in range(args.samples):
                plan.append((m, tid, s, 'part1'))
        for tid in part2_ids:
            if tid in part1_ids:
                continue          # corpus30 ⊂ corpus200: не гонять дважды
            for s in range(args.samples):
                plan.append((m, tid, s, 'part2'))

    done = load_done()
    todo = [p for p in plan if (p[0], p[1], p[2]) not in done]
    # §5.2: порядок перемешать по всему плану — иначе при --max-resident 1
    # свопы и прогрев лягут на модели неравномерно
    random.Random(args.seed).shuffle(todo)
    print(f"план {len(plan)} вызовов, сделано {len(plan)-len(todo)}, "
          f"осталось {len(todo)}")

    t0 = time.time()
    for i, (model, tid, sample, part) in enumerate(todo, 1):
        task = by_id[tid]
        meta = {'model': model, 'task_id': tid, 'sample': sample}
        try:
            text, ntok, secs = call(model, task['prompt'], meta)
            q = float(scorer.q_env(text, 'code', task['tests']))
            rec = {**meta, 'part': part, 'q_env': q, 'solved': int(q >= 1.0),
                   'tokens': ntok, 'seconds': secs, 'broken': None,
                   'text': text[:3000], 'ts': int(time.time())}
        except RuntimeError as exc:
            rec = {**meta, 'part': part, 'q_env': None, 'solved': None,
                   'tokens': 0, 'seconds': 0.0, 'broken': str(exc),
                   'text': '', 'ts': int(time.time())}
        with open(OUT, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            f.flush()
            os.fsync(f.fileno())
        if i % 25 == 0 or i == len(todo):
            el = (time.time() - t0) / 60
            print(f"  {i}/{len(todo)}  {el:.0f} мин, осталось ~"
                  f"{el/i*(len(todo)-i):.0f} мин", flush=True)
    print("готово")
    return 0


# ── разбор ─────────────────────────────────────────────────────────────────
def do_analyze(args):
    from core.scorer import QualityScorer      # noqa: F401  (шкала та же)
    rows = [json.loads(l) for l in open(OUT, encoding='utf-8') if l.strip()]
    attempts, hard, soft, used = load_populations()
    hard_s, soft_s = set(hard), set(soft)

    print("── популяции (трёхассертная шкала) ──")
    print(f"   прогоны: {used}   (acc_* исключены: там шкала из двух ассертов)")
    print(f"   задач с попытками coder15: {len(attempts)}")
    print(f"   P_hard (провалил ВСЕ):     {len(hard)}   ожидалось {EXPECT_HARD}"
          f"  {'OK' if len(hard) == EXPECT_HARD else 'РАСХОЖДЕНИЕ'}")
    print(f"   P_soft (провалил хотя бы одну): {len(soft)}   ожидалось "
          f"{EXPECT_SOFT}  {'OK' if len(soft) == EXPECT_SOFT else 'РАСХОЖДЕНИЕ'}")
    if len(hard) != EXPECT_HARD or len(soft) != EXPECT_SOFT:
        print("   СТОП: популяция построена не так, как в разборе. c не "
              "считать, пока не сойдётся.")
        return 2

    by_model = collections.defaultdict(list)
    for r in rows:
        by_model[r['model']].append(r)

    out = {}
    print("\n── часть 1: baseline на corpus30 ──")
    print(f"   {'модель':<14} {'q_env':>7} {'solved':>7} {'n':>5} "
          f"{'разрыв с coder15':>17} {'с qwen3':>9}")
    for m, short in CANDIDATES.items():
        rs = [r for r in by_model[m] if r['part'] == 'part1' and r['q_env'] is not None]
        if not rs:
            continue
        q = st.mean(r['q_env'] for r in rs)
        sv = sum(r['solved'] for r in rs) / len(rs)
        g1 = BASELINE_REF['coder15'] - q
        g2 = abs(BASELINE_REF['qwen3'] - q)
        print(f"   {short:<14} {q:>7.4f} {sv:>7.1%} {len(rs):>5} "
              f"{g1:>17.4f} {g2:>9.4f}"
              + ('' if g1 >= GAP_MIN and g2 >= GAP_MIN
                 else '   ← разрыв < 0.09, бандит не различит'))
        out.setdefault(short, {})['baseline_q_env'] = q
        out[short]['baseline_solved'] = sv
        out[short]['gap_coder15'] = g1

    print("\n── часть 2: дополнительность ──")
    print(f"   {'модель':<10} {'популяция':<8} {'c':>7} {'95% ДИ':>18} "
          f"{'n':>5} {'p(H0: c>=0.125)':>17}")
    for m, short in CANDIDATES.items():
        for name, pop in (('P_soft', soft_s), ('P_hard', hard_s)):
            rs = [r for r in by_model[m]
                  if r['task_id'] in pop and r['q_env'] is not None]
            k = sum(r['solved'] for r in rs)
            n = len(rs)
            p, lo, hi = wilson(k, n)
            pv = binom_le(k, n, C_STAR) if n else 1.0
            print(f"   {short:<10} {name:<8} {p:>7.4f} "
                  f"[{lo:.4f}, {hi:.4f}] {n:>5} {pv:>17.4g}")
            out.setdefault(short, {})[f'c_{name}'] = p
            out[short][f'c_{name}_ci'] = [lo, hi]
            out[short][f'c_{name}_n'] = n

    print("\n── вложенность и поимённый список (§3) ──")
    solved_by_best = {t for t, v in attempts.items() if any(v)}
    for m, short in CANDIDATES.items():
        rs = [r for r in by_model[m] if r['q_env'] is not None]
        a = [r for r in rs if r['task_id'] in solved_by_best]
        b = [r for r in rs if r['task_id'] in hard_s]
        pa = sum(r['solved'] for r in a) / max(len(a), 1)
        pb = sum(r['solved'] for r in b) / max(len(b), 1)
        ids = sorted({r['task_id'] for r in b if r['solved']})
        print(f"   {short:<10} P(решает | coder15 решил) = {pa:.4f}, "
              f"| coder15 провалил всё = {pb:.4f}, отношение "
              f"{pb/max(pa,1e-9):.3f}")
        print(f"      решил, а coder15 не смог: {ids if ids else '—'} "
              f"← ПРОВЕРИТЬ ГЛАЗАМИ, что это не вырожденный тест")
        out.setdefault(short, {})['nested_ratio'] = pb / max(pa, 1e-9)
        out[short]['won_task_ids'] = ids

    print("\n── статусы q_env по моделям (§5.3: дефект шаблона или модели) ──")
    def status(q):
        for v, s in ((0.0, 'no_pytest'), (0.1, 'timeout'),
                     (0.2, 'syntax/import'), (0.4, 'ratio<=0.5'),
                     (0.5, 'no_tests_ran'), (0.7, 'imported_no_tests'),
                     (1.0, 'all_passed')):
            if abs(q - v) < 1e-9:
                return s
        return 'partial>0.5'
    base = None
    for m, short in CANDIDATES.items():
        rs = [r for r in by_model[m] if r['q_env'] is not None]
        c = collections.Counter(status(r['q_env']) for r in rs)
        tot = max(len(rs), 1)
        si = c['syntax/import'] / tot
        if short == 'smol17':
            base = si
        print(f"   {short:<10} " + '  '.join(
            f"{k}={v/tot:.1%}" for k, v in c.most_common(4)))
        out.setdefault(short, {})['syntax_import_share'] = si
    if base is not None:
        for m, short in CANDIDATES.items():
            si = out.get(short, {}).get('syntax_import_share')
            if si is not None and si > 2 * base:
                print(f"   ТРЕВОГА (§6 ONBOARDING): у {short} "
                      f"syntax/import {si:.1%} — вдвое выше smol17 ({base:.1%}). "
                      f"Это почти всегда дефект шаблона, а не модели. "
                      f"Остановиться и вернуться к MODELS_ONBOARDING §5.")

    print("\n── срывы вызовов (§5.4) ──")
    br = collections.Counter(r['model'] for r in rows if r.get('broken'))
    tot = collections.Counter(r['model'] for r in rows)
    for m, short in CANDIDATES.items():
        print(f"   {short:<10} {br.get(m,0)}/{tot.get(m,0)} = "
              f"{br.get(m,0)/max(tot.get(m,1),1):.2%} "
              f"(исключены из знаменателя, а не засчитаны провалом)")
    shares = [br.get(m, 0)/max(tot.get(m, 1), 1) for m in CANDIDATES]
    if shares and max(shares) - min(shares) > 0.05:
        print("   ВНИМАНИЕ: доля срывов различается между моделями более чем "
              "на 5 п.п. — проверить, не систематика ли это")

    print("\n── правило решения (§4), зафиксировано ДО прогона ──")
    for m, short in CANDIDATES.items():
        d = out.get(short, {})
        c_soft = d.get('c_P_soft')
        lo = (d.get('c_P_soft_ci') or [0, 0])[0]
        if c_soft is None:
            continue
        c17 = out.get('smol17', {}).get('c_P_soft')
        if short == 'smol17':
            verdict = 'контроль'
        elif lo > C_STAR:
            verdict = 'ВКЛЮЧИТЬ как генератора'
        elif c_soft > C_STAR:
            verdict = 'условно; перемерить с 6 сэмплами'
        elif c17 is not None and c_soft < c17:
            verdict = 'НЕ включать (ниже smol17, который уже в пуле)'
        else:
            verdict = 'НЕ включать (точка ниже порога)'
        gap_ok = (d.get('gap_coder15') or 0) >= GAP_MIN
        c_hard = d.get('c_P_hard')
        # §3: «Одна цифра тут вводит в заблуждение, две дают вилку» — решение
        # принимается по P_soft, но P_hard обязан стоять рядом: расхождение
        # между ними и есть мера того, насколько кандидат вложен в coder15.
        print(f"   {short:<10} c_soft={c_soft:.4f} (нижняя {lo:.4f})  "
              f"c_hard={c_hard:.4f} → {verdict}"
              + ('' if gap_ok else '\n              + разрыв baseline < 0.09: '
                                   'только вместе с маршрутизацией по задачам'))
        if c_hard is not None and c_soft > C_STAR > c_hard:
            print(f"              ВИЛКА: по P_soft проходит, по P_hard нет "
                  f"({c_hard:.3f} < {C_STAR}). Кандидат помогает там, где "
                  f"coder15 спотыкается,")
            print(f"              но не там, где он бессилен. Это не «включить» "
                  f"и не «отвергнуть» — это «включить с маршрутизацией».")
    print("\n   Независимо от исхода (§4): smol360 (0/38) и qwen05 (2/38) из "
          "пула убрать —")
    print("   они не дают дополнительности и тратят 37.8 % генераций.")

    json.dump(out, open(RESULT, 'w', encoding='utf-8'), ensure_ascii=False,
              indent=1)
    print(f"\nсохранено: {RESULT}")
    return 0


# ── контрфактика: что если не исполнять код верхнего уровня ────────────────
# Найдено по ходу прогона. Llama пишет внутрь код-фенса собственные тесты:
# 32 % её ответов содержат исполняемый код на верхнем уровне модуля (59 print,
# 30 assert, 19 присваиваний на первых 104 вызовах), у smol17 таких 5 %, у
# Gemma — ноль. Песочница делает `import solution` первым шагом, этот код
# исполняется, самописный ассерт падает — и ВЕРНОЕ решение получает 0.2, ту же
# оценку, что синтаксическая ошибка.
#
# Правило оценки менять нельзя: sh2_200 и пороги получены при нём. Поэтому
# основной результат считается как есть, а здесь — отдельный контрфактический
# расчёт: сколько c у каждой модели, если срезать верхний уровень до импорта.
# Разница между двумя числами и есть цена стилистического артефакта.
# Срезается ТОЛЬКО самописная проверка: голые выражения (print и прочие вызовы)
# и ассерты. Присваивания верхнего уровня НЕ трогаются — это данные, которыми
# функция может пользоваться (константы, таблицы), и их срезка ломает верное
# решение. Обнаружено на пробе: при срезке всего подряд solved у smol17 упал
# с 28.6 % до 26.5 %, то есть контрфактика портила то, что должна была мерить.
TOPLEVEL_DROP = (ast.Expr, ast.Assert)


def strip_toplevel(code: str):
    """Убрать самописные тесты верхнего уровня. None — не разобралось."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    keep, dropped = [], 0
    for n in tree.body:
        # докстринг модуля — это Expr, но он безвреден; трогаем только то,
        # что реально исполняется с эффектом
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant):
            keep.append(n)
            continue
        if isinstance(n, TOPLEVEL_DROP):
            dropped += 1
            continue
        keep.append(n)
    if not dropped:
        return code
    tree.body = keep
    try:
        return ast.unparse(tree)
    except Exception:
        return None


def do_counterfactual(args):
    import yaml
    from core.scorer import QualityScorer
    scorer = QualityScorer(yaml.safe_load(
        open(os.path.join(ROOT, 'config', 'settings.yaml'), encoding='utf-8')))
    c200 = json.load(open(os.path.join(HERE, 'data', 'corpus200.json'),
                          encoding='utf-8'))['tasks']
    c30 = json.load(open(os.path.join(HERE, 'data', 'corpus30.json'),
                         encoding='utf-8'))['tasks']
    tests = {t['task_id']: t['tests'] for t in c200}
    for t in c30:
        tests.setdefault(t['task_id'], t['tests'])

    rows = [json.loads(l) for l in open(OUT, encoding='utf-8') if l.strip()]
    rows = [r for r in rows if r['q_env'] is not None and r.get('text')]
    if args.limit:
        rows = rows[:args.limit]
    _att, hard, soft, _u = load_populations()
    hard_s, soft_s = set(hard), set(soft)

    from core.acceptance import extract_code
    stats = collections.defaultdict(lambda: {'n': 0, 'changed': 0,
                                             'was': 0, 'now': 0})
    per = collections.defaultdict(lambda: collections.defaultdict(
        lambda: [0, 0]))
    for r in rows:
        m = CANDIDATES.get(r['model'], r['model'])
        code = extract_code(r['text'])
        s = stats[m]
        s['n'] += 1
        s['was'] += r['solved']
        stripped = strip_toplevel(code)
        if stripped is None or stripped == code:
            now = r['solved']
        else:
            s['changed'] += 1
            q = float(scorer.q_env(stripped, 'code', tests[r['task_id']]))
            now = int(q >= 1.0)
        s['now'] += now
        for name, pop in (('P_soft', soft_s), ('P_hard', hard_s)):
            if r['task_id'] in pop:
                per[m][name][0] += now
                per[m][name][1] += 1

    print("── контрфактика: код верхнего уровня не исполняется ──")
    print(f"   {'модель':<10} {'n':>5} {'срезано':>8} {'solved было':>12} "
          f"{'стало':>8}")
    for m, s in stats.items():
        print(f"   {m:<10} {s['n']:>5} {s['changed']:>8} "
              f"{s['was']/s['n']:>12.1%} {s['now']/s['n']:>8.1%}")
    print(f"\n   {'модель':<10} {'популяция':<8} {'c как есть':>11} "
          f"{'c контрфакт':>12}")
    base = json.load(open(RESULT, encoding='utf-8')) if os.path.exists(RESULT) else {}
    for m in per:
        for name in ('P_soft', 'P_hard'):
            k, n = per[m][name]
            was = (base.get(m) or {}).get(f'c_{name}')
            p, lo, hi = wilson(k, n)
            print(f"   {m:<10} {name:<8} "
                  f"{(f'{was:.4f}' if was is not None else '—'):>11} "
                  f"{p:>12.4f}  [{lo:.4f}, {hi:.4f}]  n={n}")
    print("\n   Разница между колонками — цена стилистического артефакта, "
          "а не свойство модели.")
    return 0


# ── preflight ──────────────────────────────────────────────────────────────
def preflight(args):
    print('─' * 72)
    ok = True
    # 1. не идёт ли другой прогон: он занял бы карту и сервер
    busy = []
    for tag in ('acc_200', 'sh2_200'):
        p = os.path.join(HERE, 'data', f'results_{tag}.jsonl')
        if os.path.exists(p) and time.time() - os.path.getmtime(p) < 300:
            busy.append(tag)
    if busy:
        print(f"ЗАНЯТО: {busy} пишется прямо сейчас. Два прогона на одной "
              f"карте — это")
        print("        и потеря скорости, и риск перепутать состояние "
              "llm_server. Дождаться.")
        ok = False
    else:
        print("другие прогоны: не активны")

    # 2. популяции
    try:
        attempts, hard, soft, used = load_populations()
        good = len(hard) == EXPECT_HARD and len(soft) == EXPECT_SOFT
        print(f"популяции: задач {len(attempts)}, P_hard={len(hard)} "
              f"P_soft={len(soft)} "
              f"{'OK' if good else f'РАСХОЖДЕНИЕ с {EXPECT_HARD}/{EXPECT_SOFT}'}")
        print(f"           прогоны: {used}")
        print(f"           acc_* исключены: шкала из двух ассертов несравнима")
        ok = ok and good
    except Exception as exc:
        print(f"популяции: НЕ ПОСТРОИЛИСЬ ({exc})")
        ok = False

    # 3. сервер и модели
    try:
        with urllib.request.urlopen(SERVER + '/models', timeout=10) as r:
            models = [m['id'] for m in json.load(r)['data']]
        missing = [m for m in CANDIDATES if m not in models]
        print(f"llm_server: OK; отсутствуют модели: {missing or 'нет'}")
        ok = ok and not missing
    except Exception as exc:
        print(f"llm_server: НЕДОСТУПЕН ({exc})")
        ok = False

    n = len(CANDIDATES) * (30 + (len(attempts) - 30)) * args.samples \
        if 'attempts' in dir() else 0
    print(f"вызовов при S={args.samples}: ~{3*199*args.samples}")
    print(f"квантование: gemma Q5_K_S против llama Q4_0 — РАСХОЖДЕНИЕ, "
          f"записать в отчёт")
    print(f"             отрицательный результат по llama не отделим от "
          f"квантования (ONBOARDING §1)")
    print('─' * 72)
    return 0 if ok else 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--analyze', action='store_true')
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--samples', type=int, default=3,
                    help='3 даёт мощность 0.84 при c=0.20; 4 поднимает до '
                         '0.93 ценой +597 вызовов (§3)')
    ap.add_argument('--seed', type=int, default=20260812)
    ap.add_argument('--counterfactual', action='store_true',
                    help='пересчёт c, если не исполнять код верхнего уровня')
    ap.add_argument('--limit', type=int, default=0,
                    help='ограничить число текстов (для проверки)')
    ap.add_argument('--force', action='store_true',
                    help='запустить, несмотря на preflight')
    a = ap.parse_args()
    os.makedirs(os.path.join(HERE, 'data'), exist_ok=True)
    if a.preflight:
        return preflight(a)
    if a.counterfactual:
        return do_counterfactual(a)
    if a.analyze:
        return do_analyze(a)
    if a.run:
        if preflight(a) != 0 and not a.force:
            print("preflight не пройден — прогон не запускается (--force "
                  "чтобы всё равно)")
            return 2
        return do_run(a)
    ap.print_help()
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
