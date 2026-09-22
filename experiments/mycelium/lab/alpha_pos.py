"""lab/alpha_pos.py — замер alpha_pos на реальных моделях.

Схема опыта — `simulations/logmine/TASK_ALPHA_POS_v2.md` (исходная постановка
`TASK_ALPHA_POS.md`, в ней ничего не изменилось, кроме двух пунктов ниже).

Что решает. Все положительные результаты серии симуляций держатся на допущении,
что длинный контекст теряет середину ИЗ-ЗА РАССТОЯНИЯ, а не просто из-за объёма.
При alpha_pos = 1.0 преимущество дробления +0.451, при 0.0 → −0.007.

    P(L) = acc(B,L) − acc(A,L)          позиционный эффект
    V(L) = acc(C,L) − acc(A,L)          объёмный эффект
    alpha_pos(L) = clip(P/V, 0, 1)

Две поправки, обе внесены по итогам перебора логов sh2_200:

  1. ГРЕЙДЕР СВОЙ И БИНАРНЫЙ. `core.scorer` здесь не импортируется намеренно:
     `q_env` схлопывает 40.6 % всех замеров в одно значение 0.4, а парный
     МакНемар на схлопнутой шкале теряет ровно те пары, ради которых ставится
     опыт. Грейдер ниже — точное совпадение уникального кода, ничего больше.

  2. РЕТРАИ И ОТДЕЛЬНЫЙ ЛОГ СРЫВОВ. В прогоне sh2_200 было 108 срывов вызовов
     (~3 %). В парной схеме сорванный вызов убивает элемент во ВСЕХ трёх
     условиях, поэтому 3 % срывов стоят дороже 3 % мощности. Здесь: до
     MAX_RETRY попыток с откатом, каждый срыв — строкой в отдельный файл,
     элемент с неустранимым срывом помечается broken и выбрасывается из всех
     условий явно, а не молча.

Запуск:
    python lab/llm_server.py --port 8077 --n-ctx 65536 --max-resident 1 \
           --only qwen2.5-coder-1.5b-instruct-q4_0
    python lab/alpha_pos.py --preflight
    python lab/alpha_pos.py --n-items 110 --grid 4096,16384,65536
    python lab/alpha_pos.py --analyze
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics as st
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

SERVER = 'http://127.0.0.1:8077/v1'
OUT = os.path.join(HERE, 'data', 'alpha_pos_calls.jsonl')
FAILLOG = os.path.join(HERE, 'data', 'alpha_pos_failures.jsonl')

MAX_RETRY = 5
BACKOFF = 2.0            # секунды, удваивается
VOTES = 3                # прогонов на вызов, мажоритарное голосование
CALL_TIMEOUT = 900

SYSTEM = ("Answer using only the information given in the text above. "
          "Reply with the access code and nothing else.")


# ── 1. Материал ────────────────────────────────────────────────────────────
# Факт уникален, не выводится из общих знаний и проверяется точным совпадением.
STATIONS = [
    'Кызылорда', 'Аягоз', 'Тобол', 'Ишим', 'Чарын', 'Эмба', 'Иргиз', 'Сарысу',
    'Нура', 'Тургай', 'Уил', 'Шидерты', 'Селеты', 'Каратал', 'Аксу', 'Лепсы',
    'Каргалы', 'Шаган', 'Жайык', 'Илек', 'Хобда', 'Утва', 'Кушум', 'Барбастау',
]
LETTERS = 'BCDFGHJKLMNPQRSTVWXZ'          # без гласных: код не читается словом
CODE_RE = re.compile(r'\b([A-Z]{2}-\d{4})\b')


def make_code(rng):
    return (rng.choice(LETTERS) + rng.choice(LETTERS) + '-'
            + f'{rng.randrange(1000, 10000)}')


def build_items(n, seed=20260811):
    rng = random.Random(seed)
    seen, items = set(), []
    while len(items) < n:
        code = make_code(rng)
        if code in seen:
            continue
        seen.add(code)
        i = len(items)
        st_name = STATIONS[i % len(STATIONS)]
        idx = 3 + (i // len(STATIONS))
        obj = f'насосной станции {st_name}-{idx}'
        items.append({
            'item_id': f'it{i:04d}',
            'code': code,
            'fact': f'Код доступа к резервному контуру {obj} — {code}.',
            'question': f'Какой код доступа у резервного контура {obj}?',
        })
    return items


def load_distractor(paths):
    """Связный текст из той же предметной области. Случайный шум не годится:
    факт выделялся бы контрастом и эффект был бы недооценён."""
    chunks = []
    for p in paths:
        if os.path.isdir(p):
            files = [os.path.join(p, f) for f in sorted(os.listdir(p))
                     if f.lower().endswith(('.md', '.txt'))]
        else:
            files = [p]
        for f in files:
            try:
                txt = open(f, encoding='utf-8', errors='replace').read()
            except OSError:
                continue
            for para in re.split(r'\n\s*\n', txt):
                para = ' '.join(para.split())
                if len(para) < 200 or '```' in para:
                    continue
                # убрать всё, что похоже на наш формат кода
                if CODE_RE.search(para):
                    continue
                chunks.append(para)
    if not chunks:
        raise SystemExit('источник отвлекающего текста пуст')
    return chunks


# ── 2. Сборка контекста ────────────────────────────────────────────────────
def assemble(chunks, rng, target_chars, fact, pos_frac):
    """Отвлекающий текст на target_chars символов, факт на доле pos_frac."""
    body, size = [], 0
    while size < target_chars:
        c = chunks[rng.randrange(len(chunks))]
        body.append(c)
        size += len(c) + 2
    cut = max(1, int(len(body) * pos_frac))
    body.insert(cut, fact)
    return '\n\n'.join(body)


# ── 3. Вызов с ретраями и отдельным логом срывов ───────────────────────────
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


def call(model, context, question, nonce, faillog, meta):
    """Один вызов. Возвращает (text, prompt_tokens) или бросает RuntimeError.

    nonce ломает кеш префикса: без него условия A и B на одном и том же
    отвлекающем тексте получили бы разную фактическую обработку."""
    user = f'[ref {nonce}]\n\n{context}\n\nВопрос: {question}'
    payload = {'model': model,
               'messages': [{'role': 'system', 'content': SYSTEM},
                            {'role': 'user', 'content': user}],
               'temperature': 0.0, 'max_tokens': 24}
    delay = BACKOFF
    for attempt in range(1, MAX_RETRY + 1):
        try:
            d = _post(payload)
            txt = (d['choices'][0]['message']['content'] or '')
            return txt, int(d.get('usage', {}).get('prompt_tokens', 0))
        except Transient as exc:
            rec = dict(meta, attempt=attempt, error=str(exc),
                       kind='transient', ts=int(time.time()))
            with open(faillog, 'a', encoding='utf-8') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            if attempt == MAX_RETRY:
                raise RuntimeError(f'сорвался {MAX_RETRY} раз: {exc}')
            time.sleep(delay)
            delay *= 2
        except Exception as exc:
            rec = dict(meta, attempt=attempt, error=f'{type(exc).__name__}: {exc}',
                       kind='fatal', ts=int(time.time()))
            with open(faillog, 'a', encoding='utf-8') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            raise RuntimeError(str(exc))


# ── 4. Грейдер: свой, бинарный, без core.scorer ────────────────────────────
def grade(answer_text, expected_code):
    """Точное совпадение уникального кода. Бинарно.

    Специально НЕ используется core.scorer._q_code_safe / q_env: та шкала
    шестиуровневая и схлопывает всё с ratio<=0.5 в одну точку, а здесь нужен
    ровно один бит на вызов, иначе парный тест теряет мощность."""
    if not answer_text:
        return 0
    found = CODE_RE.findall(answer_text.upper())
    return 1 if expected_code.upper() in found else 0


def majority(bits):
    return 1 if sum(bits) * 2 > len(bits) else 0


# ── 5. Статистика ──────────────────────────────────────────────────────────
def mcnemar_exact(b, c):
    """Двусторонний точный МакНемар по несогласованным парам.

    Хи-квадрат на долях здесь занизил бы мощность вдвое: выборка парная,
    один и тот же элемент проходит все условия."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, ctr - h), min(1.0, ctr + h))


# ── 6. Прогон ──────────────────────────────────────────────────────────────
def load_done(path):
    done = {}
    if not os.path.exists(path):
        return done
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[(r['item_id'], r['L'], r['cond'])] = r
    return done


def run(args):
    items = build_items(args.n_items, args.seed)
    chunks = load_distractor(args.distractor.split(';'))
    grid = [int(x) for x in args.grid.split(',')]
    rng = random.Random(args.seed ^ 0x5eed)
    done = load_done(OUT)
    print(f"элементов {len(items)}, сетка L {grid}, абзацев-отвлекателей "
          f"{len(chunks)}")
    if done:
        print(f"resume: уже записано {len(done)} ячеек")

    # Контроли — один раз на элемент, а не на каждую длину, и записываются
    # с L=0: иначе анализ не знает, под какой длиной их искать.
    plan = [(it, 0, cond) for it in items for cond in ('ceiling', 'guess')]
    plan += [(it, L, cond)
             for it in items for L in grid for cond in ('A', 'B', 'C')]
    # Порядок перемешан: иначе систематика прогрева и свопа модели ложится
    # на условия неравномерно.
    rng.shuffle(plan)

    t0 = time.time()
    for n_done, (it, L, cond) in enumerate(plan, 1):
        key = (it['item_id'], L, cond)
        if key in done:
            continue
        chars = int((L or grid[0]) * args.chars_per_token)
        if cond == 'ceiling':
            ctx, pos = it['fact'], 0.0
        elif cond == 'guess':
            ctx, pos = assemble(chunks, rng, chars, '', 0.5), -1.0
        elif cond == 'A':
            pos = rng.uniform(0.45, 0.55)       # джиттер: не ровно центр
            ctx = assemble(chunks, rng, chars, it['fact'], pos)
        elif cond == 'B':
            pos = rng.uniform(0.0, 0.05)
            ctx = assemble(chunks, rng, chars, it['fact'], pos)
        else:                                    # C: четверть объёма
            pos = rng.uniform(0.45, 0.55)
            ctx = assemble(chunks, rng, chars // 4, it['fact'], pos)

        meta = {'item_id': it['item_id'], 'L': L, 'cond': cond}
        runs, toks, broken = [], [], None
        for v in range(VOTES):
            nonce = f'{it["item_id"]}-{L}-{cond}-{v}-{rng.randrange(1 << 30)}'
            try:
                txt, ntok = call(args.model, ctx, it['question'], nonce,
                                 FAILLOG, dict(meta, vote=v))
            except RuntimeError as exc:
                broken = str(exc)
                break
            runs.append({'vote': v, 'text': txt[:200],
                         'ok': grade(txt, it['code'])})
            toks.append(ntok)

        rec = {**meta, 'code': it['code'], 'pos_frac': round(pos, 4),
               'prompt_tokens': toks, 'runs': runs,
               'result': majority([r['ok'] for r in runs]) if runs else None,
               'broken': broken, 'model': args.model,
               'ts': int(time.time())}
        with open(OUT, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            f.flush()
            os.fsync(f.fileno())
        done[key] = rec
        el = time.time() - t0
        print(f"  [{n_done}/{len(plan)}] {it['item_id']} L={L} {cond}: "
              f"{rec['result']} tok={toks[0] if toks else '-'} "
              f"{'BROKEN ' + broken if broken else ''} "
              f"({el/60:.0f} мин)", flush=True)


# ── 7. Анализ ──────────────────────────────────────────────────────────────
def analyze(args):
    rows = [json.loads(l) for l in open(OUT, encoding='utf-8') if l.strip()]
    by = {(r['item_id'], r['L'], r['cond']): r for r in rows}
    items = sorted({r['item_id'] for r in rows})
    grid = sorted({r['L'] for r in rows if r['L'] > 0})

    broken = {r['item_id'] for r in rows if r.get('broken')}
    no_ceiling = {i for i in items
                  if (i, 0, 'ceiling') in by
                  and not by[(i, 0, 'ceiling')]['result']}
    guessable = {i for i in items
                 if (i, 0, 'guess') in by
                 and by[(i, 0, 'guess')]['result']}
    bad = broken | no_ceiling | guessable
    good = [i for i in items if i not in bad]

    print("── отсев ──")
    print(f"   элементов всего: {len(items)}")
    print(f"   выброшено по неустранимому срыву вызова: {len(broken)}")
    print(f"   выброшено контролем потолка (модель не знает факт и без "
          f"отвлечения): {len(no_ceiling)}")
    print(f"   выброшено контролем угадывания (факт выводим из вопроса): "
          f"{len(guessable)}")
    print(f"   осталось для парного анализа: {len(good)}")
    if guessable:
        print("   ВНИМАНИЕ: ненулевой контроль угадывания — материал надо "
              "переделать, а не чистить")

    print("\n── по длинам ──")
    out = {}
    for L in grid:
        acc, disc = {}, {}
        for cond in ('A', 'B', 'C'):
            k = sum(1 for i in good
                    if by.get((i, L, cond), {}).get('result'))
            n = sum(1 for i in good if (i, L, cond) in by)
            acc[cond] = wilson(k, n) + (n,)
        for cond in ('B', 'C'):
            b = sum(1 for i in good
                    if by.get((i, L, cond), {}).get('result')
                    and not by.get((i, L, 'A'), {}).get('result'))
            c = sum(1 for i in good
                    if not by.get((i, L, cond), {}).get('result')
                    and by.get((i, L, 'A'), {}).get('result'))
            disc[cond] = (b, c, mcnemar_exact(b, c))
        P = acc['B'][0] - acc['A'][0]
        V = acc['C'][0] - acc['A'][0]
        ap = max(0.0, min(1.0, P / V)) if abs(V) > 1e-9 else float('nan')
        print(f"\n   L = {L}")
        for cond in ('A', 'B', 'C'):
            p, lo, hi, n = acc[cond]
            print(f"      {cond}: acc {p:.3f} [{lo:.3f}, {hi:.3f}]  n={n}")
        print(f"      P(L) = acc(B)−acc(A) = {P:+.3f}   МакНемар b={disc['B'][0]} "
              f"c={disc['B'][1]} p={disc['B'][2]:.4f}")
        print(f"      V(L) = acc(C)−acc(A) = {V:+.3f}   МакНемар b={disc['C'][0]} "
              f"c={disc['C'][1]} p={disc['C'][2]:.4f}")
        print(f"      alpha_pos(L) = {ap:.3f}")
        toks = [t for i in good for t in by.get((i, L, 'A'), {})
                .get('prompt_tokens', [])]
        if toks:
            print(f"      фактических токенов промпта в условии A: медиана "
                  f"{st.median(toks):.0f} (заказано {L})")
        out[L] = {'acc': {k: v[0] for k, v in acc.items()},
                  'P': P, 'V': V, 'alpha_pos': ap,
                  'mcnemar': {k: v for k, v in disc.items()}}

    print("\n── правило решения (TASK_ALPHA_POS §6) ──")
    best = max((v['alpha_pos'] for v in out.values()
                if not math.isnan(v['alpha_pos'])), default=float('nan'))
    if math.isnan(best):
        print("   V(L) неотличим от нуля ни на одной длине — объёмного "
              "эффекта нет, отношение не определено")
    elif best >= 0.5:
        print(f"   alpha_pos = {best:.3f} >= 0.5 → постановка подтверждена, "
              f"строить; зафиксировать длину излома как порог дробления")
    elif best >= 0.2:
        print(f"   0.2 <= alpha_pos = {best:.3f} < 0.5 → эффект есть, но "
              f"слабый: дробление окупается только выше излома и только "
              f"для локальных задач")
    else:
        print(f"   alpha_pos = {best:.3f} < 0.2 → деградация объёмная, "
              f"дробление контекста не лечит потерю середины, направление "
              f"закрывается")

    fails = [json.loads(l) for l in open(FAILLOG, encoding='utf-8')
             if l.strip()] if os.path.exists(FAILLOG) else []
    print(f"\n── срывы вызовов ──")
    print(f"   записей в {os.path.basename(FAILLOG)}: {len(fails)}")
    if fails:
        kinds = {}
        for f in fails:
            kinds[f.get('kind')] = kinds.get(f.get('kind'), 0) + 1
        print(f"   по типам: {kinds}")
        print(f"   из них привели к потере элемента: {len(broken)}")

    path = os.path.join(HERE, 'data', 'alpha_pos_result.json')
    json.dump({'grid': {str(k): v for k, v in out.items()},
               'n_items_used': len(good), 'n_broken': len(broken)},
              open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f"\nсохранено: {path}")


# ── 8. Preflight ───────────────────────────────────────────────────────────
def preflight(args):
    grid = [int(x) for x in args.grid.split(',')]
    print('─' * 70)

    # Объём отвлекающего текста. Если уникального текста меньше, чем нужно на
    # самый длинный контекст, абзацы начнут повторяться, и факт снова
    # выделится контрастом — ровно то, чего материал должен избегать.
    try:
        chunks = load_distractor(args.distractor.split(';'))
        vol = sum(len(c) for c in chunks)
        need = int(max(grid) * args.chars_per_token)
        print(f"отвлекающий текст: {len(chunks)} абзацев, {vol} символов; "
              f"на L={max(grid)} нужно {need}")
        if vol < need:
            print(f"  ВНИМАНИЕ: текста в {need/vol:.1f} раза меньше нужного — "
                  f"абзацы будут повторяться.")
            print(f"  Дай --distractor на корпус побольше, иначе факт "
                  f"выделяется контрастом и эффект искажён.")
    except SystemExit as exc:
        print(f"источник отвлекающего текста: {exc}")
        return 2

    calls = args.n_items * (len(grid) * 3 + 2) * VOTES
    print(f"сетка L: {grid}  (сервер надо было поднять с --n-ctx >= "
          f"{max(grid) + 256})")
    print(f"элементов: {args.n_items}, голосов на вызов: {VOTES}")
    print(f"вызовов всего: ~{calls}")
    print(f"ВНИМАНИЕ: n_ctx задаётся при старте сервера и общий для всех "
          f"моделей.")
    print(f"На 6 ГБ VRAM 1.5B-q4 при ctx 65536 KV-кеш ~1.9 ГБ — проверить "
          f"до прогона,")
    print(f"иначе сетку сдвинуть на 2048/8192/32768 (тот же шаг ×4).")

    try:
        with urllib.request.urlopen(SERVER + '/models', timeout=10) as r:
            models = [m['id'] for m in json.load(r)['data']]
        print(f"llm_server: OK, модели {models}")
    except Exception as exc:
        print(f"llm_server НЕДОСТУПЕН ({exc})")
        print("  подними: python lab/llm_server.py --port 8077 "
              f"--n-ctx {max(grid)+256} --max-resident 1")
        print('─' * 70)
        return 2
    if args.model not in models:
        print(f"модели {args.model} на сервере нет")
        print('─' * 70)
        return 2

    # фактический n_ctx проверяется единственным честным способом: запросом
    try:
        d = _post({'model': args.model,
                   'messages': [{'role': 'user', 'content': 'тест ' * 64}],
                   'temperature': 0.0, 'max_tokens': 4}, timeout=300)
        print(f"пробный вызов прошёл, prompt_tokens="
              f"{d.get('usage', {}).get('prompt_tokens')}")
    except Exception as exc:
        print(f"пробный вызов не прошёл: {exc}")
        print('─' * 70)
        return 2
    print('─' * 70)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-items', type=int, default=110,
                    help='110-120: после отсева должно остаться 100')
    ap.add_argument('--grid', default='4096,16384,65536')
    ap.add_argument('--model', default='qwen2.5-coder-1.5b-instruct-q4_0')
    ap.add_argument('--seed', type=int, default=20260811)
    ap.add_argument('--chars-per-token', type=float, default=3.2,
                    help='калибровка длины; фактические токены пишутся в лог')
    ap.add_argument('--distractor',
                    default=ROOT + ';' + os.path.join(ROOT, 'docs'),
                    help='файлы/каталоги через ; — связный текст той же области')
    ap.add_argument('--analyze', action='store_true')
    ap.add_argument('--preflight', action='store_true')
    a = ap.parse_args()
    os.makedirs(os.path.join(HERE, 'data'), exist_ok=True)
    if a.preflight:
        return preflight(a)
    if a.analyze:
        return analyze(a)
    if preflight(a) != 0:
        return 2
    return run(a)


if __name__ == '__main__':
    sys.exit(main() or 0)
