"""
lab/bench_pool.py — Ш0.1: одиночный baseline каждой модели пула.

Зачем. После Ш1.3 надо будет проверить, что Thompson разделил модели ПРАВИЛЬНО,
а не просто «разделил». Сверять не с чем, если неизвестно, кто из них на самом
деле сильнее на этом корпусе. Этот скрипт и даёт эталонный порядок.

Что делает: гоняет каждую модель пула в одиночку по корпусу MBPP, без всякой
оркестрации, и считает ту же внешнюю q_env (исполнение эталонных тестов), что
и основной прогон.

Дополнительно проверяет два условия из Ш0.1, без которых замер вырожден:
  * различимость моделей по качеству — иначе mu бандита нечего разделять;
  * различимость реплик одной модели по temp_offset — иначе внутрикастовое
    разнообразие нулевое (в прошлом прогоне 25/30 кандидатов совпадали, а
    D_sem был 0.072).

Запуск (сервер должен быть поднят):
    python lab/llm_server.py --port 8077 --preload
    python lab/bench_pool.py --n 10
    python lab/bench_pool.py --n 30 --temps 0.55,0.7,0.85
"""

import argparse
import json
import os
import statistics as st
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import yaml                                              # noqa: E402
from core.scorer import QualityScorer                    # noqa: E402

SERVER = 'http://127.0.0.1:8077/v1'
CORPUS = os.path.join(ROOT, 'lab', 'data', 'corpus30.json')
OUT = os.path.join(ROOT, 'lab', 'data', 'baseline_pool.json')

SYSTEM = ("You are a Generator. Solve the task in Python. "
          "Reply with a single ```python code block and nothing else.")


def http_json(url, payload=None, timeout=600):
    if payload is None:
        req = urllib.request.Request(url)
    else:
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def list_models():
    return [m['id'] for m in http_json(SERVER + '/models')['data']]


def generate(model, prompt, temperature, max_tokens=512):
    t0 = time.time()
    d = http_json(SERVER + '/chat/completions', {
        'model': model,
        'messages': [{'role': 'system', 'content': SYSTEM},
                     {'role': 'user', 'content': prompt}],
        'temperature': temperature,
        'max_tokens': max_tokens,
    })
    dt = time.time() - t0
    return (d['choices'][0]['message']['content'] or '',
            int(d['usage'].get('completion_tokens', 0)), dt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=10, help='сколько задач корпуса')
    ap.add_argument('--models', default='', help='через запятую; по умолчанию все')
    ap.add_argument('--temps', default='0.7',
                    help='через запятую: проверить различимость реплик по температуре')
    ap.add_argument('--out', default=OUT)
    args = ap.parse_args()

    with open(CORPUS, encoding='utf-8') as f:
        tasks = json.load(f)['tasks'][:args.n]
    with open(os.path.join(ROOT, 'config', 'settings.yaml'), encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    scorer = QualityScorer(cfg)

    try:
        available = list_models()
    except urllib.error.URLError as exc:
        print(f"сервер недоступен на {SERVER}: {exc}\n"
              f"поднимите: python lab/llm_server.py --port 8077 --preload")
        return 2

    models = [m.strip() for m in args.models.split(',') if m.strip()] or available
    unknown = [m for m in models if m not in available]
    if unknown:
        print(f"нет таких моделей на сервере: {unknown}\nдоступны: {available}")
        return 2
    temps = [float(t) for t in args.temps.split(',') if t.strip()]

    print(f"корпус: {len(tasks)} задач | моделей: {len(models)} | "
          f"температур: {temps}\n")

    results = {}
    for model in models:
        per_temp = {}
        for temp in temps:
            qs, solved, toks, secs, answers = [], 0, 0, 0.0, []
            for t in tasks:
                try:
                    txt, n_out, dt = generate(model, t['prompt'], temp)
                except Exception as exc:
                    print(f"  {model} t={temp} task {t['task_id']}: ОШИБКА {exc}")
                    txt, n_out, dt = '', 0, 0.0
                q = scorer.q_env(txt, 'code', t['tests'])
                qs.append(q)
                solved += 1 if q >= 1.0 else 0
                toks += n_out
                secs += dt
                answers.append(txt)
            per_temp[temp] = {
                'q_env_mean': round(st.mean(qs), 4),
                'solved': round(solved / len(tasks), 4),
                'tok_s': round(toks / secs, 1) if secs else 0.0,
                'tokens_mean': round(toks / len(tasks), 1),
                'q_env': qs,
                'answers_head': [a[:200] for a in answers[:2]],
            }
            print(f"  {model:<38} t={temp:<5} q_env={per_temp[temp]['q_env_mean']:.3f}  "
                  f"solved={per_temp[temp]['solved']:.0%}  "
                  f"{per_temp[temp]['tok_s']:.0f} ток/с")
        results[model] = per_temp

    # ── сводка и проверки Ш0.1 ─────────────────────────────────────────────
    print("\n=== BASELINE (эталонный порядок для сверки с mu после Ш1.3) ===")
    order = sorted(results, key=lambda m: -results[m][temps[0]]['q_env_mean'])
    for i, m in enumerate(order, 1):
        r = results[m][temps[0]]
        print(f"  {i}. {m:<38} q_env={r['q_env_mean']:.3f}  "
              f"solved={r['solved']:.0%}  {r['tok_s']:.0f} ток/с")

    qmeans = [results[m][temps[0]]['q_env_mean'] for m in models]
    spread = max(qmeans) - min(qmeans)
    print(f"\nразброс качества по пулу: {spread:.3f}")
    if spread < 0.15:
        print("  FAIL Ш0.1: модели неразличимы по качеству — mu бандита нечего")
        print("       разделять, и проверить корректность отбора будет нельзя.")
    else:
        print("  PASS Ш0.1: пул различим по качеству")

    if len(temps) > 1:
        print("\nразличимость реплик по температуре (внутрикастовое разнообразие):")
        ok_any = False
        for m in models:
            vals = [results[m][t]['q_env_mean'] for t in temps]
            d = max(vals) - min(vals)
            ident = sum(1 for i in range(len(tasks))
                        if len({results[m][t]['q_env'][i] for t in temps}) == 1)
            print(f"  {m:<38} разброс по t: {d:.3f}  "
                  f"задач с одинаковым исходом: {ident}/{len(tasks)}")
            if ident < len(tasks) * 0.7:
                ok_any = True
        print("  PASS" if ok_any else "  FAIL",
              "— температура даёт различимые ответы"
              if ok_any else "— реплики неразличимы, нужен другой источник разнообразия")

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({'corpus_n': len(tasks), 'temps': temps,
                   'results': results, 'order': order,
                   'spread': spread}, f, ensure_ascii=False, indent=2)
    print(f"\nсохранено: {args.out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
