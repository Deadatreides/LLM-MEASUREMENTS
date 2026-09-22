# -*- coding: utf-8 -*-
"""Обзор GitHub: всё про инференс нейросетей на C, C# и ассемблере — по метаданным.

Поисковый API GitHub без токена: 10 запросов в минуту, до 100 репозиториев на страницу. Запросы идут
с паузой, при отказе 403 скрипт ждёт сброса окна. Результат — reports/gh_scan.json (все поля) и
reports/gh_scan.md (сводка по языкам, по звёздам).

Запуск: python gh_scan.py
"""
import io
import json
import os
import sys
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "reports")

QUERIES = {
    "C": [
        "llm inference", "llama", "transformer inference", "gguf", "mixture of experts",
        "quantization inference", "neural network inference stars:>200", "llm offload",
        "topic:llm", "topic:inference", "topic:llama", "cuda llm", "int4 kernel",
        "speculative decoding", "kv cache", "mmap model weights", "gpt-2 inference", "tensor library",
    ],
    "C#": [
        "llm inference", "llama", "onnx inference", "transformer stars:>100", "topic:llm", "gguf",
        "neural network inference stars:>100", "tensor", "quantization",
    ],
    "Assembly": [
        "llm", "llama", "inference", "neural network", "matrix multiplication", "simd stars:>50",
        "blas", "transformer", "avx2", "avx512",
    ],
}


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "myc-research", "Accept": "application/vnd.github+json"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (403, 429):
                reset = int(e.headers.get("X-RateLimit-Reset", "0") or 0)
                wait = max(10, reset - int(time.time()) + 2) if reset else 60
                print(f"  лимит, жду {wait} с", flush=True)
                time.sleep(min(wait, 120))
                continue
            if e.code == 422:
                print(f"  запрос отвергнут (422): {url}", flush=True)
                return {"items": []}
            raise
    return {"items": []}


def main():
    os.makedirs(OUT, exist_ok=True)
    repos = {}
    for lang, qs in QUERIES.items():
        for q in qs:
            full = f"{q} language:{lang}"
            url = ("https://api.github.com/search/repositories?"
                   + urllib.parse.urlencode({"q": full, "sort": "stars", "order": "desc", "per_page": 100}))
            d = get(url)
            items = d.get("items", [])
            print(f"[{lang}] {q}: {len(items)} (всего {d.get('total_count')})", flush=True)
            for it in items:
                key = it["full_name"]
                rec = repos.setdefault(key, {
                    "full_name": key, "language": it.get("language"), "stars": it.get("stargazers_count"),
                    "forks": it.get("forks_count"), "open_issues": it.get("open_issues_count"),
                    "description": (it.get("description") or "")[:300], "topics": it.get("topics", []),
                    "pushed_at": it.get("pushed_at"), "created_at": it.get("created_at"),
                    "archived": it.get("archived"), "queries": [],
                })
                rec["queries"].append(f"{lang}: {q}")
            time.sleep(6.5)

    rows = sorted(repos.values(), key=lambda r: -(r["stars"] or 0))
    with io.open(os.path.join(OUT, "gh_scan.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    lines = ["# GitHub: инференс на C, C#, ассемблере", "",
             f"Снято {time.strftime('%Y-%m-%d %H:%M')}; репозиториев {len(rows)}.", ""]
    for lang in QUERIES:
        sub = [r for r in rows if r["language"] == lang]
        lines += [f"## {lang} — {len(sub)}", "", "| звёзды | форки | issues | обновлён | репозиторий | описание |", "|---|---|---|---|---|---|"]
        for r in sub[:120]:
            desc = r["description"].replace("|", "/").replace("\n", " ")
            lines.append(f"| {r['stars']} | {r['forks']} | {r['open_issues']} | {(r['pushed_at'] or '')[:10]} | "
                         f"{r['full_name']}{' (архив)' if r['archived'] else ''} | {desc} |")
        lines.append("")
    with io.open(os.path.join(OUT, "gh_scan.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("SCAN-COMPLETE", len(rows), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
