"""
tools/search.py — Внешняя сенсорика: GitHub + Reddit + Web
Σ_v8.5 «Мицелий»

Аналогия: мицелий выделяет ферменты наружу и переваривает листву.
Внешний поиск — канал d_e S < 0 (приток негэнтропии).
Без него система варится в собственных ошибках.

Запускается асинхронно ПАРАЛЛЕЛЬНО с API-вызовами.
Результат сохраняется в /swarm/context/ для чтения агентами.
"""

import asyncio
import aiohttp
import os
import time
import json
from typing import List, Dict, Optional


class SearchModule:
    def __init__(self, cfg: dict):
        s = cfg['search']
        self.enabled = s['enabled']
        self.gh_n = s['github_results']
        self.rd_n = s['reddit_results']
        self.web_n = s['web_results']
        self.compress_tokens = s['compress_tokens']
        self.context_dir = "swarm/context"
        os.makedirs(self.context_dir, exist_ok=True)

        # Ключи из env
        self.gh_token = os.getenv('GITHUB_TOKEN', '')
        self.google_key = os.getenv('GOOGLE_API_KEY', '')
        self.google_cx = os.getenv('GOOGLE_SEARCH_CX', '')

    async def search_all(self, query: str) -> Dict[str, List[dict]]:
        """Параллельный поиск по всем источникам"""
        if not self.enabled:
            return {}

        tasks = [
            self._github_search(query),
            self._web_search(query),
            self._reddit_search(query),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        out = {}
        for name, res in zip(['github', 'web', 'reddit'], results):
            if isinstance(res, Exception):
                out[name] = []
            else:
                out[name] = res

        # Сохраняем для стигмергии
        ts = int(time.time())
        path = f"{self.context_dir}/search_{ts}.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'query': query, 'ts': ts, 'results': out}, f,
                      ensure_ascii=False, indent=2)
        return out

    async def _github_search(self, query: str) -> List[dict]:
        headers = {'Accept': 'application/vnd.github.v3+json'}
        if self.gh_token:
            headers['Authorization'] = f'token {self.gh_token}'

        params = {'q': f'{query} language:python stars:>10',
                  'per_page': self.gh_n, 'sort': 'stars'}
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get('https://api.github.com/search/repositories',
                                  headers=headers, params=params,
                                  timeout=aiohttp.ClientTimeout(total=5)) as r:
                    data = await r.json()
                    items = data.get('items', [])
                    return [{'title': i['full_name'],
                             'url': i['html_url'],
                             'desc': i.get('description', ''),
                             'stars': i.get('stargazers_count', 0)}
                            for i in items[:self.gh_n]]
        except Exception as e:
            return [{'error': str(e)}]

    async def _web_search(self, query: str) -> List[dict]:
        if not self.google_key or not self.google_cx:
            # Fallback: DuckDuckGo instant answer (без ключа)
            return await self._ddg_search(query)

        params = {'key': self.google_key, 'cx': self.google_cx,
                  'q': query, 'num': self.web_n}
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(
                    'https://www.googleapis.com/customsearch/v1',
                    params=params, timeout=aiohttp.ClientTimeout(total=5)
                ) as r:
                    data = await r.json()
                    items = data.get('items', [])
                    return [{'title': i.get('title', ''),
                             'url': i.get('link', ''),
                             'snippet': i.get('snippet', '')}
                            for i in items]
        except Exception as e:
            return [{'error': str(e)}]

    async def _ddg_search(self, query: str) -> List[dict]:
        """DuckDuckGo instant answer API (без ключа, ограниченный)"""
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(
                    'https://api.duckduckgo.com/',
                    params={'q': query, 'format': 'json', 'no_html': '1'},
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as r:
                    data = await r.json(content_type=None)
                    results = []
                    for item in data.get('RelatedTopics', [])[:self.web_n]:
                        if isinstance(item, dict) and 'Text' in item:
                            results.append({
                                'title': item.get('Text', '')[:100],
                                'url': item.get('FirstURL', ''),
                                'snippet': item.get('Text', '')
                            })
                    return results
        except Exception:
            return []

    async def _reddit_search(self, query: str) -> List[dict]:
        headers = {'User-Agent': 'mycelium-swarm:v8.5 by /u/user'}
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(
                    'https://www.reddit.com/search.json',
                    params={'q': query, 'limit': self.rd_n,
                            'sort': 'relevance', 'type': 'link'},
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as r:
                    data = await r.json()
                    posts = data.get('data', {}).get('children', [])
                    return [{'title': p['data'].get('title', ''),
                             'url': p['data'].get('url', ''),
                             'score': p['data'].get('score', 0),
                             'selftext': p['data'].get('selftext', '')[:500]}
                            for p in posts[:self.rd_n]]
        except Exception as e:
            return [{'error': str(e)}]

    def format_context(self, results: Dict[str, List[dict]],
                       max_tokens: int = 480) -> str:
        """Форматирует результаты поиска в краткий контекст"""
        lines = []
        for source, items in results.items():
            for item in items[:3]:
                if 'error' in item:
                    continue
                title = item.get('title', item.get('desc', ''))[:100]
                url = item.get('url', '')[:80]
                snippet = item.get('snippet', item.get('selftext', ''))[:200]
                if title:
                    lines.append(f"[{source}] {title}")
                    if snippet:
                        lines.append(f"  {snippet}")
                    if url:
                        lines.append(f"  URL: {url}")

        text = '\n'.join(lines)
        # Грубая токенизация: ~4 символа на токен
        max_chars = max_tokens * 4
        return text[:max_chars]

    def read_recent_evolution(self, n: int = 3) -> str:
        """Читает n последних файлов из /swarm/evolution/ как стигмергию"""
        evo_dir = "swarm/evolution"
        if not os.path.exists(evo_dir):
            return ""

        files = sorted(
            [f for f in os.listdir(evo_dir) if f.endswith('.py')],
            key=lambda f: os.path.getmtime(os.path.join(evo_dir, f)),
            reverse=True
        )[:n]

        snippets = []
        now = time.time()
        for fname in files:
            path = os.path.join(evo_dir, fname)
            age_days = (now - os.path.getmtime(path)) / 86400
            # Весовое затухание: w = exp(-age/7d)
            weight = float(__import__('math').exp(-age_days / 7))
            if weight > 0.1:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(500)
                snippets.append(f"# evolution (w={weight:.2f})\n{content}")

        return '\n---\n'.join(snippets)
