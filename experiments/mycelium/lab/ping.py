"""Проверка локального сервера: список моделей + одна генерация."""
import json
import sys
import time
import urllib.request

URL = 'http://127.0.0.1:8077/v1'
model = sys.argv[1] if len(sys.argv) > 1 else 'qwen3-1.7b'

print(urllib.request.urlopen(URL + '/models', timeout=10).read().decode())

payload = {
    'model': model,
    'messages': [
        {'role': 'system', 'content': 'You are a Generator. Produce a diverse, creative response.'},
        {'role': 'user', 'content': 'Write a Python function add(a, b) that returns a + b. Return one code block.'},
    ],
    'temperature': 0.7,
    'max_tokens': 120,
}
req = urllib.request.Request(URL + '/chat/completions',
                             data=json.dumps(payload).encode(),
                             headers={'Content-Type': 'application/json'})
t0 = time.monotonic()
r = json.load(urllib.request.urlopen(req, timeout=600))
print(f"{time.monotonic()-t0:.1f}s", r['usage'])
print(r['choices'][0]['message']['content'][:800])
