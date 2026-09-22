"""Резюмируемая докачка весов Qwen3-0.6B (роль S — синтезатор, §4 задания)."""
import os
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DST_DIR = os.path.join(ROOT, 'models', 'Qwen3-0.6B')
DST = os.path.join(DST_DIR, 'model.safetensors')
URL = 'https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/model.safetensors'

os.makedirs(DST_DIR, exist_ok=True)


def size():
    return os.path.getsize(DST) if os.path.exists(DST) else 0


total = None
for attempt in range(200):
    have = size()
    req = urllib.request.Request(URL, headers={'Range': f'bytes={have}-'})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            cr = r.headers.get('Content-Range')
            if cr:
                total = int(cr.split('/')[-1])
            elif total is None:
                total = int(r.headers.get('Content-Length', 0))
            with open(DST, 'ab') as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
        if total and size() >= total:
            print(f'OK {DST} {size()} bytes', flush=True)
            break
        print(f'partial {size()}/{total} — retry {attempt}', flush=True)
    except Exception as e:
        print(f'retry {attempt} at {size()}/{total}: {type(e).__name__} {str(e)[:100]}',
              flush=True)
    time.sleep(2)
else:
    raise SystemExit(f'failed, have {size()}/{total}')
