"""
lab/llm_server.py — минимальный OpenAI-совместимый сервер для локальных моделей.

Не часть Mycelium. Существует только чтобы provider='local' в models.yaml
получил живой endpoint /v1/chat/completions вместо API.

ФАЗА 1 (офлайн). Два бэкенда, GGUF основной:

  gguf — llama-cpp-python, полный offload на GPU, модели резидентны.
         Замерено на GTX 1660 SUPER: 75–191 ток/с, загрузка 0.4–1.3 с.
  hf   — transformers + bitsandbytes NF4, для того, что скачано в
         safetensors (Qwen3-1.7B). Замерено: ~10 ток/с, загрузка 6–31 с.

Разница в скорости 7–19x, поэтому GGUF предпочтителен всегда, а hf-модель в
пуле становится узким местом шага. Если модель есть в обоих форматах — брать
GGUF.

Свопа нет ни в одном бэкенде: в 4 битах пул из пяти малых моделей занимает
~3.9 ГиБ и помещается в 6144 МиБ целиком. Раньше веса жили в CPU RAM в fp16 и
переезжали на GPU по одной — 1.0–2.8 с на переключение касты.

Две вещи, продиктованные железом и сохранённые из прежней версии:

1. ОСТАНОВКА ПО ЗАКРЫТОМУ КОД-БЛОКУ. Как только в ответе появился полный
   ```...``` — генерация прекращается. Без этого модель добивает лимит прозой,
   а на лимите код обрывается на полуслове и q_env даёт 0.2 за SyntaxError,
   то есть измеряется длина ответа, а не правильность.
2. МИКРО-БАТЧИНГ для hf-бэкенда: оркестратор шлёт вызовы касты параллельно.
   Для gguf не нужен — там 1 реплика на модель и скорость на порядок выше.

Автоматически отсекаются две категории, каждая уже дала тихую порчу замера:

  * MoE в квантованном режиме — bitsandbytes подменяет только nn.Linear, а
    эксперты в transformers 5.x это слитые 3D-параметры. Модель молча
    загрузится неквантованной. Проверено в trace-probe/HONESTY.md §7.
    Так отсеиваются granite-3.1-1b-a400m-base (granitemoe!) и OLMoE.
  * БАЗОВЫЕ модели без chat-шаблона — они повторяют промпт вместо ответа.
    Так отсеялся SmolLM2-360M.Q5_K_M.gguf: general.name = "SmolLM2 360M",
    шаблона нет, llama.cpp подставил дефолтный Llama-2 [INST], и модель
    вернула его как текст.

Запуск:
    python lab/llm_server.py --port 8077
    python lab/llm_server.py --preload            # прогреть весь пул и VRAM
    python lab/llm_server.py --list               # что видно, без загрузки
    python lab/llm_server.py --fp16               # прежний режим со свопом
"""

import argparse
import json
import os
import queue
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(ROOT, 'models')

# Короткие имена → то, что стоит в models.yaml как model_name.
ALIASES = {
    'qwen3-1.7b': 'Qwen3-1.7B',
    'qwen3-0.6b': 'Qwen3-0.6B',
}

QUANTIZE_4BIT = True
VRAM_BUDGET_GIB = 5.2          # 6144 МиБ минус дисплей/драйвер
MAX_RESIDENT = 4               # сколько моделей держать в VRAM; остальные — LRU
BATCH_MAX = 3
BATCH_GRACE_S = 0.6
MAX_INPUT_TOKENS = 3072
N_CTX = 2048

MOE_MARKERS = ('num_experts', 'num_local_experts', 'n_routed_experts',
               'moe_intermediate_size', 'num_experts_per_tok')

_registry = {}                 # name -> {'backend','path','note'}
_skipped = []
_loaded = {}                   # name -> backend-specific handle
_last_used = {}                # name -> monotonic, для LRU
_locks = {}                    # name -> threading.Lock
_on_gpu = None                 # только для fp16-режима
_jobs = queue.Queue()

STATS = {'calls': 0, 'batches': 0, 'swaps': 0, 'prompt_tokens': 0,
         'completion_tokens': 0, 'gen_seconds': 0.0, 'swap_seconds': 0.0,
         'stopped_on_fence': 0, 'evictions': 0, 'reloads': 0}


def _log(msg):
    print(f"[llm_server {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── обнаружение моделей ─────────────────────────────────────────────────────

def _is_moe(cfg_path):
    try:
        with open(cfg_path, encoding='utf-8') as f:
            cfg = json.load(f)
    except Exception:
        return False, ''
    hit = [k for k in MOE_MARKERS if k in cfg]
    return (True, f"{cfg.get('model_type', '?')} ({', '.join(hit)})") if hit else (False, '')


def _gguf_has_chat_template(path):
    """Базовая модель без шаблона повторяет промпт. Дешевле проверить здесь,
    чем потом искать, почему одна каста выдаёт мусор."""
    try:
        from lab.gguf_boot import ensure_llama_cpp
        ensure_llama_cpp()
        from llama_cpp import Llama
        probe = Llama(model_path=path, n_gpu_layers=0, n_ctx=64, verbose=False)
        tmpl = probe.metadata.get('tokenizer.chat_template', '') or ''
        name = probe.metadata.get('general.name', '?')
        del probe
        return bool(tmpl.strip()), name
    except Exception as exc:
        _log(f"не удалось прочитать метаданные {os.path.basename(path)}: {exc}")
        return True, '?'          # не блокируем из-за сбоя проверки


def discover(exclude_moe=True, check_templates=True):
    reg, skipped = {}, []
    if not os.path.isdir(MODELS_DIR):
        return reg, skipped

    # 1) GGUF — основной путь
    for dirpath, _dirs, files in os.walk(MODELS_DIR):
        for fn in sorted(files):
            if not fn.lower().endswith('.gguf'):
                continue
            path = os.path.join(dirpath, fn)
            key = os.path.splitext(fn)[0].lower()
            if check_templates:
                ok, gname = _gguf_has_chat_template(path)
                if not ok:
                    skipped.append(
                        f"{fn}: базовая модель без chat-шаблона "
                        f"(general.name='{gname}') — повторяет промпт")
                    continue
            reg[key] = {'backend': 'gguf', 'path': path, 'note': ''}

    # 2) safetensors — то, чего нет в GGUF
    for entry in sorted(os.listdir(MODELS_DIR)):
        path = os.path.join(MODELS_DIR, entry)
        cfg_path = os.path.join(path, 'config.json')
        if not os.path.isdir(path) or not os.path.exists(cfg_path):
            continue
        moe, why = _is_moe(cfg_path)
        if moe and exclude_moe:
            skipped.append(f"{entry}: MoE {why} — bitsandbytes не квантует "
                           f"слитых экспертов (HONESTY.md §7)")
            continue
        reg[entry.lower()] = {'backend': 'hf', 'path': path, 'note': ''}

    for alias, dirname in ALIASES.items():
        path = os.path.join(MODELS_DIR, dirname)
        cfg_path = os.path.join(path, 'config.json')
        if not os.path.exists(cfg_path):
            continue
        if _is_moe(cfg_path)[0] and exclude_moe:
            continue
        reg[alias] = {'backend': 'hf', 'path': path, 'note': f'alias->{dirname}'}
    return reg, skipped


# ── загрузка ────────────────────────────────────────────────────────────────

def _vram_used_gib():
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return (total - free) / 2 ** 30
    except Exception:
        pass
    return 0.0


def _evict_lru(need_for=''):
    """Выгрузить давно не использованную модель, освободив VRAM.

    Пять моделей пула в 4 битах дают 5.92 ГиБ при карте 6144 МиБ — не влезают.
    Но выгрузка в GGUF почти бесплатна: повторная загрузка 0.4–1.4 с против
    30–35 с у прежней fp16-схемы, а веса остаются в дисковом кэше ОС, то есть
    фактически в оперативной памяти (её здесь 24 ГБ).

    Поэтому вместо «держать всё в VRAM» — LRU: резидентны последние
    MAX_RESIDENT, остальные подгружаются по требованию. Это не возврат к
    старому свопу: там модель ПЕРЕМЕЩАЛАСЬ между CPU и GPU за 1.0–2.8 с и
    держала копию весов в RAM всё время.
    """
    global _on_gpu
    if MAX_RESIDENT <= 0:
        return
    while len(_loaded) >= MAX_RESIDENT:
        victim = min((n for n in _loaded), key=lambda n: _last_used.get(n, 0.0))
        if victim == need_for:
            break
        handle = _loaded.pop(victim, None)
        _locks.pop(victim, None)
        _last_used.pop(victim, None)
        del handle
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        STATS['evictions'] += 1
        if _on_gpu == victim:
            _on_gpu = None
        _log(f"выгружено {victim} (LRU), резидентно {_vram_used_gib():.2f} GiB")


def _ensure_loaded(name):
    if name in _loaded:
        _last_used[name] = time.monotonic()
        return _loaded[name]
    if name not in _registry:
        raise KeyError(f"модель '{name}' не найдена. Доступны: "
                       f"{sorted(_registry)}")
    _evict_lru(need_for=name)
    spec = _registry[name]
    before = _vram_used_gib()
    t0 = time.monotonic()

    if spec['backend'] == 'gguf':
        from lab.gguf_boot import ensure_llama_cpp
        ensure_llama_cpp()
        from llama_cpp import Llama
        _log(f"loading {name} (gguf, offload=-1) ...")
        handle = Llama(model_path=spec['path'], n_gpu_layers=-1,
                       n_ctx=N_CTX, verbose=False, seed=-1)
    else:
        import torch
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                  BitsAndBytesConfig)
        quant = QUANTIZE_4BIT and torch.cuda.is_available()
        _log(f"loading {name} (hf, {'NF4' if quant else 'fp16'}) ...")
        tok = AutoTokenizer.from_pretrained(spec['path'])
        if quant:
            qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                    bnb_4bit_compute_dtype=torch.float16,
                                    bnb_4bit_use_double_quant=True)
            model = AutoModelForCausalLM.from_pretrained(
                spec['path'], quantization_config=qc, device_map={'': 0})
        else:
            model = AutoModelForCausalLM.from_pretrained(
                spec['path'], dtype=torch.float16)
        model.eval()
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = 'left'
        handle = (tok, model)

    _loaded[name] = handle
    _locks[name] = threading.Lock()
    _last_used[name] = time.monotonic()
    STATS['reloads'] += 1
    dt = time.monotonic() - t0
    after = _vram_used_gib()
    _log(f"loaded {name} in {dt:.1f}s  +{after - before:.2f} GiB  "
         f"(резидентно {after:.2f} GiB)")
    if after > VRAM_BUDGET_GIB:
        _log(f"ВНИМАНИЕ: {after:.2f} GiB при бюджете {VRAM_BUDGET_GIB} GiB — "
             f"возможен OOM. Уберите модель из пула.")
    return handle


def _to_gpu(name):
    """Своп нужен только прежнему fp16-режиму без квантования."""
    global _on_gpu
    if QUANTIZE_4BIT or _registry.get(name, {}).get('backend') == 'gguf':
        _on_gpu = name
        return
    import torch
    if not torch.cuda.is_available() or _on_gpu == name:
        return
    t0 = time.monotonic()
    if _on_gpu is not None and _registry[_on_gpu]['backend'] == 'hf':
        _loaded[_on_gpu][1].to('cpu')
        torch.cuda.empty_cache()
    _loaded[name][1].to('cuda')
    _on_gpu = name
    dt = time.monotonic() - t0
    STATS['swaps'] += 1
    STATS['swap_seconds'] += dt
    _log(f"swap -> {name} ({dt:.1f}s)")


# ── генерация ───────────────────────────────────────────────────────────────

_THINK_MODELS = ('qwen3',)          # семейства с reasoning-режимом


def _wants_no_think(name):
    return any(t in name.lower() for t in _THINK_MODELS)


def _apply_no_think(name, messages):
    """Qwen3 по умолчанию уходит в <think> и выедает весь лимит на рассуждение.

    Замерено на Qwen3-1.7B-Q4_0: без переключателя 384 токена целиком ушли в
    рассуждение и ответа не осталось; с '/no_think' — 60 токенов и готовый
    код. chat_template_kwargs={'enable_thinking': False} в llama-cpp-python
    0.3.34 не поддерживается, поэтому используется мягкий переключатель
    самой модели.

    Это же делал прежний прогон через transformers (enable_thinking=False),
    так что режим сопоставим со старым baseline.
    """
    if not _wants_no_think(name):
        return messages
    out = [dict(m) for m in messages]
    for m in out:
        if m.get('role') == 'system':
            if '/no_think' not in (m.get('content') or ''):
                m['content'] = (m.get('content') or '') + ' /no_think'
            return out
    return [{'role': 'system', 'content': '/no_think'}] + out


def _strip_think(text):
    """Снять <think>...</think> ДО обрезки по фенсу.

    Порядок важен: рассуждение может содержать собственные ``` , и тогда
    _truncate_after_fence обрежет ответ внутри рассуждения, вернув мусор.

    НЕЗАКРЫТЫЙ <think> — отдельный случай, и наивная обработка тут уже
    обнулила целую модель. При '/no_think' Qwen3-1.7B-Q4_0 выдаёт:

        '<think>\\n\\n```python\\ndef find_shared_elements(...): ...\\n```'

    то есть тег ОТКРЫТ, рассуждения нет, дальше сразу ответ, закрывающего
    тега не будет никогда. Первая версия резала по r'<think>.*$' и съедала
    ответ целиком: 29 из 30 задач дали пустой текст, q_env упала до 0.42, и
    вывод был бы «Qwen3 не умеет в код» — при том, что код правильный.

    Поэтому: закрытые пары убираем всегда; при незакрытом теге смотрим, есть
    ли за ним признаки ответа (код-фенс или def). Есть — снимаем только сам
    тег. Нет — это оборванное на лимите рассуждение, и полезного там нет.
    """
    import re
    if '<think>' not in text:
        return text
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    if '<think>' in text:
        head, _, tail = text.partition('<think>')
        looks_like_answer = ('```' in tail) or ('def ' in tail) or ('class ' in tail)
        text = head + (tail if looks_like_answer else '')
    return text.strip()


def _truncate_after_fence(text):
    """Обрезать по закрывающему ``` первого код-блока (если он закрыт)."""
    first = text.find('```')
    if first < 0:
        return text, False
    second = text.find('```', first + 3)
    if second < 0:
        return text, False
    return text[:second + 3], True


def _build_prompt_hf(tok, messages):
    if getattr(tok, 'chat_template', None):
        try:
            return tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)          # Qwen3: без <think>
        except TypeError:
            return tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
    parts = []
    for m in messages:
        tag = {'system': 'Instructions', 'user': 'Task',
               'assistant': 'Answer'}.get(m.get('role', 'user'), m.get('role'))
        parts.append(f"### {tag}:\n{m.get('content','')}")
    parts.append("### Answer:\n")
    return '\n\n'.join(parts)


def _run_gguf(job):
    llm = _ensure_loaded(job.model)
    with _locks[job.model]:
        t0 = time.monotonic()
        out = llm.create_chat_completion(
            messages=_apply_no_think(job.model, job.messages),
            max_tokens=job.max_tokens,
            temperature=job.temperature,
            top_p=0.95,
            # Останов по закрывающему фенсу делаем пост-обрезкой: stop-строки
            # сработали бы на ОТКРЫВАЮЩЕМ ``` и обрубили ответ до кода.
        )
        dt = time.monotonic() - t0
    raw = out['choices'][0]['message']['content'] or ''
    text, fenced = _truncate_after_fence(_strip_think(raw))
    if fenced:
        STATS['stopped_on_fence'] += 1
    u = out.get('usage', {})
    n_in = int(u.get('prompt_tokens', 0))
    n_out = int(u.get('completion_tokens', len(text.split())))
    STATS['gen_seconds'] += dt
    STATS['prompt_tokens'] += n_in
    STATS['completion_tokens'] += n_out
    return text, n_in, n_out


def _run_hf_batch(jobs):
    import torch
    from transformers import StoppingCriteria

    name = jobs[0].model
    tok, model = _ensure_loaded(name)
    _to_gpu(name)

    class FenceStopping(StoppingCriteria):
        def __init__(self, n_in, batch_size, eos_ids):
            self.n_in = n_in
            self.finished = [False] * batch_size
            self.eos_ids = set(i for i in eos_ids if i is not None)

        def __call__(self, input_ids, scores, **kwargs):
            for i in range(input_ids.shape[0]):
                if self.finished[i]:
                    continue
                new = input_ids[i][self.n_in:]
                if new.numel() and int(new[-1]) in self.eos_ids:
                    self.finished[i] = True
                    continue
                if int(new.numel()) % 8:
                    continue
                if tok.decode(new, skip_special_tokens=True).count('```') >= 2:
                    self.finished[i] = True
            return all(self.finished)

    texts = [_build_prompt_hf(tok, j.messages) for j in jobs]
    enc = tok(texts, return_tensors='pt', padding=True, truncation=True,
              max_length=MAX_INPUT_TOKENS)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    n_in = int(enc['input_ids'].shape[1])
    max_new = max(j.max_tokens for j in jobs)
    temperature = max(j.temperature for j in jobs)

    eos_ids = [tok.eos_token_id]
    gc_eos = getattr(model.generation_config, 'eos_token_id', None)
    if gc_eos is not None:
        eos_ids += gc_eos if isinstance(gc_eos, list) else [gc_eos]

    kwargs = dict(max_new_tokens=max_new, pad_token_id=tok.pad_token_id,
                  stopping_criteria=[FenceStopping(n_in, len(jobs), eos_ids)])
    if temperature > 0.01:
        kwargs.update(do_sample=True, temperature=temperature, top_p=0.95, top_k=50)
    else:
        kwargs.update(do_sample=False)

    t0 = time.monotonic()
    with torch.inference_mode():
        out = model.generate(**enc, **kwargs)
    STATS['gen_seconds'] += time.monotonic() - t0
    STATS['batches'] += 1

    results = []
    for i, _j in enumerate(jobs):
        raw = tok.decode(out[i][n_in:], skip_special_tokens=True)
        text, fenced = _truncate_after_fence(_strip_think(raw))
        if fenced:
            STATS['stopped_on_fence'] += 1
        n_out = len(tok(text, add_special_tokens=False)['input_ids'])
        STATS['prompt_tokens'] += n_in
        STATS['completion_tokens'] += n_out
        results.append((text, n_in, n_out))
    return results


# ── диспетчер ───────────────────────────────────────────────────────────────

class Job:
    __slots__ = ('model', 'messages', 'temperature', 'max_tokens',
                 'done', 'result', 'error')

    def __init__(self, model, messages, temperature, max_tokens):
        self.model = model
        self.messages = messages
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.done = threading.Event()
        self.result = None
        self.error = None


def _dispatcher():
    while True:
        job = _jobs.get()
        batch = [job]
        backend = _registry.get(job.model, {}).get('backend', 'gguf')
        if backend == 'hf':
            # собрать соседей по касте: у hf батч почти бесплатен
            deadline = time.monotonic() + BATCH_GRACE_S
            while len(batch) < BATCH_MAX:
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                try:
                    nxt = _jobs.get(timeout=left)
                except queue.Empty:
                    break
                if nxt.model == job.model:
                    batch.append(nxt)
                else:
                    _jobs.put(nxt)
                    break
        try:
            STATS['calls'] += len(batch)
            if backend == 'gguf':
                for j in batch:
                    j.result = _run_gguf(j)
            else:
                for j, res in zip(batch, _run_hf_batch(batch)):
                    j.result = res
        except Exception as exc:
            traceback.print_exc()
            for j in batch:
                j.error = f"{type(exc).__name__}: {exc}"
        finally:
            for j in batch:
                j.done.set()


# ── HTTP ────────────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *a):
        pass

    def _send(self, code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip('/').endswith('/models'):
            self._send(200, {'object': 'list', 'data': [
                {'id': k, 'object': 'model', 'backend': v['backend']}
                for k, v in sorted(_registry.items())]})
        elif self.path.rstrip('/').endswith('/stats'):
            self._send(200, dict(STATS, vram_gib=round(_vram_used_gib(), 2),
                                 loaded=sorted(_loaded)))
        else:
            self._send(404, {'error': {'message': 'not found'}})

    def do_POST(self):
        n = int(self.headers.get('Content-Length', 0))
        try:
            req = json.loads(self.rfile.read(n) or b'{}')
        except Exception as exc:
            self._send(400, {'error': {'message': f'bad json: {exc}'}})
            return

        name = str(req.get('model', '')).lower()
        if name not in _registry:
            self._send(404, {'error': {
                'message': f"unknown model '{name}'; available: {sorted(_registry)}"}})
            return

        job = Job(name, req.get('messages', []),
                  req.get('temperature', 0.7), req.get('max_tokens', 512))
        _jobs.put(job)
        job.done.wait()
        if job.error:
            self._send(500, {'error': {'message': job.error}})
            return

        text, n_in, n_out = job.result
        self._send(200, {
            'id': f'local-{int(time.time()*1000)}',
            'object': 'chat.completion',
            'model': name,
            'choices': [{'index': 0, 'finish_reason': 'stop',
                         'message': {'role': 'assistant', 'content': text}}],
            'usage': {'prompt_tokens': n_in, 'completion_tokens': n_out,
                      'total_tokens': n_in + n_out},
        })


def main():
    global QUANTIZE_4BIT, _registry, _skipped
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8077)
    ap.add_argument('--preload', action='store_true')
    ap.add_argument('--list', action='store_true',
                    help='показать пул и выйти (без загрузки весов)')
    ap.add_argument('--fp16', action='store_true',
                    help='прежний режим hf: fp16 + своп. Только для '
                         'воспроизведения старого baseline')
    ap.add_argument('--only', default='')
    ap.add_argument('--max-resident', type=int, default=MAX_RESIDENT,
                    help='сколько моделей держать в VRAM одновременно; '
                         'остальные подгружаются по требованию (LRU). '
                         'Перезагрузка GGUF 0.4-1.4 с, веса остаются в '
                         'дисковом кэше ОС. 0 = без ограничения')
    ap.add_argument('--n-ctx', type=int, default=N_CTX,
                    help='контекст на модель. KV-кэш линеен по нему, и при '
                         'пяти резидентных моделях это главный рычаг VRAM: '
                         '2048 даёт 6.00 GiB (не влезает в 6144 МиБ), '
                         '1024 — умещается. Для MBPP 1024 с запасом.')
    args = ap.parse_args()

    if args.fp16:
        QUANTIZE_4BIT = False
    globals()['N_CTX'] = int(args.n_ctx)
    globals()['MAX_RESIDENT'] = int(args.max_resident)

    _registry, _skipped = discover(exclude_moe=QUANTIZE_4BIT,
                                   check_templates=True)

    _log(f"режим hf={'fp16+swap' if not QUANTIZE_4BIT else 'NF4'}, "
         f"gguf=offload -1, n_ctx={N_CTX}, max_resident={MAX_RESIDENT}")
    for k, v in sorted(_registry.items()):
        _log(f"  [{v['backend']:4}] {k}")
    for s in _skipped:
        _log(f"  ОТСЕЯНО {s}")
    if not _registry:
        _log(f"ПУСТО: в {MODELS_DIR} нет ни .gguf, ни каталогов с config.json")
    if args.list:
        return

    if args.preload:
        names = [n.strip() for n in args.only.split(',') if n.strip()] or list(_registry)
        for k in names:
            _ensure_loaded(k)
        _log(f"итого резидентно {_vram_used_gib():.2f} GiB "
             f"из бюджета {VRAM_BUDGET_GIB} GiB")

    threading.Thread(target=_dispatcher, daemon=True).start()
    srv = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    _log(f"listening on http://127.0.0.1:{args.port}/v1")
    srv.serve_forever()


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
