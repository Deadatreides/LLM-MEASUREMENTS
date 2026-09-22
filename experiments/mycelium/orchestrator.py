"""
orchestrator.py — Главный оркестратор роя
Σ_v8.5 «Мицелий»

Полный шаг согласно спецификации v8.5 §12:
t0:  embed (47ms GPU / CPU)
t1:  FAISS + PPR (4.2ms + 13.4ms)
t2:  async поиск (старт, не блокирует)
t3:  PID → K_ACT
t4:  гиперцикл → квоты каст
t5:  Thompson → выбор моделей
t6:  G: 3 API вызова параллельно
t7:  Value: отсев
t8:  C: 2 API вызова
t9:  ждём поиск + компрессор
t10: S: 1 API вызов
t11: OpenCode тест
t12: E_total, обновление памяти
t13: git commit при E>0.8
"""

import asyncio
import aiohttp
import os
import time
import subprocess
import json
import yaml
import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from dotenv import load_dotenv

from core.bandit import SwarmBandit
from core.chaos import ChaosController
from core.hypercycle import Hypercycle, AgentEnergy
from core.blackboard import SemanticBlackboard, ThermodynamicState
from core.memory import Memory
from core.scorer import QualityScorer, EmergenceMetric, semantic_diversity
from core.acceptance import select as acceptance_select
from core.acceptance import trace_of as acceptance_trace
from tools.search import SearchModule

load_dotenv()
from core.logging_setup import setup_logging
setup_logging()
log = logging.getLogger('swarm')



@dataclass
class RuntimeStepState:
    H_before: float = 0.0
    H_after: float = 0.0
    delta_I: float = 0.0
    phi: float = 0.0
    e_total: float = 0.0
    lambda_L: float = 0.0
    q_env: float = 0.5


def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    if max_words <= 20:
        return ' '.join(words[-max_words:])
    head = max_words // 3
    tail = max_words - head - 1
    return ' '.join(words[:head] + ['...[truncated]...'] + words[-tail:])


def _fit_model_context(model_cfg: dict, prompt: str, system: str) -> tuple:
    ctx = int(model_cfg.get('ctx', 8192))
    max_tokens = int(model_cfg.get('max_tokens', 2048))
    max_tokens = min(max_tokens, max(64, ctx // 4), 2048)
    input_budget = max(64, ctx - max_tokens - 32)
    system_budget = min(max(32, input_budget // 4), len(system.split()))
    prompt_budget = max(32, input_budget - system_budget)
    return (_truncate_words(prompt, prompt_budget),
            _truncate_words(system, system_budget),
            max_tokens)


def load_config():
    with open('config/settings.yaml', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    with open('config/models.yaml', encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    cfg['models'] = raw['models']
    return cfg


PROVIDER_KEYS = {
    'groq': 'GROQ_API_KEY',
    'deepseek': 'DEEPSEEK_API_KEY',
    'cerebras': 'CEREBRAS_API_KEY',
    'sambanova': 'SAMBANOVA_API_KEY',
    'openrouter': 'OPENROUTER_API_KEY',
    'google': 'GOOGLE_API_KEY',
    'cloudflare': 'CLOUDFLARE_API_KEY',
    'local': None,
}


async def call_model(session: aiohttp.ClientSession, model_cfg: dict,
                     prompt: str, system: str, temperature: float,
                     timeout: int = 30) -> dict:
    """Один API вызов с обработкой ошибок, экспоненциальным backoff для transient ошибок и отслеживанием fatal ошибок.

    [Ф1.1] timeout берётся из models.yaml (`timeout:` у модели), если задан.
    Дефолт 30 с разумен для API-провайдера и абсурден для локальной 1.7B на
    ~10 ток/с: ответ идёт 20–90 с, и раньше всё падало по таймауту с четырьмя
    ретраями (по 130 с на провалившийся вызов). Значение поднимал харнесс
    снаружи (lab/run_experiment.py), то есть тестировалась конфигурация,
    отличная от штатной. Теперь оно живёт в конфиге рядом с моделью.
    """
    import logging
    plog = logging.getLogger('providers')

    timeout = int(model_cfg.get('timeout') or timeout)
    start_time = time.monotonic()
    provider = model_cfg['provider']
    endpoint = model_cfg['endpoint']
    model_name = model_cfg['model_name']

    # Cloudflare: подставить account_id
    if provider == 'cloudflare':
        account_id = os.getenv('CLOUDFLARE_ACCOUNT_ID', '')
        endpoint = endpoint.replace('{account_id}', account_id)

    headers = {'Content-Type': 'application/json'}

    # [Ф1.2] Локальным считается ЛЮБОЙ provider с префиксом 'local'.
    # Раньше сравнивалось с буквальным 'local', и после разделения провайдеров
    # по моделям (local_coder15, local_smol17, ...) — а оно нужно, чтобы
    # bandit.max_per_provider снова работал — каждый вызов стал требовать
    # API-ключ и падать с fatal. Прогон при этом завершался «успешно»: 0
    # токенов, q_env=0.4 на всех задачах, status=ok. Тот же класс тихого нуля,
    # что pytest-FileNotFoundError -> 0.0 и таймаут 30 с на локальной модели.
    is_local = provider == 'local' or provider.startswith('local_')

    # Check key existence
    key_val = ""
    if provider == 'google':
        key_val = os.getenv(PROVIDER_KEYS[provider], '')
        headers['x-goog-api-key'] = key_val
    elif is_local:
        pass  # локальный сервер без ключа
    else:
        key_name = PROVIDER_KEYS.get(provider, '')
        if key_name:
            key_val = os.getenv(key_name, "")
            headers['Authorization'] = f'Bearer {key_val}'

    if provider == 'openrouter':
        headers['HTTP-Referer'] = 'https://mycelium-swarm.local'

    # Check if a non-local API key is clearly missing
    if not is_local and not key_val.strip():
        plog.error(f"Missing API key for provider {provider}. Marking call as FATAL.")
        return {
            'id': model_cfg['id'],
            'ok': False,
            'fatal': True,
            'error': 'missing_api_key',
            'latency': 0.0
        }

    prompt, system, max_tokens = _fit_model_context(model_cfg, prompt, system)

    # [Ш0.1] temp_offset — источник внутрикастового разнообразия.
    # Реплики одной модели с одинаковой температурой дают неразличимые
    # ответы: в прошлом прогоне у 25 из 30 задач q_env всех кандидатов
    # совпадала, D_sem = 0.072. Разнообразие «внутри касты идёт от
    # сэмплирования» на практике не работало, потому что сэмплирование при
    # одной температуре и одном промпте почти детерминировано.
    # Замерено lab/bench_pool.py: t=0.55 против 0.85 меняет исход в 9–13
    # задачах из 30 у каждой модели пула.
    eff_temp = float(np.clip(temperature + float(model_cfg.get('temp_offset', 0.0)),
                             0.05, 1.5))
    payload = {
        'model': model_name,
        'messages': [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': prompt}
        ],
        'temperature': eff_temp,
        'max_tokens': max_tokens,
    }

    max_attempts = 4
    backoff = 1.0

    for attempt in range(1, max_attempts + 1):
        try:
            start = time.monotonic()
            async with session.post(
                f"{endpoint}/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as resp:
                latency = time.monotonic() - start
                
                # Check for transient Server Error (5xx) or Rate Limit (429)
                if resp.status == 200:
                    data = await resp.json()
                    
                    # OpenRouter error checking within success response
                    if isinstance(data, dict) and 'error' in data:
                        err_data = data['error']
                        err_msg = err_data.get('message', '')
                        err_code = err_data.get('code', 0)
                        
                        # Detect fatal issues in message
                        fatal_keywords = ['insufficient_balance', 'auth', 'key', 'quota', 'limit_exceeded']
                        is_fatal = any(kw in str(err_msg).lower() or kw in str(err_code).lower() for kw in fatal_keywords)
                        
                        plog.warning(f"Error payload from {model_cfg['id']}: {err_msg} | fatal={is_fatal}")
                        
                        if is_fatal:
                            return {
                                'id': model_cfg['id'],
                                'ok': False,
                                'fatal': True,
                                'error': f"api_err_{err_code}",
                                'latency': latency
                            }
                        
                        # Transient error inside JSON -> trigger retry or cooldown
                        if attempt < max_attempts:
                            await asyncio.sleep(backoff)
                            backoff *= 2.0
                            continue
                            
                        return {
                            'id': model_cfg['id'],
                            'ok': False,
                            'error': f"api_err_{err_code}",
                            'latency': latency
                        }
                    
                    text = data['choices'][0]['message']['content']
                    tokens_out = data.get('usage', {}).get('completion_tokens', len(text.split()))
                    return {
                        'id': model_cfg['id'],
                        'text': text,
                        'tokens_out': tokens_out,
                        'latency': latency,
                        'ok': True
                    }
                    
                elif resp.status == 429:
                    plog.warning(f"Rate limit (429) on {model_cfg['id']} | Attempt {attempt}/{max_attempts}")
                    if attempt < max_attempts:
                        await asyncio.sleep(backoff)
                        backoff *= 2.0
                        continue
                    return {
                        'id': model_cfg['id'],
                        'ok': False,
                        'error': 'rate_limit',
                        'latency': latency
                    }
                    
                elif resp.status in [500, 502, 503, 504]:
                    plog.warning(f"Transient server error ({resp.status}) on {model_cfg['id']} | Attempt {attempt}/{max_attempts}")
                    if attempt < max_attempts:
                        await asyncio.sleep(backoff)
                        backoff *= 2.0
                        continue
                    return {
                        'id': model_cfg['id'],
                        'ok': False,
                        'error': f"server_error_{resp.status}",
                        'latency': latency
                    }
                    
                else:
                    # Non-transient errors (400, 401, 402, 403, 404, etc.)
                    text = await resp.text()
                    plog.error(f"Fatal HTTP Error {resp.status} from {model_cfg['id']}: {text[:150]}")
                    
                    # Detect balance / auth errors specifically
                    is_fatal = resp.status in [400, 401, 402, 403, 404]
                    return {
                        'id': model_cfg['id'],
                        'ok': False,
                        'fatal': is_fatal,
                        'error': f"http_{resp.status}",
                        'latency': latency
                    }
                    
        except asyncio.TimeoutError:
            plog.warning(f"Timeout Error on {model_cfg['id']} | Attempt {attempt}/{max_attempts}")
            if attempt < max_attempts:
                await asyncio.sleep(backoff)
                backoff *= 2.0
                continue
            return {
                'id': model_cfg['id'],
                'ok': False,
                'error': 'timeout',
                'latency': float(timeout)
            }
            
        except Exception as e:
            plog.warning(f"Connection Exception on {model_cfg['id']}: {e} | Attempt {attempt}/{max_attempts}")
            if attempt < max_attempts:
                await asyncio.sleep(backoff)
                backoff *= 2.0
                continue
            return {
                'id': model_cfg['id'],
                'ok': False,
                'error': str(e),
                'latency': 0.0
            }

    return {
        'id': model_cfg['id'],
        'ok': False,
        'error': 'max_retries_exceeded',
        'latency': time.monotonic() - start_time
    }



class MyceliumOrchestrator:
    """Главный оркестратор. Запускается из main.py или OpenCode."""

    def __init__(self):
        self.cfg = load_config()
        os.makedirs('logs', exist_ok=True)
        os.makedirs('swarm/tasks', exist_ok=True)
        os.makedirs('swarm/pool', exist_ok=True)
        os.makedirs('swarm/context', exist_ok=True)
        os.makedirs('swarm/evolution', exist_ok=True)
        os.makedirs('data', exist_ok=True)

        self.models_by_id = {m['id']: m for m in self.cfg['models']}
        self.bandit = SwarmBandit(self.cfg, self.cfg['models'])
        self.chaos = ChaosController(self.cfg)
        self.hypercycle = Hypercycle(self.cfg)
        self.memory = Memory(self.cfg)
        self.blackboard = SemanticBlackboard(self.cfg)
        self.scorer = QualityScorer(self.cfg)
        self.emergence = EmergenceMetric(self.cfg)
        self.search = SearchModule(self.cfg)
        self.energy_model = AgentEnergy(self.cfg)
        # [TASK_ACCEPTANCE §B] Проверяющий для приёмки по исполнению.
        # None по умолчанию: в продакшене эталонных тестов нет, и без
        # проверяющего поведение остаётся прежним (берётся s_ok[0]).
        # Лаборатория подставляет core.acceptance.SandboxProbe.
        self.acceptance_probe = None
        self._last_gate = None

        self.step = 0
        self.budget_tokens = self.cfg['budget']['tokens_per_hour']
        self.tokens_used = 0
        self.token_refill_rate = self.cfg['budget']['refill_per_sec']
        self._last_refill = time.monotonic()
        self._last_energy_raw = sum(a.energy for a in self.bandit.agents.values())
        self._reward_baseline = 0.0

        self.bandit.load('data/bandit_state.json')

        # FIX 4: проверить API ключи при старте — предупредить о пустых
        _missing = []
        for provider, env_key in PROVIDER_KEYS.items():
            if env_key and not os.getenv(env_key, '').strip():
                _missing.append(f"{provider}({env_key})")
        if _missing:
            log.warning("API keys missing — providers will fail: %s", ', '.join(_missing))
        self._prev_e_total: float = 0.0
        self._last_thermo: ThermodynamicState = ThermodynamicState()
        self._r_mem: float = 0.0   # retrieval influence scalar (связь #1)
        log.info("Mycelium v8.9 initialized.")

    def _step_budget(self) -> float:
        """[Ш1.1] Бюджет токенов НА ШАГ для cost-члена E_total.

        Дефект R3: было `self.budget_tokens / 3600` = 800000/3600 = 222.2 —
        это токенов в СЕКУНДУ (что подтверждает соседний budget.refill_per_sec
        = 222.2), подставленное как бюджет ШАГА. Шаг тратит ~894 токена за
        ~83 с, то есть отношение cost/budget = 4.0 и min(...,1) == 1.0 в 197
        из 198 шагов. Член вырождался в константу −0.182·(K/9) = −0.121 —
        86.6 % средней E_total, и нёс ноль информации.

        Каскад от этой одной строки: E_total всегда отрицательна, поэтому
        порог add_flow (0.7) не брался ни разу (ноль рёбер Physarum),
        is_evolution_worthy (0.8) не срабатывал никогда, а
        LeakyIntegrator.update(max(0, e_total)) всегда получал 0.
        """
        fx = self.cfg.get('fix', {}) or {}
        if fx.get('budget_per_step'):
            return float(fx.get('budget_tokens_per_step', 18000))
        return self.budget_tokens / 3600

    def _refill_budget(self):
        now = time.monotonic()
        elapsed = now - self._last_refill
        self.tokens_used = max(0,
            self.tokens_used - elapsed * self.token_refill_rate)
        self._last_refill = now

    def _budget_ok(self, estimated_tokens: int = 2048) -> bool:
        self._refill_budget()
        return self.tokens_used + estimated_tokens <= self.budget_tokens

    async def run_step(self, task: str, task_type: str = 'general') -> dict:
        """
        Один полный шаг роя.
        Возвращает dict с ответом, метриками, логами.
        """
        step_start = time.monotonic()
        log.info(f"Step {self.step} | task_type={task_type} | len={len(task)}")

        # ── t0: Embedding ────────────────────────────────────────────────────
        task_vec = self.memory.encode(task)

        # ── t1: FAISS + PPR поиск ────────────────────────────────────────────
        past = self.memory.search(task_vec, k=5)
        context_from_memory = '\n'.join(
            f"[past] {p['answer'][:200]}" for p in past
        ) if past else ""

        # Связь #1: R_mem — влияние retrieval на bandit reward (ТЗ §1)
        # R_mem = mean(norm_entropy + norm_priority) по retrieved blocks
        # Хранится до вызова _build_bandit_rewards()
        self._r_mem = self._compute_r_mem(past)
        self.bandit.set_memory_context(past)
        self.blackboard.set_topology_bias(self.memory.topology_bias())

        novelty = self._external_energy(task_vec, past)
        r_ext = novelty  # финальный R_ext будет пересчитан после q_env

        # ── t2: Асинхронный поиск (не блокируем) ────────────────────────────
        # [Ф1.0] Единственный исходящий трафик в системе. В офлайн-фазе задача
        # вообще не должна покидать машину, поэтому таск не создаётся, а не
        # создаётся-и-игнорируется: иначе GitHub/Google получают текст задачи
        # даже при search.enabled=false на уровне format_context.
        search_enabled = bool(self.cfg.get('search', {}).get('enabled', False))
        search_task = asyncio.create_task(
            self.search.search_all(task[:200])
        ) if search_enabled else None

        # ── t3: PID → K_ACT ─────────────────────────────────────────────────
        chaos_state = self.chaos.update(self._last_energy_raw)
        k_act_raw = chaos_state['k_act']
        lambda_L = chaos_state['lambda_L']
        step_state = RuntimeStepState(lambda_L=lambda_L)
        self.bandit.set_lambda(lambda_L)
        self.memory.set_lambda(lambda_L)
        # Связь #2: передать λ в hypercycle для alpha_eff
        self.hypercycle.set_lambda(lambda_L)
        # FIX 3: clamp K_ACT к числу живых агентов — иначе select вернёт < k_act
        alive_count = sum(1 for a in self.bandit.agents.values() if a.alive)
        k_act = max(1, min(k_act_raw, alive_count))
        if k_act != k_act_raw:
            log.warning("  K_ACT clamped %d→%d (alive_agents=%d)", k_act_raw, k_act, alive_count)

        log.info(f"  Chaos: λ_L={lambda_L:.3f} regime={chaos_state['regime']} "
                 f"K_ACT={k_act} E_raw={self._last_energy_raw:.3f} R_ext={r_ext:.3f}")

        # ── t4: Гиперцикл → квоты каст ──────────────────────────────────────
        alloc = self.hypercycle.allocate(k_act)
        log.info(f"  Hypercycle alloc: G={alloc['G']} C={alloc['C']} S={alloc['S']}")

        # ── t5: Thompson → модели ────────────────────────────────────────────
        temperature = self.bandit.get_temperature(task_vec)
        g_ids = self.bandit.select(alloc['G'], task_vec, allowed_roles=['G'])
        c_ids = self.bandit.select(alloc['C'], task_vec, allowed_roles=['C'])
        s_ids = self.bandit.select(alloc['S'], task_vec, allowed_roles=['S'])
        all_selected = g_ids + c_ids + s_ids

        log.info(f"  Selected: G={g_ids} C={c_ids} S={s_ids} T={temperature:.2f}")

        # ── t6: Генераторы параллельно ───────────────────────────────────────
        evo_context = self.search.read_recent_evolution(n=3)
        system_g = (
            "You are a Generator. Produce a diverse, creative response. "
            + (f"Context from memory:\n{context_from_memory[:500]}\n" if context_from_memory else "")
            + (f"Evolution hints:\n{evo_context[:300]}\n" if evo_context else "")
        )

        async with aiohttp.ClientSession() as session:
            g_results = await asyncio.gather(*[
                call_model(session, self.models_by_id[mid], task,
                           system_g, temperature)
                for mid in g_ids if mid in self.models_by_id
            ])
        self._record_provider_results(g_results)

        # ── t7: Value отсев ──────────────────────────────────────────────────
        g_ok = [r for r in g_results if r.get('ok')]
        if not g_ok:
            log.warning("  All generators failed!")
            g_ok = [{'id': 'fallback', 'text': 'I was unable to generate a response.',
                     'tokens_out': 10, 'latency': 0, 'ok': True}]

        # Быстрый Q_ext отсев
        for r in g_ok:
            r['q_ext'] = self.scorer.q_ext(r['text'], task_type)

        # [Ш1.3] Пер-агентный ВНЕШНИЙ сигнал по каждому черновику.
        # Считается только при включённом флаге: это N дополнительных запусков
        # песочницы на шаг (~0.9 с каждый на MBPP).
        # Зависит от Ш1.2: без тестов q_env возвращает одну и ту же величину
        # для всех черновиков, и пер-агентная награда выродится обратно в
        # общую. Поэтому ниже стоит проверка на различимость.
        if bool(self.cfg.get('fix', {}).get('per_agent_reward', False)):
            for r in g_ok:
                # Тонкий сигнал вместо шестиуровневой q_env: та схлопывает всё
                # с ratio<=0.5 ровно в 0.4, и два черновика с 0 и 2 пройденными
                # тестами из 4 становятся неразличимы. Замерено: 12 из 20
                # черновиков попадали ровно в 0.4.
                fine = self.scorer.q_env_fine(r['text'], task_type)
                r['q_env_own'] = fine if fine is not None else \
                    self.scorer.q_env(r['text'], task_type)
            own = [r.get('q_env_own') for r in g_ok]
            if len(own) > 1 and len(set(own)) == 1:
                log.warning(
                    "  per_agent_reward: сигнал одинаков у всех %d черновиков "
                    "(%.3f). Если это не честная ничья (оба решили верно), "
                    "внешней различимости нет.", len(own), own[0])

        g_ok.sort(key=lambda r: -r['q_ext'])
        g_top = g_ok[:max(1, len(g_ok))]  # передаём все критикам

        # ── t8: Критики ──────────────────────────────────────────────────────
        g_texts = '\n\n---\n'.join(
            f"Draft {i+1}:\n{r['text'][:800]}"
            for i, r in enumerate(g_top)
        )
        system_c = (
            "You are a Critic. Identify logical errors, gaps and improvements "
            "in these drafts. Be concise and specific."
        )

        async with aiohttp.ClientSession() as session:
            c_results = await asyncio.gather(*[
                call_model(session, self.models_by_id[mid],
                           f"Task: {task[:300]}\n\n{g_texts}",
                           system_c, temperature * 0.8)
                for mid in c_ids if mid in self.models_by_id
            ])
        self._record_provider_results(c_results)
        c_ok = [r for r in c_results if r.get('ok')]

        # ── t9: Ждём поиск ───────────────────────────────────────────────────
        search_context = ""
        if search_task is not None:
            try:
                search_results = await asyncio.wait_for(search_task, timeout=3.0)
                search_context = self.search.format_context(search_results, max_tokens=480)
            except asyncio.TimeoutError:
                log.debug("  Search timed out")

        # ── t10: Синтезатор ──────────────────────────────────────────────────
        critiques = '\n'.join(r['text'][:400] for r in c_ok[:2]) if c_ok else ""
        system_s = (
            "You are a Synthesizer. Produce the final, best answer by "
            "combining the drafts and addressing all critique points."
            + (f"\n\nSearch context:\n{search_context}" if search_context else "")
        )
        synthesis_prompt = (
            f"Task: {task}\n\n"
            f"Drafts:\n{g_texts}\n\n"
            + (f"Critique:\n{critiques}\n" if critiques else "")
        )

        async with aiohttp.ClientSession() as session:
            s_results = await asyncio.gather(*[
                call_model(session, self.models_by_id[mid],
                           synthesis_prompt, system_s, temperature * 0.7)
                for mid in s_ids if mid in self.models_by_id
            ])
        self._record_provider_results(s_results)
        s_ok = [r for r in s_results if r.get('ok')]

        if s_ok:
            # [TASK_ACCEPTANCE §B] Прежде здесь безусловно брался s_ok[0] —
            # первый ответивший синтезатор, без какого-либо отбора. Замерено:
            # на 87 шагах из 200 проходящий тесты вариант БЫЛ, и в 32 из них
            # итог его не содержал (f2 = 0.368). Гейт применяется ниже, после
            # депт-лупа, чтобы в пул попал и его результат.
            final_answer = s_ok[0]['text']
            tokens_final = s_ok[0]['tokens_out']
            latency_final = s_ok[0]['latency']
        elif g_ok:
            final_answer = g_ok[0]['text']
            tokens_final = g_ok[0]['tokens_out']
            latency_final = g_ok[0]['latency']
        else:
            final_answer = "System error: no successful calls."
            tokens_final = 10
            latency_final = 0

        # Передаём текущий e_base как ориентир для глобального критерия (FIX 2)
        _e_ref = getattr(self, '_prev_e_total', 0.0)
        # [lab, обратимо] depth.thermo_causal_fix: не класть final_answer в блок
        # ДО расчёта термодинамики. Иначе H_old ≡ H_new ⇒ ΔH ≡ 0 ⇒ Φ ≡ 0.
        _thermo_fix = bool(self.cfg.get('depth', {}).get('thermo_causal_fix', False))
        self._h_old_fix = None
        _drafts = [r['text'] for r in g_ok[:3]]
        depth_results = await self._run_depth_loop(
            task=task,
            task_type=task_type,
            initial_answer=final_answer,
            draft_texts=_drafts if _thermo_fix else _drafts + [final_answer],
            s_ids=s_ids,
            temperature=temperature,
            lambda_L=lambda_L,
            current_e_total=_e_ref,
        )
        if depth_results:
            final_answer = depth_results[-1]['text']
            tokens_final += sum(r.get('tokens_out', 0) for r in depth_results)

        # ── t10b: приёмка по исполнению ──────────────────────────────────────
        # [TASK_ACCEPTANCE §B.3/§B.4] Тот же самый core.acceptance.select, что
        # зовёт ветка BoN — не похожий код, а один вызов. Разница между ветками
        # остаётся ровно в том, как порождаются кандидаты.
        # Пул: депт-лупный итог, синтезы, черновики. Порядок важен: элемент 0
        # это то, что вернуло бы прежнее правило s_ok[0]+депт-луп, и по нему
        # считается would_first.
        self._last_gate = None
        self._pre_gate_answer = final_answer      # что вернуло бы прежнее правило
        if getattr(self, 'acceptance_probe', None) is not None:
            cand_texts = [final_answer]
            cand_texts += [r['text'] for r in s_ok if r['text'] != final_answer]
            cand_texts += [r['text'] for r in g_ok if r['text'] != final_answer]
            gate = acceptance_select(
                cand_texts,
                q_ext=lambda t: self.scorer.q_ext(t, task_type),
                probe=self.acceptance_probe)
            self._last_gate = acceptance_trace(gate)
            if gate.chosen > 0:
                final_answer = cand_texts[gate.chosen]
            log.info("  Acceptance: stage=%s кандидатов=%d ст1=%d ст2=%d "
                     "взят=%d (прежнее правило дало бы 0, argmax q_ext — %d)",
                     gate.stage, len(cand_texts), gate.n_stage1, gate.n_stage2,
                     gate.chosen, gate.would_qext)

        # ── t11: OpenCode тест ───────────────────────────────────────────────
        q_env = self.scorer.q_env(final_answer, task_type)
        step_state.q_env = q_env

        # Финальный R_ext с учётом q_env (GPT fix: q_env weight 0.25→0.35)
        r_ext = self._compute_r_ext(novelty, q_env)
        log.info(f"  R_ext_final={r_ext:.3f} novelty={novelty:.3f} q_env={q_env:.3f}")

        # Обновляем токен бюджет
        # [lab-fix] перенесено сюда из t12: total_tokens читается ниже в
        # compute_thermodynamics(tokens_used=total_tokens) до своего присваивания
        # → UnboundLocalError. Порядок вычислений не меняется: g_ok/c_ok/s_ok/
        # depth_results уже финальны на этой точке.
        total_tokens = sum(r.get('tokens_out', 0) for r in g_ok + c_ok + s_ok + depth_results)
        self.tokens_used += total_tokens

        # ── t11b: Термодинамика Пригожина (ПОСЛЕ q_env, ДО memory.add) ──────────
        # Каузальный порядок: H_old зафиксирован до upsert внутри compute_thermodynamics
        block_id = f"step-{self.step}"
        self._last_thermo = self.blackboard.compute_thermodynamics(
            block_id=block_id,
            final_answer=final_answer,
            task=task,
            task_type=task_type,
            q_env=q_env,
            tokens_used=total_tokens,
            h_old=getattr(self, '_h_old_fix', None),
        )
        thermo = self._last_thermo
        step_state.H_before = thermo.H_old
        step_state.H_after = thermo.H_new
        step_state.phi = thermo.Phi
        log.info(
            "  Thermo: H %.4f→%.4f ΔH=%.4f π_r=%.5f "
            "d_eS=%.5f d_iS=%.5f Φ=%.5f alive=%s",
            thermo.H_old, thermo.H_new, thermo.delta_H, thermo.pi_r,
            thermo.d_eS, thermo.d_iS, thermo.Phi, thermo.alive,
        )

        # ── t12: E_total ─────────────────────────────────────────────────────
        # Получаем эмбеддинги для D_sem
        all_answers = [r['text'] for r in g_ok + s_ok + depth_results]
        if len(all_answers) > 1:
            embeddings = [self.memory.encode(a[:500]) for a in all_answers[:6]]
            d_sem = semantic_diversity(embeddings)
        else:
            d_sem = 0.0

        # Q синтеза и Q лучшего одиночки
        q_syn = self.scorer.q_ext(final_answer, task_type)
        q_best_gen = max((r['q_ext'] for r in g_ok), default=0.0)
        e_score = q_syn - q_best_gen

        # (total_tokens посчитан выше, перед t11b)

        i_before = self.memory.I
        e_base = self.emergence.compute_base(
            e_score=e_score,
            d_sem=d_sem,
            q_syn=q_syn,
            k_act=k_act,
            cost=total_tokens,
            budget=self._step_budget(),
            lambda_L=lambda_L,
        )

        trace_id = self.memory.add(
            prompt=task, answer=final_answer, vec=task_vec,
            e_total=e_base, e_score=e_score, d_sem=d_sem,
            k_act=k_act, models=all_selected, tokens_out=tokens_final
        )
        delta_I = self.memory.I - i_before
        e_total = self.emergence.apply_delta_i(e_base, delta_I)
        step_state.delta_I = delta_I
        step_state.e_total = e_total
        self.memory.update_trace_e_total(trace_id, e_total)
        self.memory.update_trace_thermo(trace_id, thermo.H_new, thermo.Phi)

        self._prev_e_total = e_total  # для depth loop FIX 2

        # Связь C: передать Φ и ΔI в memory для приоритизации следующего поиска
        # priority = 0.6*Φ + 0.4*ΔI будет использован в memory.search()
        self.memory.set_search_priority(phi=thermo.Phi, delta_i=delta_I)

        log.info(f"  E_total={e_total:.3f} E_base={e_base:.3f} E_score={e_score:.3f} "
                 f"D_sem={d_sem:.3f} Q_syn={q_syn:.3f} dI={delta_I:.6f} "
                 f"Phi={thermo.Phi:.4f} priority={0.6*max(thermo.Phi,0)+0.4*max(delta_I,0):.4f}")

        # [Ш1.3] marginal contribution по кастам.
        #   G — свой собственный внешний результат черновика;
        #   S — превышение синтеза над лучшим черновиком, то есть ровно та
        #       величина, которую система и объявляет целью (E_score);
        #   C — на первом проходе получает оценку синтеза: честный leave-one-out
        #       по критикам требует ещё одного вызова синтеза на шаг, это
        #       отдельный пункт после замера Ш1.
        per_agent = None
        if bool(self.cfg.get('fix', {}).get('per_agent_reward', False)):
            own = [r['q_env_own'] for r in g_ok if 'q_env_own' in r]
            best_draft = max(own) if own else 0.0
            fine_syn = self.scorer.q_env_fine(final_answer, task_type)
            syn_signal = fine_syn if fine_syn is not None else q_env
            # E_score по внешнему сигналу, сдвинутый в [0,1]: 0.5 — синтез
            # воспроизвёл лучший черновик, >0.5 — улучшил, <0.5 — испортил.
            s_reward = float(np.clip(0.5 + (syn_signal - best_draft), 0.0, 1.0))
            per_agent = {}
            for aid in g_ids:
                rec = next((r for r in g_ok if r.get('id') == aid), None)
                if rec is not None and 'q_env_own' in rec:
                    per_agent[aid] = float(rec['q_env_own'])
            for aid in s_ids:
                per_agent[aid] = s_reward
            # [Ш1.3] Критики: leave-one-out вместо общей оценки синтеза.
            # Раньше C и S получали одно и то же число, то есть критик не
            # отличался от синтезатора вообще. LOO — то, что «Анализ
            # Sigma_v6.7» и называл marginal contribution:
            #     r_j = q(итог) - q(итог без критики j)
            # Стоит одного дополнительного вызова синтеза на критика. На GGUF
            # это ~2-5 с при 125 ток/с — при API-пуле было бы неприемлемо.
            loo = bool(self.cfg.get('fix', {}).get('critic_loo', False))
            if loo and c_ok and len(c_ok) > 1 and s_ids:
                await self._critic_loo(
                    task, task_type, c_ok, c_ids, g_texts, s_ids,
                    temperature, syn_signal, per_agent)
            else:
                for aid in c_ids:
                    per_agent[aid] = s_reward

        # [Ш1.3c] Кто какую касту реально отыграл. Без этого пер-ролевую
        # статистику вести не по чему: agent.roles — это ДОПУСТИМЫЕ роли
        # (у всех [G,C,S]), а не сыгранная.
        roles_played = {}
        for aid in g_ids:
            roles_played[aid] = 'G'
        for aid in c_ids:
            roles_played[aid] = 'C'
        for aid in s_ids:
            roles_played[aid] = 'S'

        rewards = self._build_bandit_rewards(all_selected, e_total, per_agent)
        self.bandit.update(all_selected, rewards, task_vec,
                           temperature=temperature,
                           roles_played=roles_played)
        self._update_agent_energy(all_selected, rewards, total_tokens, d_sem, r_ext, thermo.delta_H, q_env)

        mean_e_by_caste = {
            'G': float(np.mean([r['q_ext'] for r in g_ok])) if g_ok else 0.5,
            'C': float(np.mean([self.scorer.q_ext(r['text']) for r in c_ok])) if c_ok else 0.5,
            'S': q_syn,
        }
        self.hypercycle.update(mean_e_by_caste)

        self._last_energy_raw = sum(a.energy for a in self.bandit.agents.values())

        # ── t13: git commit при эмерджентности ──────────────────────────────
        if self.emergence.is_evolution_worthy(e_total) and task_type == 'code':
            self._commit_evolution(final_answer, e_total, trace_id)

        # Сохранить bandit state каждые 50 шагов
        if self.step % 50 == 0:
            self.bandit.save('data/bandit_state.json')

        step_time = time.monotonic() - step_start
        self.step += 1

        result = {
            'answer': final_answer,
            'trace_id': trace_id,
            'e_total': e_total,
            'e_base': e_base,
            'e_score': e_score,
            'd_sem': d_sem,
            'k_act': k_act,
            'lambda_L': lambda_L,
            'regime': chaos_state['regime'],
            'models': all_selected,
            'step_time': step_time,
            'tokens': total_tokens,
            'step': self.step,
            'I': self.memory.I,
            'delta_I': delta_I,
            'thermo': {
                'H_old':   thermo.H_old,
                'H_new':   thermo.H_new,
                'delta_H': thermo.delta_H,
                'pi_r':    thermo.pi_r,
                'd_eS':    thermo.d_eS,
                'd_iS':    thermo.d_iS,
                'Phi':     thermo.Phi,
                'alive':   thermo.alive,
            },
            # [TASK_ACCEPTANCE §D] След гейта на шаге: результат каждой
            # ступени, что взято и что взяли бы прежние правила. Нужен, чтобы
            # старая и новая приёмка сравнивались ВНУТРИ прогона.
            'gate': self._last_gate,
            'pre_gate_answer': self._pre_gate_answer,
        }

        # Structured Telemetry Log
        try:
            from core.telemetry import log_telemetry
            log_telemetry({
                'step': self.step, # self.step was already incremented at line 511
                'task_type': task_type,
                'lambda': lambda_L,
                'phi': thermo.Phi,
                'entropy': thermo.H_new,
                'delta_h': thermo.delta_H,
                'e_total': e_total,
                'regime': chaos_state['regime'],
                'k_act': k_act,
                'retrieval_count': len(past) if 'past' in locals() else 0,
                'memory_nodes': len(self.memory.graph.nodes),
                'memory_edges': len(self.memory.graph.edges),
                'alive_agents': sum(1 for a in self.bandit.agents.values() if a.alive),
                'provider_failures': sum(h.consecutive_failures for h in self.bandit.provider_health.values()),
                'tokens_used': total_tokens,
                'latency': step_time
            })
        except Exception as tel_err:
            log.warning(f"Telemetry logging failed: {tel_err}")

        log.info(f"  Done in {step_time:.2f}s | I={self.memory.I:.3f}")
        return result


    def _build_bandit_rewards(self, selected: List[str], e_total: float,
                              per_agent: Optional[Dict[str, float]] = None
                              ) -> Dict[str, float]:
        """
        Связь A: entropy → reward_adj (ТЗ §A).
        H_hat = H / (1 + H)  — ограниченная энтропия текущего блока
        reward_adj = reward * (1 + a * H_hat),  a=0.3
        H берётся из последнего compute_thermodynamics (thermo.H_new).

        [Ш1.3] per_agent — marginal contribution по агенту. Если передан и флаг
        fix.per_agent_reward включён, база берётся своя на агента, а не одна
        глобальная.

        Дефект R1: без этого всем агентам шага уходило ОДНО число, и в прогоне
        mu всех девяти агентов совпадали ровно (spread = 0.000000) — granite и
        Qwen3-1.7B стали неразличимы для селектора. Следствия каскадные:
        энергии идут по одной траектории и сходятся к точке между death и
        split (0 событий за весь лог), k_i каст почти не расходятся.

        Целевая функция проекта из первого же разбора требовала именно
        пер-агентного s_i(x):
            U_t(x) = Σ_i π(i|x)·s_i(x) − λ Σ π·c_i − μ Σ π·ℓ_i
        """
        use_pa = bool(self.cfg.get('fix', {}).get('per_agent_reward', False))
        reward = float(np.clip(0.5 + e_total - self._reward_baseline, 0.0, 1.0))
        self._reward_baseline = 0.95 * self._reward_baseline + 0.05 * e_total

        # Entropy boost от структурной энтропии блока (связь A предыдущего этапа)
        H = self._last_thermo.H_new
        H_hat = H / (1.0 + H)
        a = 0.3
        reward_adj = reward * (1.0 + a * H_hat)

        # Связь #1: R_mem → effective_reward (ТЗ §1)
        # effective_reward = reward_adj * (1 + b * R_mem),  b=0.2
        b = 0.2
        R_mem = getattr(self, '_r_mem', 0.0)

        if use_pa and per_agent:
            # Модификаторы (энтропия блока, R_mem) — общие для шага: они про
            # контекст, а не про агента. Различает агентов именно база.
            gain = (1.0 + a * H_hat) * (1.0 + b * R_mem)
            out = {}
            for aid in selected:
                base = per_agent.get(aid)
                if base is None:
                    base = reward          # нет своей оценки — общая
                out[aid] = float(np.clip(float(base) * gain, 0.0, 1.0))
            log.info("  reward per-agent: " +
                     " ".join(f"{k}={v:.3f}" for k, v in out.items()))
            return out

        effective_reward = reward_adj * (1.0 + b * R_mem)
        effective_reward = float(np.clip(effective_reward, 0.0, 1.0))

        log.debug("  reward: base=%.4f H_hat=%.4f R_mem=%.4f eff=%.4f",
                  reward, H_hat, R_mem, effective_reward)
        return {aid: effective_reward for aid in selected}

    def _record_provider_results(self, results: List[dict]):
        plog = logging.getLogger('providers')
        for result in results or []:
            model = self.models_by_id.get(result.get('id'))
            if not model:
                continue
            
            is_ok = bool(result.get('ok'))
            err_msg = result.get('error', '')
            is_fatal = bool(result.get('fatal'))
            
            # Log structured trace to providers.log via UTF-8 safe providers logger
            plog.info(
                f"CallResult | id={result.get('id')} | provider={model['provider']} | "
                f"ok={is_ok} | fatal={is_fatal} | err={err_msg} | latency={result.get('latency', 0.0):.3f}s"
            )
            
            if is_fatal:
                # Force immediate cooldown by registering 3 failures
                for _ in range(3):
                    self.bandit.record_provider_result(
                        model['provider'],
                        ok=False,
                        latency=float(result.get('latency', 0.0) or 0.0)
                    )
            else:
                self.bandit.record_provider_result(
                    model['provider'],
                    ok=is_ok,
                    latency=float(result.get('latency', 0.0) or 0.0),
                )


    def _split_into_blocks(self, text: str, task_type: str) -> List[str]:
        """
        Разбить текст ответа на семантические блоки (ТЗ п.4 / Grok).

        Для кода: разбиваем по функциям/классам через ast.parse().
        Для остального: разбиваем по параграфам / предложениям.

        Возвращает список блоков ≥ 50 символов.
        Если разбиение не получилось — возвращаем весь текст как один блок.

        Применяется в depth loop для определения гранулярности blackboard.
        """
        import ast as ast_mod, re

        blocks = []

        if task_type == 'code':
            # Пробуем разбить по топ-level определениям через AST
            code = self.scorer._extract_python_code(text) if hasattr(self.scorer, '_extract_python_code') else text
            try:
                tree = ast_mod.parse(code)
                lines = code.split('\n')
                for node in ast_mod.walk(tree):
                    if isinstance(node, (ast_mod.FunctionDef, ast_mod.AsyncFunctionDef,
                                         ast_mod.ClassDef)):
                        start = node.lineno - 1
                        end = getattr(node, 'end_lineno', start + 10)
                        block_text = '\n'.join(lines[start:end])
                        if len(block_text) >= 50:
                            blocks.append(block_text)
            except SyntaxError:
                pass  # fallback ниже

        if not blocks:
            # Fallback: разбиваем по параграфам (двойной перевод строки)
            parts = re.split(r'\n{2,}', text.strip())
            blocks = [p.strip() for p in parts if len(p.strip()) >= 50]

        if not blocks:
            # Последний fallback: по предложениям
            sentences = re.split(r'(?<=[.!?])\s+', text)
            blocks = [s.strip() for s in sentences if len(s.strip()) >= 50]

        return blocks if blocks else [text]

    async def _run_depth_loop(self, task: str, task_type: str, initial_answer: str,
                              draft_texts: List[str], s_ids: List[str],
                              temperature: float, lambda_L: float,
                              current_e_total: float = 0.0) -> List[dict]:
        """
        Depth loop с двойным критерием остановки (ТЗ п.2 / GPT Fix):

        СТАРАЯ проблема: цикл оптимизировал ЛОКАЛЬНУЮ энтропию блока,
        не связанную с глобальной целью. Мог «полировать» мусор.

        ИСПРАВЛЕНИЕ: добавить глобальный критерий ΔE_total:
          - Вычислить e_total_approx для кандидата (через compute_base)
          - Если e_total_approx < current_e_total - epsilon_global → отклонить и прервать
          - Аналогия: принцип оптимальности Беллмана — каждый шаг должен
            улучшать глобальную ценность, а не только локальную метрику

        Двойной критерий остановки:
          1. |H_new - H_old| < epsilon   (локальная энтропия стабилизировалась)
          2. E_total_new < E_total_current - epsilon_global  (глобальный регресс)
        """
        depth_cfg = self.cfg.get('depth', {})
        if not depth_cfg.get('enabled', False) or not s_ids:
            return []

        block_id = f"step-{self.step}"
        block = None
        for text in draft_texts[:self.blackboard.max_versions]:
            try:
                block = self.blackboard.upsert(block_id, text, task=task, task_type=task_type)
            except Exception as exc:
                log.debug(f"  Blackboard add failed: {exc}")
        # [lab, обратимо] снимок H_old строго после черновиков и строго до того,
        # как depth loop начнёт класть в блок свои кандидаты.
        if depth_cfg.get('thermo_causal_fix', False) and block is not None:
            self._h_old_fix = self.blackboard.effective_entropy(block_id)

        if block is None or block.entropy < depth_cfg.get('entropy_threshold', 0.08):
            return []

        selected_block = self.blackboard.select([block], lambda_L=lambda_L)
        if selected_block is None:
            return []

        model_id = next((mid for mid in s_ids if mid in self.models_by_id), None)
        if model_id is None:
            return []

        max_steps_cfg = int(depth_cfg.get('max_steps', 2))
        # Связь B: λ → depth_scale = 1.0 + 0.5*λ (ТЗ §B)
        # При λ>0 (хаос) — больше итераций, при λ<0 (порядок) — меньше
        depth_scale = 1.0 + 0.5 * float(lambda_L)
        max_steps = max(1, int(round(max_steps_cfg * depth_scale)))
        log.debug("  depth_scale=%.3f max_steps=%d (λ=%.3f)",
                  depth_scale, max_steps, lambda_L)
        epsilon = float(depth_cfg.get('epsilon', 0.01))
        epsilon_global = float(depth_cfg.get('epsilon_global', 0.05))  # FIX 2
        token_budget = int(depth_cfg.get('token_budget', 2048))
        used_tokens = 0
        h_old = selected_block.entropy
        current = initial_answer
        current_etotal = current_e_total   # FIX 2: отслеживаем глобальный E_total
        results: List[dict] = []

        log.info(f"  Depth loop start: block={selected_block.id} H={h_old:.4f} "
                 f"E_total_ref={current_etotal:.4f}")
        async with aiohttp.ClientSession() as session:
            for depth_step in range(max_steps):
                if used_tokens >= token_budget:
                    break
                prompt = (
                    f"Task:\n{task[:600]}\n\n"
                    f"Current answer:\n{current[:1600]}\n\n"
                    "Improve the answer by resolving contradictions between drafts. "
                    "Keep only claims that survive the critique. Be concise."
                )
                system = (
                    "You are the depth loop of a semantic blackboard. "
                    "Your goal is to reduce semantic entropy without adding unsupported claims."
                )
                result = await call_model(
                    session, self.models_by_id[model_id], prompt, system,
                    min(temperature, 0.65)
                )
                self._record_provider_results([result])
                if not result.get('ok'):
                    log.debug(f"  Depth loop failed: {result.get('error')}")
                    break

                candidate = result['text']
                used_tokens += result.get('tokens_out', len(candidate.split()))
                selected_block.add_version(candidate, task=task, task_type=task_type)
                h_new = selected_block.entropy

                q_candidate = self.scorer.q_ext(candidate, task_type)
                q_current   = self.scorer.q_ext(current, task_type)

                # FIX 2: аппроксимируем E_total для кандидата
                e_score_cand = q_candidate - q_current
                # [Ш2.1] Раньше сюда подставлялась энтропия блока как «прокси
                # D_sem». В натах это давало tanh(2.5)=0.986, то есть кандидату
                # даром доставалось +0.27 «разнообразия» — при том, что
                # разнообразие между ОДНИМ кандидатом и ОДНИМ текущим ответом
                # это вообще другая величина. Считаем настоящий D_sem по паре.
                if bool(self.cfg.get('fix', {}).get('entropy_norm', False)):
                    try:
                        d_sem_cand = semantic_diversity([
                            self.memory.encode(candidate[:500]),
                            self.memory.encode(current[:500])])
                    except Exception:
                        d_sem_cand = 0.0
                else:
                    d_sem_cand = h_new
                e_total_approx = self.emergence.compute_base(
                    e_score=e_score_cand,
                    d_sem=d_sem_cand,
                    q_syn=q_candidate,
                    k_act=1,               # один агент в depth loop
                    cost=result.get('tokens_out', 100),
                    budget=self._step_budget(),
                    lambda_L=lambda_L,
                )

                # FIX 2: глобальный критерий — прерываем при деградации E_total
                if e_total_approx < current_etotal - epsilon_global:
                    log.info(f"  Depth step {depth_step}: GLOBAL STOP "
                             f"E_approx={e_total_approx:.4f} < ref={current_etotal:.4f} "
                             f"- epsilon={epsilon_global}")
                    break

                # Принять кандидата если: H снизилась или качество не хуже
                if h_new <= h_old + epsilon or q_candidate >= q_current:
                    current = candidate
                    current_etotal = e_total_approx   # обновить ориентир
                    result['entropy'] = h_new
                    result['id'] = result.get('id', model_id)
                    results.append(result)

                log.info(f"  Depth step {depth_step}: H {h_old:.4f}->{h_new:.4f} "
                         f"q={q_candidate:.3f} E_approx={e_total_approx:.4f} "
                         f"tokens={used_tokens}/{token_budget}")
                if abs(h_new - h_old) < epsilon:
                    break
                h_old = h_new

        return results

    async def _critic_loo(self, task, task_type, c_ok, c_ids, g_texts, s_ids,
                          temperature, syn_signal, per_agent):
        """[Ш1.3] Marginal contribution критика: пересинтез без его критики.

        r_j = 0.5 + (q(итог со всеми) − q(итог без критика j))
        Больше 0.5 — критика помогла, меньше — помешала, 0.5 — не изменила.

        Именно эту величину «Анализ Sigma_v6.7» и предписывал в разделе «что
        оставить» (marginal contribution). Без неё критики получали оценку
        синтезатора и были для бандита неотличимы от него.
        """
        model_id = next((mid for mid in s_ids if mid in self.models_by_id), None)
        if model_id is None:
            return
        id_to_agent = {}
        for aid, rec in zip(c_ids, c_ok):
            id_to_agent[rec.get('id', aid)] = aid

        async with aiohttp.ClientSession() as session:
            for held_out in list(c_ok):
                rest = [r for r in c_ok if r is not held_out]
                crit = '\n'.join(r['text'][:400] for r in rest[:2])
                system_s = ("You are a Synthesizer. Produce the final, best "
                            "answer by combining the drafts and addressing all "
                            "critique points.")
                prompt = (f"Task: {task}\n\nDrafts:\n{g_texts}\n\n"
                          + (f"Critique:\n{crit}\n" if crit else ""))
                res = await call_model(session, self.models_by_id[model_id],
                                       prompt, system_s, temperature * 0.7)
                aid = id_to_agent.get(held_out.get('id'))
                if aid is None:
                    continue
                if not res.get('ok'):
                    per_agent[aid] = 0.5
                    continue
                without = self.scorer.q_env_fine(res['text'], task_type)
                if without is None:
                    per_agent[aid] = 0.5
                    continue
                per_agent[aid] = float(np.clip(
                    0.5 + (syn_signal - without), 0.0, 1.0))
        log.info("  critic LOO: " + " ".join(
            f"{id_to_agent.get(r.get('id'))}={per_agent.get(id_to_agent.get(r.get('id')), 0.5):.3f}"
            for r in c_ok))

    def _compute_r_mem(self, past: List[dict]) -> float:
        """
        Связь #1: R_mem = retrieval influence scalar (ТЗ §1)
        R_mem = mean(norm_entropy(block) + norm_priority(block)) по retrieved

        norm_entropy  = e_total / (1 + e_total)   прокси entropy из e_total
        norm_priority = e_total / (1 + e_total)   (единый сигнал качества)

        При пустом retrieval: R_mem = 0.
        """
        if not past:
            return 0.0
        vals = []
        for p in past:
            e = float(p.get('e_total', 0.0))
            # Нормируем через sigmoid-like: v/(1+|v|)
            norm = e / (1.0 + abs(e)) if e > 0 else 0.0
            vals.append(norm)
        return float(np.mean(vals)) if vals else 0.0

    def _external_energy(self, task_vec, past: List[dict]) -> float:
        """
        Вычисляет новизну задачи (novelty) по отношению к памяти.
        Возвращает raw novelty ∈ [0, 1].
        Финальный R_ext формируется в _compute_r_ext() после получения q_env.
        """
        if not past:
            return 1.0
        sims = []
        for item in past[:5]:
            prompt = item.get('prompt') or ''
            if not prompt:
                continue
            try:
                prev_vec = self.memory.encode(prompt[:500])
                sim = float(np.dot(task_vec, prev_vec) /
                            ((np.linalg.norm(task_vec) * np.linalg.norm(prev_vec)) + 1e-9))
                sims.append(sim)
            except Exception as exc:
                log.debug(f"  R_ext prompt similarity failed: {exc}")
        max_sim = max(sims) if sims else 0.0
        return float(np.clip(1.0 - max_sim, 0.0, 1.0))

    def _compute_r_ext(self, novelty: float, q_env: float) -> float:
        """
        Финальный R_ext = внешняя энергия агентов (ТЗ п.5 / Gemini / Grok).

          R_ext = 0.40 * novelty    [новизна задачи ∈ [0,1]]
                + 0.125             [базовая энергия = 0.25 * 0.5]
                + 0.35 * q_env      [качество исполнения кода ∈ [0,1]]

        Фактическая сумма коэффициентов: 0.40 + 0.125 + 0.35 = 0.875
        (не 1.0 — R_ext это не вероятностное распределение,
        а абсолютная величина в диапазоне [0.125, 0.875])

        При novelty=1.0, q_env=1.0: R_ext = 0.875 (максимум)
        При novelty=0.0, q_env=0.0: R_ext = 0.125 (базовый минимум)
        """
        return 0.4 * novelty + 0.125 + 0.35 * q_env

    def _update_agent_energy(self, selected: List[str], rewards: Dict[str, float],
                             total_tokens: int, d_sem: float, r_ext: float,
                             delta_H: float = 0.0, q_env: float = 0.5):
        if not selected:
            return
        family_counts: Dict[str, int] = {}
        for aid in selected:
            agent = self.bandit.agents.get(aid)
            if agent:
                family_counts[agent.family] = family_counts.get(agent.family, 0) + 1

        tokens_per_agent = max(1, total_tokens // max(1, len(selected)))
        x_mean = float(np.mean(list(self.hypercycle.x.values()))) if self.hypercycle.x else 1 / 3

        for aid in selected:
            agent = self.bandit.agents.get(aid)
            if agent is None or not agent.alive:
                continue
            role = next((r for r in agent.roles if r in self.hypercycle.x), 'G')
            # Термодинамический вклад: η·max(0,ΔH)·Q_env (ТЗ §5)
            # eta_thermo из конфига (по умолчанию 0.30)
            eta_thermo = self.cfg.get('energy', {}).get('eta_thermo', 0.30)
            r_ext_combined = r_ext + eta_thermo * max(0.0, delta_H) * q_env
            new_energy, event = self.energy_model.step(
                energy=agent.energy,
                r=rewards.get(aid, 0.0),
                n_same=family_counts.get(agent.family, 1),
                is_diverse=d_sem > 0.18,
                tokens_out=tokens_per_agent,
                x_prev=self.hypercycle.x.get(role, x_mean),
                x_mean=x_mean,
                r_ext=r_ext_combined,
            )
            agent.energy = new_energy
            if event == 'death':
                self.bandit.kill(aid)
                log.info(f"  Energy: killed {aid} E={new_energy:.3f}")
            elif event == 'split':
                child_id = self.bandit.split(aid)
                if child_id and aid in self.models_by_id:
                    child_cfg = dict(self.models_by_id[aid])
                    child_cfg['id'] = child_id
                    self.models_by_id[child_id] = child_cfg
                log.info(f"  Energy: split {aid} -> {child_id} E={new_energy:.3f}")

    def _commit_evolution(self, code: str, e_total: float, trace_id: int):
        """Сохраняет успешный код в /swarm/evolution/ с git commit"""
        ts = int(time.time())
        fname = f"swarm/evolution/{ts}.py"
        header = f"# E_total={e_total:.3f} trace_id={trace_id}\n"
        with open(fname, 'w', encoding='utf-8') as f:
            f.write(header + code)
        try:
            repo = subprocess.run(
                ['git', 'rev-parse', '--is-inside-work-tree'],
                capture_output=True, timeout=5, text=True
            )
            if repo.returncode != 0:
                log.warning(f"Evolution saved to {fname}, but git repo is not initialized.")
                return
            add = subprocess.run(
                ['git', 'add', fname], capture_output=True, timeout=5, text=True
            )
            if add.returncode != 0:
                log.warning(f"git add failed for {fname}: {add.stderr[:300]}")
                return
            commit = subprocess.run(
                ['git', 'commit', '-m', f'evolve: E={e_total:.3f} step={self.step}'],
                capture_output=True, timeout=5, text=True
            )
            if commit.returncode != 0:
                log.warning(f"git commit failed for {fname}: {commit.stderr[:300]}")
        except Exception as exc:
            log.warning(f"Evolution saved to {fname}, but git commit failed: {exc}")


# ── Синхронная обёртка для OpenCode ─────────────────────────────────────────
_orchestrator: Optional[MyceliumOrchestrator] = None

def get_orchestrator() -> MyceliumOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = MyceliumOrchestrator()
    return _orchestrator

def ask(task: str, task_type: str = 'general') -> str:
    """
    Точка входа для OpenCode.
    Пример использования в OpenCode:
        from orchestrator import ask
        result = ask("Write a Python function to sort a list", "code")
    """
    orch = get_orchestrator()
    result = asyncio.run(orch.run_step(task, task_type))
    return result['answer']
