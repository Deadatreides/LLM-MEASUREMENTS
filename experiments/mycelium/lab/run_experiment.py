"""
lab/run_experiment.py — прогон Mycelium 8.9.2 на локальных моделях, 5 условий.
TASK-MYCELIUM-RUN.md §5, §6, §9.

Код самого Mycelium не переписывается: все условия задаются
монки-патчами поверх готового экземпляра MyceliumOrchestrator
и правкой config-словаря в памяти.

  M        полный Mycelium, эталон
  M-depth  депт-луп без критериев остановки → всегда до лимита шагов.
           В задании это условие называлось M−Φ, переименовано: Φ депт-луп
           не гейтит, реальные гейты — по энтропии блока и по E_total.
           Дополнительно Φ не влияет на приоритет памяти.
  M-lambda chaos отключён: λ_L ≡ 0, K_ACT ≡ k_act_center, PID не работает
  M-HC     гиперцикл заморожен: x = {G:1/3, C:1/3, S:1/3}
  M-fixed  M + depth.thermo_causal_fix: H_old снимается до того, как депт-луп
           положит final_answer в блок. Без этого ΔH ≡ 0 и Φ ≡ 0 тождественно.
  BoN      без оркестрации: N генераций теми же моделями, отбор по Q_ext,
           N подобрано под фактический бюджет токенов условия M
  BoN-G    то же, но только моделью-генератором Qwen3-1.7B (сильный baseline)

Запуск:
    python lab/run_experiment.py --n-tasks 3  --tag smoke
    python lab/run_experiment.py --n-tasks 30 --tag full
"""

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
import time

os.environ.setdefault('MYCELIUM_ALLOW_WINDOWS_CODE_EXEC', '1')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np                                    # noqa: E402
import orchestrator as orch_mod                       # noqa: E402
from core.scorer import QualityScorer                 # noqa: E402
from core.acceptance import (SandboxProbe, split_tests,   # noqa: E402
                             select as acceptance_select,
                             trace_of as acceptance_trace)

CONDITIONS = ['M', 'M-depth', 'M-lambda', 'M-HC', 'M-fixed', 'BoN', 'BoN-G',
              'M-swap']
SERVER_URL = 'http://127.0.0.1:8077/v1'
BON_MAX_N = 14
CALL_TIMEOUT_S = 600

# ── пул для BoN: те же модели, что в config/models.yaml ─────────────────────
# [Ф1.2] Обновлено под GGUF-пул. Прежний состав (qwen3-1.7b / granite-1b /
# qwen3-0.6b в safetensors) больше не существует: granite оказался granitemoe
# и отсеивается, а остальные заменены GGUF-версиями (в 7-19 раз быстрее).
#
# ПЕРВЫЙ в списке — модель для BoN-G, то есть honest best-of-N одной, самой
# сильной моделью. Именно этот baseline даёт главный разрыв: против него
# прошлый прогон дал M - BoN-G = -0.007 +- 0.045.
# По lab/data/baseline_pool.json сильнейшая — qwen2.5-coder-1.5b (0.684/47%).
BON_MODELS = [
    {'id': 'bon_coder15', 'provider': 'local_coder15', 'endpoint': SERVER_URL,
     'model_name': 'qwen2.5-coder-1.5b-instruct-q4_0', 'ctx': 1024,
     'max_tokens': 256, 'timeout': CALL_TIMEOUT_S},
    {'id': 'bon_qwen3', 'provider': 'local_qwen3', 'endpoint': SERVER_URL,
     'model_name': 'qwen3-1.7b-q4_0', 'ctx': 1024,
     'max_tokens': 256, 'timeout': CALL_TIMEOUT_S},
    {'id': 'bon_smol17', 'provider': 'local_smol17', 'endpoint': SERVER_URL,
     'model_name': 'smollm2-1.7b-instruct-q4_k_m', 'ctx': 1024,
     'max_tokens': 256, 'timeout': CALL_TIMEOUT_S},
]


# ── пул для M-swap ─────────────────────────────────────────────────────────
# Раскладка ролей из §4 задания ставит самую сильную модель (1.7B) на
# генерацию, а самую слабую (0.6B) — на синтез. Синтезатор физически не может
# улучшить черновик, который сильнее его самого: E_score ≤ 0 получается не из
# свойств оркестрации, а из состава ролей. M-swap меняет это местами.
#
#   G — Qwen3-0.6B ×2 + granite ×1
#   C — granite ×3
#   S — Qwen3-1.7B ×3
def _agent(aid, model_name, family, role, max_tokens):
    return {'id': aid, 'provider': 'local', 'endpoint': SERVER_URL,
            'model_name': model_name, 'family': family, 'cost_per_1k': 0.0,
            'max_rpm': 999, 'ctx': 2048, 'max_tokens': max_tokens,
            'roles': [role], 'mu': 0.5, 'sigma': 0.5}


SWAP_POOL = (
    [_agent(f'sw_qwen06_g{i}', 'qwen3-0.6b', 'qwen06', 'G', 512)
     for i in (1, 2)]
    + [_agent('sw_granite_g1', 'granite-1b', 'granite', 'G', 384)]
    + [_agent(f'sw_granite_c{i}', 'granite-1b', 'granite', 'C', 384)
       for i in (1, 2, 3)]
    + [_agent(f'sw_qwen17_s{i}', 'qwen3-1.7b', 'qwen17', 'S', 512)
       for i in (1, 2, 3)]
)


# ── 1. Scorer, знающий про эталонные тесты MBPP ────────────────────────────
class TestAwareScorer(QualityScorer):
    """q_env(answer, 'code') внутри Mycelium вызывается без tests → 0.7 «импортируется».
    Подставляем настоящие тесты текущей задачи, иначе внешней истины нет."""
    def __init__(self, cfg):
        super().__init__(cfg)
        self.current_tests = None

    def q_env(self, answer, task_type='general', tests=None):
        return super().q_env(answer, task_type,
                             tests if tests is not None else self.current_tests)

    def code_detail(self, code, tests=None):
        return super().code_detail(
            code, tests if tests is not None else self.current_tests)

    def q_env_fine(self, answer, task_type='general', tests=None, detail=None):
        return super().q_env_fine(
            answer, task_type,
            tests if tests is not None else self.current_tests, detail)


# ── 2. Запись всех вызовов моделей ─────────────────────────────────────────
class CallRecorder:
    def __init__(self):
        self.calls = []

    def reset(self):
        self.calls = []

    @staticmethod
    def _role(system: str) -> str:
        s = (system or '').lower()
        if 'you are a generator' in s:
            return 'G'
        if 'you are a critic' in s:
            return 'C'
        if 'you are a synthesizer' in s:
            return 'S'
        if 'depth loop' in s:
            return 'D'
        return '?'

    def wrap(self, fn):
        async def wrapped(session, model_cfg, prompt, system, temperature,
                          timeout=30):
            # call_model по умолчанию рвёт запрос через 30 с — это разумно для
            # API-провайдера и абсурдно для локальной 1.7B на ~10 ток/с.
            # Оркестратор timeout никогда не передаёт, так что поднимаем здесь.
            t0 = time.monotonic()
            res = await fn(session, model_cfg, prompt, system, temperature,
                           CALL_TIMEOUT_S)
            self.calls.append({
                'role': self._role(system),
                'agent': model_cfg.get('id'),
                'model': model_cfg.get('model_name'),
                'ok': bool(res.get('ok')),
                'text': res.get('text', ''),
                'tokens_out': int(res.get('tokens_out', 0) or 0),
                'latency': round(time.monotonic() - t0, 2),
                'error': res.get('error'),
            })
            return res
        return wrapped


# ── 3. Буфер логов оркестратора на один шаг ────────────────────────────────
class StepLogBuffer(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.INFO)
        self.lines = []

    def emit(self, record):
        try:
            self.lines.append(f"{record.name}: {record.getMessage()}")
        except Exception:
            pass

    def reset(self):
        self.lines = []


def parse_depth(lines):
    """Сколько шагов депт-лупа прошло и почему остановились."""
    steps = [l for l in lines if 'Depth step' in l]
    started = any('Depth loop start' in l for l in lines)
    reason = None
    if any('GLOBAL STOP' in l for l in lines):
        reason = 'global_e_total_regress'
    elif steps:
        reason = 'entropy_converged_or_limit'
    elif started:
        reason = 'no_accepted_step'
    else:
        reason = 'not_started'
    return {'started': started, 'steps': len(steps), 'stop_reason': reason}


# ── 4. Конфиг / изоляция состояния ─────────────────────────────────────────
_load_config_orig = orch_mod.load_config


def make_orchestrator(condition: str, data_dir: str):
    """Свежий оркестратор с изолированным состоянием памяти/бандита."""
    if os.path.isdir(data_dir):
        shutil.rmtree(data_dir, ignore_errors=True)
    os.makedirs(data_dir, exist_ok=True)
    # bandit_state.json путь захардкожен в orchestrator.py — чистим перед стартом
    if os.path.exists('data/bandit_state.json'):
        os.remove('data/bandit_state.json')

    def _cfg():
        cfg = _load_config_orig()
        cfg['memory']['sqlite_path'] = f'{data_dir}/traces.db'
        cfg['memory']['faiss_path'] = f'{data_dir}/faiss.index'
        cfg['memory']['graph_path'] = f'{data_dir}/graph.pkl'
        if condition == 'M-depth':
            # «Depth Loop идёт всегда до лимита шагов»:
            # epsilon<0  → |H_new-H_old| < epsilon никогда не истинно
            # epsilon_global огромный → глобальный стоп не срабатывает
            cfg['depth']['epsilon'] = -1.0
            cfg['depth']['epsilon_global'] = 1e9
        if condition == 'M-fixed':
            cfg['depth']['thermo_causal_fix'] = True
        if condition == 'M-swap':
            cfg['models'] = [dict(m) for m in SWAP_POOL]
        return cfg

    orch_mod.load_config = _cfg
    try:
        orch = orch_mod.MyceliumOrchestrator()
    finally:
        orch_mod.load_config = _load_config_orig

    orch.scorer = TestAwareScorer(orch.cfg)

    if condition == 'M-depth':
        # Φ больше не влияет на приоритет поиска в памяти
        _orig_prio = orch.memory.set_search_priority
        orch.memory.set_search_priority = lambda phi, delta_i: _orig_prio(0.0, delta_i)

    if condition == 'M-lambda':
        k_center = orch.cfg['chaos']['pid']['k_act_center']
        def _no_chaos(_e_total, _k=k_center):
            return {'lambda_L': 0.0, 'k_act': _k, 'beta_kl': 0.05,
                    'u': 0.0, 'regime': 'EDGE'}
        orch.chaos.update = _no_chaos

    if condition == 'M-HC':
        frozen = {c: 1.0 / 3.0 for c in orch.hypercycle.CASTES}
        orch.hypercycle.x = dict(frozen)
        orch.hypercycle.update = lambda mean_e, _f=frozen: dict(_f)

    return orch


# ── 5. Внешняя оценка через тесты ──────────────────────────────────────────
def q_env_ext(scorer, text, tests):
    try:
        return float(scorer.q_env(text, 'code', tests=tests))
    except Exception as exc:
        logging.getLogger('lab').warning("q_env_ext failed: %s", exc)
        return 0.0


def solved(q):
    return 1 if q >= 1.0 - 1e-9 else 0


# ── 6. BoN ─────────────────────────────────────────────────────────────────
async def run_bon(scorer, task_prompt, tests, token_budget, temperature=0.7,
                  pool=None, probe=None):
    """N генераций теми же моделями, без оркестрации.

    pool=None      — все три модели по кругу («те же модели», что у Mycelium)
    pool=[coder15] — только сильнейшая: BoN-G

    [TASK_ACCEPTANCE §B.4] Отбор — тем же core.acceptance.select, что и в M.
    Прежнее правило (argmax q_ext) теряло 15.5 п.п. из 15.5 доступных: брало
    непроходящего кандидата в 31 задаче из 135, где проходящий был.
    """
    import aiohttp
    pool = pool or BON_MODELS
    system = "You are a Generator. Produce a diverse, creative response."
    cands = []
    spent = 0
    failures = []
    t0 = time.monotonic()
    async with aiohttp.ClientSession() as session:
        for i in range(BON_MAX_N):
            if i > 0 and spent >= token_budget:
                break
            mc = pool[i % len(pool)]
            res = await orch_mod.call_model(session, mc, task_prompt, system,
                                            temperature)
            if not res.get('ok'):
                # [§5.4] Сорванный вызов НЕ кандидат: он не должен попадать
                # ни в знаменатель, ни в отбор. Раньше он добавлялся с
                # q_ext=0.0 и раздувал bon_n.
                failures.append({'model': mc['model_name'],
                                 'error': str(res.get('error'))[:200]})
                continue
            spent += int(res.get('tokens_out', 0) or 0)
            cands.append({
                'model': mc['model_name'], 'ok': True, 'text': res['text'],
                'tokens_out': int(res.get('tokens_out', 0) or 0),
                'q_ext': float(scorer.q_ext(res['text'], 'code')),
            })
    if not cands:
        return {'answer': '', 'candidates': [], 'n': 0, 'tokens': spent,
                'seconds': round(time.monotonic() - t0, 1),
                'temperature': temperature, 'failures': failures, 'gate': None}

    gate = acceptance_select([c['text'] for c in cands],
                             q_ext=lambda t: scorer.q_ext(t, 'code'),
                             probe=probe)
    return {
        'answer': cands[gate.chosen]['text'],
        'candidates': cands,
        'n': len(cands),
        'tokens': spent,
        'seconds': round(time.monotonic() - t0, 1),
        'temperature': temperature,
        'failures': failures,
        'gate': acceptance_trace(gate),
    }


# ── 7. Один (task, condition) ──────────────────────────────────────────────
async def run_one(orch, scorer, condition, task, logbuf, recorder,
                  m_token_budget=None, acceptance=True):
    prompt = task['prompt']
    # [TASK_ACCEPTANCE §B.2] Расщепление ассертов: test_0 уходит в приёмку,
    # остальные — в оценку. Без этого отбор шёл бы по той же величине, по
    # которой считается результат, то есть был бы утечкой.
    # Цена замерена офлайн: solved 29.0 % → 32.0 % (+3.0 п.п.), все числа
    # этого прогона — на ДВУХАССЕРТНОЙ шкале.
    accept_src, tests, _an, _en = split_tests(task['tests'], accept_index=0)
    probe = SandboxProbe(accept_src) if acceptance else None
    scorer.current_tests = tests

    logbuf.reset()
    recorder.reset()
    t0 = time.monotonic()

    if condition.startswith('BoN'):
        pool = [BON_MODELS[0]] if condition == 'BoN-G' else BON_MODELS
        bon = await run_bon(scorer, prompt, tests, m_token_budget or 800,
                            pool=pool, probe=probe)
        answer = bon['answer']
        cand_texts = [c['text'] for c in bon['candidates'] if c['ok']]
        q_ext_cands = [c['q_ext'] for c in bon['candidates'] if c['ok']]
        q_syn_ext = q_env_ext(scorer, answer, tests)
        q_cands_ext = [q_env_ext(scorer, t, tests) for t in cand_texts]
        rec = {
            'condition': condition, 'task_id': task['task_id'], 'status': 'ok',
            'answer': answer[:4000],
            'seconds': bon['seconds'],
            'tokens_completion': bon['tokens'],
            'n_calls': bon['n'],
            'bon_n': bon['n'],
            # внешняя истина
            'q_env_final': q_syn_ext,
            'solved': solved(q_syn_ext),
            'q_env_candidates': q_cands_ext,
            'q_env_best_candidate': max(q_cands_ext) if q_cands_ext else 0.0,
            'E_score_ext': (q_syn_ext - max(q_cands_ext)) if q_cands_ext else 0.0,
            # [§C.5] Тексты кандидатов BoN сохраняются. В sh2_200 их не было,
            # и ветку нельзя было пересчитать поассертно — то есть требование
            # §B.4 «одинаковый механизм в обеих ветках» оказалось непроверяемым
            # ровно в той ветке, против которой идёт сравнение.
            'cand_texts': [t[:3000] for t in cand_texts],
            'cand_models': [c['model'] for c in bon['candidates'] if c['ok']],
            'gate': bon.get('gate'),
            'failures': bon.get('failures') or [],
            # нативные метрики Mycelium неприменимы к BoN
            'q_ext_final': float(scorer.q_ext(answer, 'code')),
            'q_ext_candidates': q_ext_cands,
            'E_score_native': None, 'E_total': None, 'd_sem': None,
            'k_act': None, 'lambda_L': None, 'regime': None, 'thermo': None,
            'hypercycle_x': None, 'alloc': None, 'depth': None,
        }
        return rec

    x_before = dict(orch.hypercycle.x)
    orch.acceptance_probe = probe          # тот же probe, что у ветки BoN
    result = await orch.run_step(prompt, 'code')
    seconds = round(time.monotonic() - t0, 1)

    calls = recorder.calls
    gen_texts = [c['text'] for c in calls if c['role'] == 'G' and c['ok']]
    syn_texts = [c['text'] for c in calls if c['role'] == 'S' and c['ok']]
    depth_texts = [c['text'] for c in calls if c['role'] == 'D' and c['ok']]
    syn_text = result['answer']

    q_syn_ext = q_env_ext(scorer, syn_text, tests)
    q_gen_ext = [q_env_ext(scorer, t, tests) for t in gen_texts]
    best_gen_ext = max(q_gen_ext) if q_gen_ext else 0.0

    # [§D] Что вернуло бы прежнее правило s_ok[0]+депт-луп, до гейта.
    pre_gate = result.get('pre_gate_answer') or ''

    # §7.5: депт-луп — меняет ли ответ и улучшает ли q_env
    pre_depth = syn_texts[0] if syn_texts else (gen_texts[0] if gen_texts else '')
    q_pre_depth = q_env_ext(scorer, pre_depth, tests) if pre_depth else 0.0
    depth_changed = bool(depth_texts) and syn_text.strip() != pre_depth.strip()

    thermo = result['thermo']
    alloc_line = [l for l in logbuf.lines if 'Hypercycle alloc' in l]
    lam_line = [l for l in logbuf.lines if 'Chaos:' in l]

    rec = {
        'condition': condition, 'task_id': task['task_id'], 'status': 'ok',
        'answer': syn_text[:4000],
        'seconds': seconds,
        'tokens_completion': int(result['tokens']),
        'tokens_prompt_est': sum(1 for _ in ()),   # заполняется ниже
        'n_calls': len(calls),
        'calls': [{k: c[k] for k in ('role', 'agent', 'model', 'ok',
                                     'tokens_out', 'latency', 'error')}
                  for c in calls],
        # ── внешняя истина (исполнение эталонных тестов MBPP) ──
        'q_env_final': q_syn_ext,
        'solved': solved(q_syn_ext),
        # [Ш0.3] Снимок mu/energy бандита НА ШАГЕ. Без него проверить
        # разделение моделей нельзя: data/bandit_state.json пишется раз в
        # 50 шагов, и на smoke-прогоне остаётся начальным.
        'bandit_mu': {a: round(st.mu, 6) for a, st in orch.bandit.agents.items()},
        # [Ш1.3c] Пер-ролевая mu ТОЖЕ в записи. Читать её из
        # data/bandit_state.json нельзя: оркестратор сохраняет состояние на
        # шаге 0 (step % 50 == 0), и на прогоне в 30 шагов файл содержит
        # 1-2 вызова на агента. Это уже третий случай того же класса:
        # проба хардкодила бюджет, spread(mu) читалась из устаревшего файла,
        # теперь mu_role. Всё, что нужно проверить, должно писаться НА ШАГЕ.
        'bandit_mu_role': {a: {r: round(v, 6) for r, v in (st.mu_role or {}).items()}
                           for a, st in orch.bandit.agents.items()},
        'bandit_calls_role': {a: dict(st.calls_role or {})
                              for a, st in orch.bandit.agents.items()},
        'bandit_calls': {a: st.calls for a, st in orch.bandit.agents.items()},
        'bandit_energy': {a: round(st.energy, 4) for a, st in orch.bandit.agents.items()},
        'q_env_candidates': q_gen_ext,
        'q_env_best_candidate': best_gen_ext,
        'E_score_ext': q_syn_ext - best_gen_ext,
        'q_env_pre_depth': q_pre_depth,
        'depth_changed_answer': depth_changed,
        # тексты нужны для офлайн-контрфактика по §7.1 (см. lab/thermo_counterfactual.py)
        'gen_texts': [t[:3000] for t in gen_texts],
        'syn_texts': [t[:3000] for t in syn_texts],
        'q_env_synth_all': [q_env_ext(scorer, t, tests) for t in syn_texts],
        # ── нативные метрики Mycelium ──
        'q_ext_final': float(scorer.q_ext(syn_text, 'code')),
        'E_score_native': float(result['e_score']),
        'E_total': float(result['e_total']),
        'E_base': float(result['e_base']),
        'd_sem': float(result['d_sem']),
        'delta_I': float(result['delta_I']),
        'k_act': int(result['k_act']),
        'lambda_L': float(result['lambda_L']),
        'regime': result['regime'],
        'models': result['models'],
        # ── термодинамика шага ──
        'thermo': {k: (float(v) if not isinstance(v, bool) else v)
                   for k, v in thermo.items()},
        # ── гиперцикл ──
        'hypercycle_x_before': x_before,
        'hypercycle_x_after': dict(orch.hypercycle.x),
        'hypercycle_k': dict(orch.hypercycle.k),
        'alloc_log': alloc_line[0] if alloc_line else None,
        'chaos_log': lam_line[0] if lam_line else None,
        'depth': parse_depth(logbuf.lines),
        'beta_kl': float(orch.chaos.current_beta_kl),
        'alive_agents': sum(1 for a in orch.bandit.agents.values() if a.alive),
        # [§D] След гейта: ступени, что взято, что взяли бы s_ok[0] и
        # argmax q_ext. Сравнение старой и новой приёмки — внутри прогона.
        'gate': result.get('gate'),
        'q_env_pre_gate': q_env_ext(scorer, pre_gate, tests) if pre_gate else None,
    }
    rec.pop('tokens_prompt_est', None)
    return rec


# ── 8. Главный цикл ────────────────────────────────────────────────────────
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
            done[(r.get('task_id'), r.get('condition'))] = r
    return done


async def main_async(args):
    # [Ф1.3] Корпус — параметр, а не константа. corpus200.json является
    # НАДМНОЖЕСТВОМ corpus30.json (первые 30 задач те же, в том же порядке),
    # поэтому прогоны на 30 и на 200 сравнимы на общей подвыборке.
    corpus_file = args.corpus or ('corpus200.json' if args.n_tasks > 30
                                  else 'corpus30.json')
    path = os.path.join(HERE, 'data', corpus_file)
    corpus = json.load(open(path, encoding='utf-8'))['tasks']
    if args.n_tasks > len(corpus):
        raise SystemExit(f"в {corpus_file} только {len(corpus)} задач, "
                         f"запрошено {args.n_tasks}")
    print(f"корпус: {corpus_file}, взято {args.n_tasks} из {len(corpus)}")
    tasks = corpus[:args.n_tasks]

    out_path = os.path.join(HERE, 'data', f'results_{args.tag}.jsonl')
    done = load_done(out_path)
    if done:
        print(f"resume: уже записано {len(done)} пар (task,condition)")

    recorder = CallRecorder()
    orch_mod.call_model = recorder.wrap(orch_mod.call_model)

    logbuf = StepLogBuffer()
    logging.getLogger().addHandler(logbuf)

    conditions = args.conditions.split(',')
    m_tokens = {r['task_id']: r.get('tokens_completion', 0)
                for r in done.values() if r.get('condition') == 'M'}

    for cond in conditions:
        pending = [t for t in tasks if (t['task_id'], cond) not in done]
        if not pending:
            print(f"[{cond}] всё уже есть, пропуск")
            continue
        print(f"\n{'='*70}\n[{cond}] задач к прогону: {len(pending)}\n{'='*70}")

        orch = None
        scorer = None
        if not cond.startswith('BoN'):
            orch = make_orchestrator(cond, f'data/lab_{cond}')
            scorer = orch.scorer
        else:
            import yaml
            with open('config/settings.yaml', encoding='utf-8') as f:
                scorer = TestAwareScorer(yaml.safe_load(f))

        for t in tasks:
            key = (t['task_id'], cond)
            if key in done:
                continue
            t0 = time.monotonic()
            try:
                rec = await run_one(orch, scorer, cond, t, logbuf, recorder,
                                    m_token_budget=m_tokens.get(t['task_id']),
                                    acceptance=(args.acceptance != 'off'))
            except Exception as exc:
                import traceback
                traceback.print_exc()
                rec = {'condition': cond, 'task_id': t['task_id'],
                       'status': 'failed', 'error': f"{type(exc).__name__}: {exc}",
                       'seconds': round(time.monotonic() - t0, 1)}
            rec['ts'] = int(time.time())
            with open(out_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')
                f.flush()
                os.fsync(f.fileno())
            done[key] = rec
            if cond == 'M':
                m_tokens[t['task_id']] = rec.get('tokens_completion', 0)
            print(f"  [{cond}] task {t['task_id']}: status={rec.get('status')} "
                  f"q_env={rec.get('q_env_final')} "
                  f"E_ext={rec.get('E_score_ext')} "
                  f"tok={rec.get('tokens_completion')} "
                  f"{rec.get('seconds')}s", flush=True)

    print(f"\nготово: {out_path}")


def preflight(args):
    import urllib.request
    import torch
    free, total = (torch.cuda.mem_get_info() if torch.cuda.is_available()
                   else (0, 0))
    print("─" * 70)
    print(f"условия:        {args.conditions}")
    print(f"приёмка:        {args.acceptance} (общая для M и BoN, "
          f"core.acceptance.select)")
    print(f"задач:          {args.n_tasks}")
    print(f"VRAM свободно:  {free/2**30:.2f} / {total/2**30:.2f} GiB")
    try:
        import psutil
        print(f"RAM свободно:   {psutil.virtual_memory().available/2**30:.1f} GiB")
    except ImportError:
        pass
    try:
        with urllib.request.urlopen(SERVER_URL + '/models', timeout=5) as r:
            models = json.load(r)
        print(f"llm_server:     OK, модели {[m['id'] for m in models['data']]}")
    except Exception as exc:
        print(f"llm_server:     НЕДОСТУПЕН ({exc}) — запусти lab/llm_server.py")
        sys.exit(2)
    n_cond = len(args.conditions.split(','))
    calls = args.n_tasks * (n_cond - 1) * 8 + args.n_tasks * 6
    print(f"оценка вызовов: ~{calls}")
    print(f"оценка времени: ~{calls*30/60:.0f} мин "
          f"(при ~30 с на вызов: 256 новых токенов при ~9 ток/с)")
    print("─" * 70)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-tasks', type=int, default=3)
    ap.add_argument('--corpus', default='',
                    help='файл корпуса в lab/data; по умолчанию '
                         'corpus30.json при n<=30, иначе corpus200.json')
    ap.add_argument('--tag', type=str, default='smoke')
    ap.add_argument('--conditions', type=str, default=','.join(CONDITIONS))
    ap.add_argument('--acceptance', choices=['on', 'off'], default='on',
                    help='приёмка по исполнению (core.acceptance) в ОБЕИХ '
                         'ветках; off — прежнее правило s_ok[0] / argmax q_ext')
    a = ap.parse_args()
    preflight(a)
    asyncio.run(main_async(a))
