"""
core/bandit.py — Thompson Sampling + LinUCB
Σ_v8.8 «Entropic Mycelium»

Исправлено vs v6.6:
- precision_obs = 16 (не 4), соответствует σ_obs=0.25
- Exp3 как детектор дрейфа (не selector)
- LinUCB для температуры (не для выбора модели)
- Внешний осциллятор удалён; exploration остаётся в Thompson + LinUCB
"""

import numpy as np
import json
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class MemoryContext:
    """
    Типизированный контекст из памяти (FIX 7 — Gemini audit).
    Заменяет raw dict['priority_norm'] etc. → даёт static typing + явные defaults.
    Memory.search() возвращает List[dict]; конвертация — в orchestrator/bandit.
    """
    priority_norm: float = 0.0     # = e_total/(1+e_total) кандидата
    local_entropy: float = 0.0     # = H(PPR) если доступно, иначе e_total прокси
    topology_bias: float = 0.0     # PageRank блока в Physarum графе
    block_id: str = ""             # trace_id как строка (для дедупликации)

    @classmethod
    def from_dict(cls, d: dict) -> 'MemoryContext':
        """Безопасная конвертация из raw dict (backwards compat)."""
        e = float(d.get('e_total', 0.0) or 0.0)
        return cls(
            priority_norm=float(d.get('priority_norm', e / (1.0 + abs(e)))),
            local_entropy=float(d.get('local_entropy', e) or 0.0),
            topology_bias=float(d.get('topology_bias', 0.0)),
            block_id=str(d.get('id', '')),
        )


@dataclass
class AgentState:
    id: str
    mu: float = 0.5
    sigma: float = 0.5
    rewards: deque = field(default_factory=lambda: deque(maxlen=20))
    context_vec: np.ndarray = field(default_factory=lambda: np.zeros(16))
    energy: float = 4.0
    alive: bool = True
    calls: int = 0
    provider: str = ""
    family: str = ""
    roles: list = field(default_factory=list)
    max_rpm: int = 30
    last_call_times: deque = field(default_factory=lambda: deque(maxlen=60))

    # ── [Ш1.3c] Статистика по паре (агент, роль) ──────────────────────────
    # Дефект, найденный на прогоне sh1_30: награда стала пер-агентной, но не
    # пер-ролевой, а шкалы у ролей РАЗНЫЕ.
    #   G — абсолютное качество черновика, 0.15..1.0
    #   S — 0.5 + (синтез − лучший черновик), центрировано на 0.5
    #   C — 0.5 + (итог − итог без этой критики), тоже центрировано на 0.5
    # Агент, чаще тянувший синтез, получает mu ~ 0.5 независимо от того,
    # хорош он или плох, и это смешивается с его генераторскими наградами.
    # Наблюдалось прямо: mu поставила qwen3_17 (0.616) выше coder15_a (0.599),
    # хотя одиночный baseline даёт 0.569 против 0.684. Нижние три агента при
    # этом легли верно, ранговая корреляция +0.714 — то есть сигнал уже нёс
    # информацию, но верх путался.
    #
    # select() и так вызывается ПО КАСТАМ (allowed_roles=['G']), поэтому
    # статистика по (агент, роль) — не усложнение, а приведение состояния к
    # тому, как оно уже используется.
    mu_role: dict = field(default_factory=dict)
    sigma_role: dict = field(default_factory=dict)
    rewards_role: dict = field(default_factory=dict)
    calls_role: dict = field(default_factory=dict)

    # Exp3 детектор дрейфа
    exp3_w: float = 1.0
    exp3_mu_track: float = 0.5

    # [Ш1.3d] Частичный пул (shrinkage) пер-ролевой оценки к скалярной.
    #
    # Первая версия Ш1.3c брала mu_role как есть, и это дало результат ХУЖЕ
    # скалярной mu: rho(mu_role['G'], baseline) = +0.086 против +0.714.
    # Причина не в идее, а в объёме данных: разбиение по ролям делит выборку
    # втрое. Замерено на прогоне role2_30 — роль G получила 4-7 наблюдений на
    # агента, а при ema_alpha=0.047 это 17-28% пути от априорного 0.5 к
    # истинному значению. Все mu_G легли в 0.463..0.525, то есть остались
    # шумом вокруг 0.5.
    #
    # Стандартное лекарство — empirical Bayes: доверять ячейке пропорционально
    # тому, сколько в ней данных, а недостающее брать из скалярной оценки того
    # же агента (она усредняет все роли, но данных в ней втрое больше).
    #
    #     w = n_role / (n_role + k),  k = POOL_K
    #     mu_eff = w * mu_role + (1 - w) * mu
    #
    # При n=0 это ровно скалярная mu (то есть поведение sh1_30, rho=+0.714),
    # при n >> k — чистая пер-ролевая. Деградация плавная в обе стороны.
    POOL_K = 8.0

    def role_mu(self, role: str, pooled: bool = False) -> float:
        m = self.mu_role.get(role)
        if m is None:
            return self.mu
        if not pooled:
            return m
        n = float(self.calls_role.get(role, 0))
        w = n / (n + self.POOL_K)
        return w * m + (1.0 - w) * self.mu

    def role_sigma(self, role: str, pooled: bool = False) -> float:
        s = self.sigma_role.get(role)
        if s is None:
            return self.sigma
        if not pooled:
            return s
        n = float(self.calls_role.get(role, 0))
        w = n / (n + self.POOL_K)
        # У холодной ячейки неопределённость должна быть НЕ МЕНЬШЕ скалярной:
        # мало данных — меньше уверенности, а не больше.
        return max(w * s + (1.0 - w) * self.sigma, self.sigma * (1.0 - w) + s * w)


@dataclass
class ProviderHealth:
    consecutive_failures: int = 0
    cooldown_until: float = 0.0
    success_ema: float = 1.0
    avg_latency: float = 0.0


class TokenBucket:
    """Rate limiter на провайдера"""
    def __init__(self, rpm: int):
        self.rate = rpm / 60.0          # токенов/сек
        self.tokens = float(rpm)
        self.max_tokens = float(rpm)
        self.last_refill = time.monotonic()

    def can_call(self) -> bool:
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.max_tokens, self.tokens + elapsed * self.rate)
        self.last_refill = now
        return self.tokens >= 1.0

    def consume(self):
        self.tokens -= 1.0


class LinUCBArm:
    """LinUCB для выбора температуры (не моделей)"""
    def __init__(self, alpha: float = 0.62, d: int = 16):
        self.alpha = alpha
        self.A = np.eye(d)                       # инициализация I_d
        self.A_inv = np.eye(d)                   # храним инверсию
        self.b = np.zeros(d)
        self.theta = np.zeros(d)

    def score(self, x: np.ndarray) -> float:
        return x @ self.theta + self.alpha * np.sqrt(x @ self.A_inv @ x)

    def update(self, x: np.ndarray, r: float):
        # Sherman-Morrison: O(d²) вместо O(d³)
        v = self.A_inv @ x
        denom = 1.0 + x @ v
        self.A_inv -= np.outer(v, v) / denom
        self.b += r * x
        self.theta = self.A_inv @ self.b


class TemperatureBandit:
    """LinUCB выбирает температуру для API вызовов"""
    def __init__(self, temps: List[float], alpha: float, d: int):
        self.temps = temps
        self.arms: Dict[float, LinUCBArm] = {t: LinUCBArm(alpha, d) for t in temps}

    def select(self, context: np.ndarray) -> float:
        scores = {t: arm.score(context) for t, arm in self.arms.items()}
        return max(scores, key=scores.get)

    def update(self, temp: float, context: np.ndarray, reward: float):
        self.arms[temp].update(context, reward)


class SwarmBandit:
    """
    Главный бандит роя.
    Thompson Sampling — выбор моделей (Байесовский, предполагает стационарность).
    Exp3 — детектор дрейфа (адверсариальный, сигнализирует при расхождении).
    LinUCB — выбор температуры (контекстный).
    """
    def __init__(self, cfg: dict, models: List[dict]):
        self.cfg = cfg
        bnd = cfg['bandit']
        self.sigma_min = np.sqrt(bnd['thompson']['sigma_min']**2)
        self.ema_alpha = bnd['thompson']['ema_alpha']
        self.precision_obs = bnd['thompson']['precision_obs']  # = 16
        self.max_per_provider = bnd['max_per_provider']
        self.provider_cooldown_sec = float(bnd.get('provider_cooldown_sec', 60.0))

        # ── [Ш1.4] Флаги правок, по умолчанию выключены ────────────────────
        fx = cfg.get('fix', {}) or {}
        # R1b: update() пере-выбирал лучшую руку LinUCB и обновлял её вместо
        # сыгранной. Самоподтверждающийся контур: лидер не может проиграть,
        # контрфактика не собирается.
        self.linucb_played_arm = bool(fx.get('linucb_played_arm', False))
        # R10: два знаковых дефекта в select().
        self.selection_sign_fix = bool(fx.get('selection_sign_fix', False))
        # Ш1.3c: (агент, роль) вместо агента. См. AgentState.mu_role.
        self.per_role_mu = bool(fx.get('per_role_mu', False))
        # Ш1.3d: частичный пул холодных ячеек к скалярной mu. Без него
        # пер-ролевая оценка на 30 шагах хуже скалярной (rho +0.086 против
        # +0.714): 4-7 наблюдений на ячейку при ema_alpha=0.047 — это шум
        # вокруг априорного 0.5. См. AgentState.role_mu.
        self.per_role_pool = bool(fx.get('per_role_pool', False))

        # Инициализация агентов
        self.agents: Dict[str, AgentState] = {}
        self.buckets: Dict[str, TokenBucket] = {}
        self.provider_health: Dict[str, ProviderHealth] = {}
        for m in models:
            a = AgentState(
                id=m['id'], mu=m.get('mu', 0.5), sigma=m.get('sigma', 0.5),
                provider=m['provider'], family=m['family'],
                roles=m.get('roles', ['G']), max_rpm=m.get('max_rpm', 30),
                alive=bool(m.get('alive', True))
            )
            self.agents[m['id']] = a
            self.provider_health.setdefault(m['provider'], ProviderHealth())
            if m['provider'] not in self.buckets:
                self.buckets[m['provider']] = TokenBucket(m['max_rpm'])

        # LinUCB для температуры
        lu = bnd['linucb']
        self.temp_bandit = TemperatureBandit(lu['arms'], lu['alpha'], lu['context_dim'])

        self.step = 0
        self.lambda_L = 0.0
        self.context_complexity = 0.0

    def set_lambda(self, lL: float):
        self.lambda_L = lL

    def set_memory_context(self, retrieved_blocks=None,
                           context_complexity: Optional[float] = None):
        """
        FIX 7: принимает List[dict] или List[MemoryContext].
        Конвертирует raw dict → MemoryContext для типобезопасного доступа.
        """
        if context_complexity is None:
            # Нормализация входа: поддержка dict и MemoryContext
            blocks_typed: List[MemoryContext] = []
            for b in (retrieved_blocks or []):
                if isinstance(b, MemoryContext):
                    blocks_typed.append(b)
                elif isinstance(b, dict):
                    blocks_typed.append(MemoryContext.from_dict(b))
                # иначе — пропустить

            if blocks_typed:
                p = float(np.mean([b.priority_norm for b in blocks_typed]))
                h = float(np.mean([b.local_entropy / (1.0 + abs(b.local_entropy))
                                   for b in blocks_typed]))
                t = float(np.mean([b.topology_bias for b in blocks_typed]))
                ids = {b.block_id for b in blocks_typed}
                diversity = len(ids) / max(1, len(blocks_typed))
                context_complexity = 0.4 * p + 0.3 * h + 0.2 * t + 0.1 * diversity
            else:
                context_complexity = 0.0

        self.context_complexity = float(np.clip(context_complexity, 0.0, 1.0))

    def record_provider_result(self, provider: str, ok: bool, latency: float = 0.0):
        h = self.provider_health.setdefault(provider, ProviderHealth())
        latency = float(np.clip(latency or 0.0, 0.0, 300.0))
        h.avg_latency = latency if h.avg_latency <= 0 else 0.8 * h.avg_latency + 0.2 * latency
        h.success_ema = 0.9 * h.success_ema + 0.1 * (1.0 if ok else 0.0)
        if ok:
            h.consecutive_failures = 0
            if time.monotonic() >= h.cooldown_until:
                h.cooldown_until = 0.0
        else:
            h.consecutive_failures += 1
            if h.consecutive_failures >= 3:
                h.cooldown_until = time.monotonic() + self.provider_cooldown_sec

    def provider_available(self, provider: str) -> bool:
        h = self.provider_health.setdefault(provider, ProviderHealth())
        if h.cooldown_until and time.monotonic() < h.cooldown_until:
            return False
        if h.cooldown_until and time.monotonic() >= h.cooldown_until:
            h.cooldown_until = 0.0
        return True

    def _provider_latency_penalty(self, provider: str) -> float:
        h = self.provider_health.setdefault(provider, ProviderHealth())
        if h.avg_latency <= 0:
            return 1.0
        return float(np.clip(1.0 / (1.0 + h.avg_latency / 30.0), 0.65, 1.0))

    def _context_16d(self, task_vec: np.ndarray) -> np.ndarray:
        """PCA-lite: берём первые 16 компонент через SVD или просто первые 16 dims"""
        if len(task_vec) >= 16:
            return task_vec[:16] / (np.linalg.norm(task_vec[:16]) + 1e-9)
        return np.zeros(16)

    def select(self, k: int, task_vec: np.ndarray,
               allowed_roles: Optional[List[str]] = None) -> List[str]:
        """
        Выбрать k моделей через Thompson Sampling.
        Ограничения: max_per_provider, rate limits, alive.
        """
        ctx = self._context_16d(task_vec)
        samples = {}
        for aid, agent in self.agents.items():
            if not agent.alive:
                continue
            if allowed_roles and not any(r in agent.roles for r in allowed_roles):
                continue
            if not self.provider_available(agent.provider):
                continue
            if not self.buckets[agent.provider].can_call():
                continue

            # Контекстный prior: μ + c^T · x
            # [Ш1.3c] При включённом per_role_mu берём статистику той роли,
            # под которую идёт отбор. select() и так вызывается по одной касте
            # за раз (allowed_roles=['G'] и т.д.), так что роль однозначна.
            if self.per_role_mu and allowed_roles and len(allowed_roles) == 1:
                _role = allowed_roles[0]
                base_mu = agent.role_mu(_role, pooled=self.per_role_pool)
                base_sigma = agent.role_sigma(_role, pooled=self.per_role_pool)
            else:
                base_mu, base_sigma = agent.mu, agent.sigma
            q_ctx = base_mu + float(agent.context_vec @ ctx)
            # Связь B: λ → exploration_boost → увеличивает σ при хаосе (ТЗ §B)
            # exploration_boost = clamp(0.2 + 0.5*λ, 0.0, 1.0)
            lL = self.lambda_L
            exploration_boost = float(np.clip(0.2 + 0.5 * lL, 0.0, 1.0))
            context_modifier = 1.0 + 0.2 * self.context_complexity
            sigma_eff = max(base_sigma * (1.0 + exploration_boost * 0.5),
                            self.sigma_min)
            sigma_eff *= context_modifier
            raw = np.random.normal(q_ctx, sigma_eff)
            pen = self._provider_latency_penalty(agent.provider)
            if self.selection_sign_fix:
                # [Ш1.4b] Штраф применялся УМНОЖЕНИЕМ на сэмпл. Сэмпл может
                # быть отрицательным (mu ~ 0.5, sigma до 0.5+), и тогда
                # умножение на 0.65 делает его БОЛЬШЕ — штраф превращался в
                # бонус ровно для тех агентов, чьи оценки просели. Штраф —
                # аддитивный сдвиг, а не масштаб.
                samples[aid] = raw - (1.0 - pen)
            else:
                samples[aid] = raw * pen

        if not samples:
            # Fallback: любая локальная модель. Префикс, а не точное 'local':
            # провайдеры разделены по моделям (local_coder15, ...), чтобы
            # max_per_provider снова что-то ограничивал.
            return [aid for aid, a in self.agents.items()
                    if a.provider == 'local' or a.provider.startswith('local_')][:1]

        # Сортировка по выборке, с ограничением на провайдера
        # Связь B: λ → selection_pressure = 1/(1+λ) (ТЗ §B)
        # При λ>0 (хаос): давление снижается → более случайный выбор
        # При λ<0 (порядок): давление растёт → более жадный выбор
        lL_abs = abs(self.lambda_L)
        selection_pressure = 1.0 / (1.0 + lL_abs)
        # Применяем: сглаживаем разрыв топ/остальных через temperature-softmax
        if samples:
            vals = np.array(list(samples.values()))
            # Нормируем через selection_pressure как температуру
            T_sel = max(selection_pressure, 0.05)
            exp_v = np.exp((vals - vals.max()) / T_sel)
            probs = exp_v / exp_v.sum()
            keys  = list(samples.keys())
            if self.selection_sign_fix:
                # [Ш1.4b] Было: rank_noise = uniform(0,0.1) * probs, где probs —
                # softmax. У лидера probs максимальна, значит ему доставалось
                # БОЛЬШЕ всего шума — обратно замыслу «сгладить разрыв топ и
                # остальных». Шум должен расти к хвосту, а не к голове.
                rank_noise = np.random.uniform(0, 0.1, len(vals)) * (1.0 - probs)
            else:
                rank_noise = np.random.uniform(0, 0.1, len(vals)) * probs
            ordered = sorted(zip(keys, vals + rank_noise), key=lambda x: -x[1])
            ranked = [(k, v) for k, v in ordered]
        else:
            ranked = []
        selected = []
        provider_counts: Dict[str, int] = {}

        for aid, _ in ranked:
            if len(selected) >= k:
                break
            p = self.agents[aid].provider
            if provider_counts.get(p, 0) >= self.max_per_provider:
                continue
            selected.append(aid)
            provider_counts[p] = provider_counts.get(p, 0) + 1
            self.buckets[p].consume()

        return selected

    def update(self, selected: List[str], rewards: Dict[str, float],
               task_vec: np.ndarray, temperature: Optional[float] = None,
               roles_played: Optional[Dict[str, str]] = None):
        """Обновить Thompson + Exp3 детектор + контекстный вектор.

        temperature — температура, РЕАЛЬНО использованная в шаге. Нужна для
        Ш1.4: без неё update пере-выбирает лучшую руку LinUCB и обновляет её,
        то есть кредит достаётся не сыгранному действию.

        roles_played — какую касту агент реально отыграл на этом шаге. Нужно
        для Ш1.3c: награды разных ролей лежат в разных шкалах, и складывать
        их в одну mu значит терять сигнал (см. AgentState.mu_role).
        """
        ctx = self._context_16d(task_vec)
        roles_played = roles_played or {}

        for aid in selected:
            if aid not in self.agents:
                continue
            agent = self.agents[aid]
            r = rewards.get(aid, 0.0)
            agent.rewards.append(r)
            agent.calls += 1

            # ── Thompson Update ──
            # precision_obs = 16 соответствует σ_obs = 0.25
            var_r = np.var(list(agent.rewards)) if len(agent.rewards) > 2 else 0.25
            tau_hat = 1.0 / (var_r + 0.01)
            sigma2_new = max(1.0 / (1.0 / agent.sigma**2 + tau_hat),
                             self.sigma_min**2)
            agent.sigma = np.sqrt(sigma2_new)
            agent.mu += self.ema_alpha * (r - agent.mu)

            # ── [Ш1.3c] то же самое, но раздельно по роли ──────────────────
            role = roles_played.get(aid)
            if role:
                rr = agent.rewards_role.setdefault(role, deque(maxlen=20))
                rr.append(r)
                agent.calls_role[role] = agent.calls_role.get(role, 0) + 1
                s_prev = agent.sigma_role.get(role, agent.sigma
                                              if not agent.mu_role else 0.5)
                if role not in agent.sigma_role:
                    s_prev = max(agent.sigma, 0.5)    # тёплый старт по sigma
                var_rr = np.var(list(rr)) if len(rr) > 2 else 0.25
                tau_rr = 1.0 / (var_rr + 0.01)
                s2 = max(1.0 / (1.0 / max(s_prev, 1e-6) ** 2 + tau_rr),
                         self.sigma_min ** 2)
                agent.sigma_role[role] = float(np.sqrt(s2))
                # Тёплый старт: новая ячейка стартует от скалярной оценки
                # агента, а не от неинформативного 0.5. Иначе первые
                # наблюдения тратятся на то, чтобы уползти от априора.
                m_prev = agent.mu_role.get(role, agent.mu)
                agent.mu_role[role] = float(
                    m_prev + self.ema_alpha * (r - m_prev))

            # Контекстный вектор: медленное обновление
            agent.context_vec += 0.001 * (r - agent.mu) * ctx

            # ── Exp3 детектор дрейфа ──
            n = len(self.agents)
            t = max(self.step, 1)
            eta_exp3 = np.sqrt(np.log(n) / (n * t)) if n > 0 else 0.001
            agent.exp3_w *= np.exp(eta_exp3 * r)
            # Отслеживаем μ Exp3
            agent.exp3_mu_track += eta_exp3 * (r - agent.exp3_mu_track)

            # Сброс уверенности при дрейфе
            drift = abs(agent.mu - agent.exp3_mu_track)
            if drift > 0.15:
                agent.sigma = max(agent.sigma, 0.5)  # сброс к sigma_init

        # Обновить LinUCB температуры
        mean_r = np.mean(list(rewards.values())) if rewards else 0.5
        if self.linucb_played_arm and temperature is not None:
            # [Ш1.4] кредит сыгранной руке. Ближайшая по значению — потому что
            # orchestrator домножает температуру на 0.8/0.7 для C/S каст, и
            # точного совпадения с arms может не быть.
            played = min(self.temp_bandit.arms,
                         key=lambda t: abs(t - float(temperature)))
        else:
            # Прежнее поведение: пере-выбрать лучшую и обновить её.
            played = self.temp_bandit.select(ctx)
        self.temp_bandit.update(played, ctx, mean_r)

        self.step += 1

    def get_temperature(self, task_vec: np.ndarray) -> float:
        ctx = self._context_16d(task_vec)
        return self.temp_bandit.select(ctx)

    def kill(self, agent_id: str):
        if agent_id in self.agents:
            self.agents[agent_id].alive = False

    def split(self, parent_id: str) -> Optional[str]:
        """Репликация с мутацией при energy > split_threshold"""
        if parent_id not in self.agents:
            return None
        parent = self.agents[parent_id]
        child_id = f"{parent_id}_child_{self.step}"
        child = AgentState(
            id=child_id,
            mu=parent.mu,
            # Мутация: частичный антипод для diversity
            sigma=max(0.4, parent.sigma * 1.1),
            provider=parent.provider,
            family=parent.family,
            roles=parent.roles,
            max_rpm=parent.max_rpm,
            energy=4.0      # стоимость деления уже вычтена снаружи
        )
        self.agents[child_id] = child
        return child_id

    def save(self, path: str):
        state = {}
        for aid, a in self.agents.items():
            state[aid] = {'mu': a.mu, 'sigma': a.sigma,
                          'energy': a.energy, 'alive': a.alive,
                          'calls': a.calls,
                          # [Ш1.3c] без этого пер-ролевая статистика
                          # терялась при каждом перезапуске
                          'mu_role': dict(a.mu_role),
                          'sigma_role': dict(a.sigma_role),
                          'calls_role': dict(a.calls_role)}
        with open(path, 'w') as f:
            json.dump(state, f)

    def load(self, path: str):
        try:
            with open(path) as f:
                state = json.load(f)
            for aid, s in state.items():
                if aid in self.agents:
                    self.agents[aid].mu = s['mu']
                    self.agents[aid].sigma = s['sigma']
                    self.agents[aid].energy = s['energy']
                    self.agents[aid].alive = s['alive']
                    self.agents[aid].calls = s['calls']
                    self.agents[aid].mu_role = dict(s.get('mu_role', {}))
                    self.agents[aid].sigma_role = dict(s.get('sigma_role', {}))
                    self.agents[aid].calls_role = dict(s.get('calls_role', {}))
        except FileNotFoundError:
            pass
