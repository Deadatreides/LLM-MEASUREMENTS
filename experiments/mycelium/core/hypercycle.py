"""
core/hypercycle.py — Гиперцикл Эйгена
Σ_v8.5 «Мицелий»

Аналогия: рибозимы протоклетки.
РНК-полимераза катализирует копирование другой РНК,
та в свою очередь катализирует третью, замыкая цикл.

Три касты: G (Генераторы) → C (Критики) → S (Синтезаторы) → G
Нормализация предотвращает взрыв концентраций.

Ключевое исправление vs v7.4:
- max(Φ, phi_min) в знаменателе защищает от деления на ~0
- clip [x_min, x_max] на выходе
- k_i обновляется по EMA от среднего E_i касты
"""

import numpy as np
from typing import Dict, List


class Hypercycle:
    """
    Мультипликативный гиперцикл с нормализацией.
    x_i — доля квоты i-й касты в K_ACT.
    k_i — каталитическая константа касты.

    Уравнение:
        x_i(t+1) = x_i(t) * (1 + α * k_i * x_{i-1}(t)) / (1 + α * Φ)
        Φ = Σ_j k_j * x_j * x_{j-1}
    """
    CASTES = ['G', 'C', 'S']  # фиксированный порядок для i-1 цикла

    def __init__(self, cfg: dict):
        hc = cfg['hypercycle']
        self.alpha = hc['alpha']
        self.kappa_decay = hc['kappa_decay']
        self.kappa_gain = hc['kappa_gain']
        self.x_min = hc['x_min']
        self.x_max = hc['x_max']
        self.phi_min = hc['phi_min']

        # ── [Ш2.3] Флаги правок, по умолчанию выключены ────────────────────
        fx = cfg.get('fix', {}) or {}
        # Мутация: единственный реальный рычаг. При 0.03 возврат к 1/3 идёт со
        # скоростью 0.03/шаг, а каталитический дифференциал даёт ~0.0014/шаг —
        # сигнал забивается примерно в 20 раз. Замерено: lab/sim_hypercycle.py.
        self.mutation = (float(fx.get('hypercycle_mutation_rate', 0.005))
                         if fx.get('hypercycle_mutation') else 0.03)
        # Квантователь: round(x*K) требует |dx| >= 1/(2K) = 0.0833 при K=6,
        # то есть рубит непрерывную величину до того, как она успеет что-то
        # сказать. Multinomial(K, x) пропускает сигнал непрерывно.
        self.multinomial_alloc = bool(fx.get('hypercycle_multinomial', False))
        self._rng = np.random.default_rng()

        # ВНИМАНИЕ: правки «убрать alpha из Phi» здесь нет намеренно. Phi —
        # общий для всех каст знаменатель, а следующая строка делит на
        # sum(x_new), и общий множитель сокращается ТОЧНО. То есть Phi и
        # phi_min не влияют ни на что, и защита max(Phi, phi_min) из аудита
        # v8.5 охраняет деление, которое не может повлиять на результат.
        # Доказано численно: tests/test_mechanisms.py::test_phi_is_inert.

        # Инициализация: равные доли
        self.x: Dict[str, float] = {c: 1/3 for c in self.CASTES}
        self.k: Dict[str, float] = {c: 0.5 for c in self.CASTES}
        # Связь #2: λ → structural coordination (ТЗ §2)
        self._lambda_L: float = 0.0

    def set_lambda(self, lambda_L: float):
        """
        Связь #2: принять λ из chaos controller.
        λ влияет на active_links / coordination через effective_alpha.
        active_links ∝ alpha_eff = alpha_base * (1 + c * |λ|)
        """
        self._lambda_L = float(lambda_L)

    def update(self, mean_e: Dict[str, float]) -> Dict[str, float]:
        """
        mean_e: средний E_total каждой касты за прошлый шаг
        Возвращает обновлённые x (квоты).
        """
        # Шаг 1: обновление каталитических констант по EMA
        for c in self.CASTES:
            if c in mean_e:
                self.k[c] = (self.kappa_decay * self.k[c] +
                              self.kappa_gain * max(0.0, mean_e[c]))

        # Шаг 2: Φ = Σ k_j * x_j * x_{j-1}
        # Связь #2: alpha_eff = alpha_base * (1 + c*|λ|) (ТЗ §2, Вариант A/C)
        # При λ=0: alpha_eff = alpha_base (нейтрально)
        # При λ>0 (хаос): alpha_eff растёт → более сильная координация каст
        # При λ<0 (порядок): alpha_eff снижается → меньше coordination
        c_link = 0.5
        alpha_eff = self.alpha * (1.0 + c_link * abs(self._lambda_L))
        alpha_eff = float(np.clip(alpha_eff, 0.001, 0.5))  # не взрывать

        castes = self.CASTES
        phi = sum(
            alpha_eff * self.k[castes[j]] * self.x[castes[j]] * self.x[castes[j-1]]
            for j in range(len(castes))
        )
        phi = max(phi, self.phi_min)

        # Шаг 3: мультипликативное обновление
        x_new = {}
        for j, c in enumerate(castes):
            prev = castes[j-1]
            x_prev = self.x[prev]
            x_new[c] = self.x[c] * (1 + alpha_eff * self.k[c] * x_prev) / (1 + alpha_eff * phi)

        # Шаг 4: нормализация и клип
        total = sum(x_new.values())
        if total > 0:
            for c in castes:
                x_new[c] = np.clip(x_new[c] / total, self.x_min, self.x_max)
        else:
            x_new = {c: 1/3 for c in castes}

        # Мутация: шум-пол, не даёт касте вымереть. Коэффициент — параметр
        # (self.mutation), а не константа 0.03: см. __init__.
        mu = self.mutation
        for c in castes:
            x_new[c] = (1.0 - mu) * x_new[c] + mu / len(castes)

        total = sum(x_new.values())
        self.x = {c: v / total for c, v in x_new.items()}

        return dict(self.x)

    def allocate(self, k_act: int) -> Dict[str, int]:
        """
        Распределить K_ACT моделей по кастам согласно x.
        Гарантирует минимум 1 на касту.
        """
        if self.multinomial_alloc:
            return self._allocate_multinomial(k_act)
        raw = {c: max(1, round(self.x[c] * k_act)) for c in self.CASTES}
        # Подгонка суммы до k_act
        total = sum(raw.values())
        diff = k_act - total
        if diff > 0:
            # Добавить в наибольшую касту
            biggest = max(raw, key=raw.get)
            raw[biggest] += diff
        elif diff < 0:
            # Убрать из наименьшей касты (минимум 1)
            for _ in range(abs(diff)):
                shrinkable = [c for c in self.CASTES if raw[c] > 1]
                if shrinkable:
                    smallest = min(shrinkable, key=raw.get)
                    raw[smallest] -= 1
        return raw

    def _allocate_multinomial(self, k_act: int) -> Dict[str, int]:
        """[Ш2.3] k ~ Multinomial(K_ACT, x), с гарантией минимум 1 на касту.

        Предписано в «Анализе Sigma_v6.7» («вектор ресурсов распределяется по
        Multinomial, а не в один блок»). Смысл в том, что round(x*K) — жёсткий
        квантователь с порогом 1/(2K): при K=6 нужно |dx| >= 0.0833, чтобы
        раскладка вообще изменилась. Мультиномиальная выборка передаёт сигнал
        непрерывно: сильнейшая каста получает больше агентов чаще, а не
        «только после того, как перешагнёт ступеньку».
        """
        n_min = len(self.CASTES)
        if k_act <= n_min:
            return {c: 1 for c in self.CASTES}
        p = np.array([self.x[c] for c in self.CASTES], dtype=float)
        s = p.sum()
        p = p / s if s > 0 else np.full(len(self.CASTES), 1.0 / len(self.CASTES))
        draw = self._rng.multinomial(k_act - n_min, p)
        return {c: 1 + int(draw[i]) for i, c in enumerate(self.CASTES)}



class NoveltySignal:
    """
    Вычисляет R_ext = новизна задачи относительно памяти.

    GPT fix: «Энергия замкнута внутри системы» →
    нужен сигнал из ВНЕШНЕГО мира.

    R_ext = расстояние до ближайшего соседа в FAISS (L2-норма).
    Высокая новизна → задача не встречалась → больше энергии на исследование.

    Аналогия Пригожина: R_ext ≈ d_e S / dt
    (поток негэнтропии из внешней среды, Условие: |d_eS/dt| > σ)
    """

    def __init__(self, scale: float = 0.5):
        self.scale = scale
        self.baseline = 0.5
        self.ema_alpha = 0.01

    def compute(self, vec, faiss_index, k_neighbors: int = 1) -> float:
        """Новизна ∈ [0, 1]. 0 = видели; 1 = новая задача."""
        import numpy as np
        if faiss_index is None or faiss_index.ntotal < 2:
            return 0.5
        v = vec.reshape(1, -1).astype('float32')
        k = min(k_neighbors + 1, faiss_index.ntotal)
        dists, _ = faiss_index.search(v, k)
        min_dist = float(dists[0][0]) if dists[0][0] > 0 else float(dists[0][-1])
        novelty = float(np.clip(min_dist / 2.0, 0.0, 1.0))
        self.baseline = (1 - self.ema_alpha) * self.baseline + self.ema_alpha * novelty
        return novelty

    def r_ext(self, novelty: float) -> float:
        """Энергетический бонус ∈ [0, scale]."""
        return self.scale * novelty


class AgentEnergy:
    """
    Энергетическая модель отдельного агента.
    Аналогия: популяционная динамика — агент умирает или делится.

    Уравнение:
        energy_{t+1} = decay * energy_t + scale * r_eff - cost + bonus * x_prev/x_mean
    """
    def __init__(self, cfg: dict):
        e = cfg['energy']
        self.decay = e['decay']
        self.scale = e['reward_scale']
        self.external_scale = e.get('external_scale', 0.5)
        self.quadratic_decay = e.get('quadratic_decay', 0.02)
        self.bonus = e['hypercycle_bonus']
        self.death = e['death_threshold']
        self.split_th = e['split_threshold']
        self.split_cost = e['split_cost']
        self.fs_k = e['fitness_sharing_k']
        self.div_bonus = e['diversity_bonus']

    def r_eff(self, r: float, n_same: int, is_diverse: bool) -> float:
        """
        Fitness sharing + diversity bonus.
        n_same: количество вызовов похожих моделей (штраф за монополию)
        is_diverse: cos с соседями > 0.82 (бонус за уникальность)
        """
        fs = r / (1 + self.fs_k * np.log1p(n_same))
        bonus = self.div_bonus if is_diverse else 0.0
        return fs * (1 + bonus)

    def step(self, energy: float, r: float, n_same: int,
             is_diverse: bool, tokens_out: int, x_prev: float,
             x_mean: float, r_ext: float = 0.0) -> tuple:
        """
        Возвращает (new_energy, event)
        event: None | 'death' | 'split'
        """
        re = self.r_eff(r, n_same, is_diverse)
        cost = tokens_out / 1000.0
        # FIX: нормировать через отклонение от равновесия (1/3), не через сырое деление
        # Старая: hc_bonus = 0.11 * (0.87/0.33) = 0.287 → равно cost при 287 токенах → нейтрально
        # Новая:  hc_bonus = 0.11 * (0.87-0.33)*3 = 0.178 → поощряет РОСТ доли, не абсолютный размер
        hc_bonus = self.bonus * (x_prev - (1.0/3.0)) * 3.0
        if self.decay >= 1.0:
            raise ValueError("energy.decay must be < 1 for bounded energy")
        if self.quadratic_decay <= 0.0:
            raise ValueError("energy.quadratic_decay must be > 0 for bounded energy")

        new_e = (self.decay * energy + self.scale * re +
                 self.external_scale * r_ext - cost + hc_bonus -
                 self.quadratic_decay * energy**2)

        if new_e <= self.death:
            return new_e, 'death'
        elif new_e >= self.split_th:
            return new_e - self.split_cost, 'split'
        else:
            return new_e, None
