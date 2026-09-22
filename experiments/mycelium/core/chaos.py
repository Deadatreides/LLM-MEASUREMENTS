"""
core/chaos.py — Управление краем хаоса
Σ_v8.5 «Мицелий»

Аналогия: λ_L = показатель Ляпунова для нелинейной динамической системы.
λ_L > 0 → хаос (расходимость траекторий)
λ_L < 0 → стабильность (схождение)
λ_L ≈ 0 → КРАЙ ХАОСА — максимальная информационная ёмкость (Лэнгтон 1990)

PID удерживает систему на краю через K_ACT и β_KL.
"""

import numpy as np
from collections import deque


class LyapunovEstimator:
    """
    Оценка наибольшего показателя Ляпунова по временному ряду.
    λ_L(t) = (1/N) Σ ln(|ΔE_k| / |ΔE_{k-1}| + ε)

    ДЕФЕКТ ЭТОЙ ФОРМЫ (найден в Ш2.2): сумма ТЕЛЕСКОПИРУЕТСЯ.

        Σ_{i} ln(d_i / d_{i-1}) = ln(d_last / d_first)

    то есть λ_L = ln(d_last/d_first)/N и определяется ДВУМЯ граничными
    дельтами окна из двадцати — путь между ними не влияет вообще. Проверено:
    те же 40 дельт в перемешанном порядке дают −0.808 вместо +0.599.

    Практическое следствие: одна случайно малая |ΔE| на любом краю окна
    швыряет оценку на порядок. Наблюдавшийся в прогоне диапазон
    [−2.089, +0.155] — это в основном шум границ, а не режим системы.
    Управлять по такой величине нельзя, и K_ACT в этом смысле повезло, что
    PID до него не доходил.

    fix.lyapunov_regression=True переводит оценку на наклон наименьших
    квадратов ln|ΔE_k| по k: используются все точки окна, телескопирования
    нет. Это стандартный способ оценки показателя (Rosenstein и др.) и он
    отвечает на тот вопрос, который формула заявляет.
    """
    def __init__(self, window: int = 20, eps: float = 1e-6,
                 regression: bool = False):
        self.window = window
        self.eps = eps
        self.regression = regression
        self.e_history = deque(maxlen=window + 2)

    def update(self, e_total: float) -> float:
        self.e_history.append(e_total)
        if len(self.e_history) < 3:
            return 0.0

        e = list(self.e_history)
        deltas = [abs(e[i] - e[i-1]) for i in range(1, len(e))]

        if len(deltas) < 2:
            return 0.0

        if self.regression:
            # Наклон ln|ΔE| по номеру шага. Все точки окна участвуют.
            y = np.log(np.asarray(deltas, dtype=float) + self.eps)
            x = np.arange(len(y), dtype=float)
            if len(y) < 3 or not np.isfinite(y).all():
                return 0.0
            slope = float(np.polyfit(x, y, 1)[0])
            return slope

        ratios = []
        for i in range(1, len(deltas)):
            ratio = (deltas[i] + self.eps) / (deltas[i-1] + self.eps)
            if ratio > 0:
                ratios.append(np.log(ratio))

        return float(np.mean(ratios)) if ratios else 0.0


class PIDController:
    """
    Дискретный PID с anti-windup.
    Цель: λ_L → 0 (край хаоса)
    Управляет: K_ACT и β_KL
    """
    def __init__(self, kp=0.5, ki=0.1, kd=0.05, integral_decay=0.97,
                 k_center=6, k_min=3, k_max=9):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.decay = integral_decay
        self.k_center = k_center
        self.k_min = k_min
        self.k_max = k_max

        self.integral = 0.0
        self.prev_error = 0.0
        self.initialized = False

    def update(self, lambda_L: float):
        """
        Возвращает (K_ACT, β_KL, u_raw)
        """
        e = -lambda_L          # цель λ_L = 0, ошибка = 0 - λ_L

        # Anti-windup: экспоненциальное затухание интеграла
        self.integral = self.decay * self.integral + e

        if not self.initialized:
            de = 0.0
            self.initialized = True
        else:
            de = e - self.prev_error

        u = self.kp * e + self.ki * self.integral + self.kd * de
        self.prev_error = e

        k_act = int(np.clip(round(self.k_center + u), self.k_min, self.k_max))

        # β_KL адаптируется к хаосу: при λ_L > 0 увеличиваем KL-регуляризацию
        beta_kl = float(np.clip(0.05 + 0.1 * max(0.0, lambda_L), 0.05, 0.30))

        return k_act, beta_kl, float(u)

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.initialized = False


class ChaosController:
    """Объединяет Ляпунова + PID в одном интерфейсе."""

    def __init__(self, cfg: dict):
        c = cfg['chaos']
        _fx = cfg.get('fix', {}) or {}
        self.lyapunov = LyapunovEstimator(
            window=c['lyapunov']['window'],
            eps=c['lyapunov']['eps'],
            regression=bool(_fx.get('lyapunov_regression', False)),
        )
        self.pid = PIDController(
            kp=c['pid']['kp'], ki=c['pid']['ki'], kd=c['pid']['kd'],
            integral_decay=c['pid']['integral_decay'],
            k_center=c['pid']['k_act_center'],
            k_min=c['pid']['k_act_min'], k_max=c['pid']['k_act_max']
        )
        self.history_maxlen = c.get('history_maxlen', 10000)

        # ── [Ш2.2] Флаг правки, по умолчанию выключен ──────────────────────
        # Дефект R6: PID берёт МГНОВЕННОЕ значение λ_L в момент t = 50k, а не
        # среднее за интервал. Это ломает разделение временных масштабов, на
        # котором построена вся схема: медленный контур обязан получать
        # усреднённое быстрое, а не одну случайную выборку из него. При
        # λ_L ∈ [−2.4, +0.25] и шуме порядка самой величины один отсчёт раз в
        # 50 шагов — подбрасывание монеты.
        fx = cfg.get('fix', {}) or {}
        self.pid_integrate = bool(fx.get('pid_integrate', False))
        self.pid_interval = int(fx.get('pid_interval', 50)) if self.pid_integrate else 50
        self._lam_accum = []

        self.current_lambda = 0.0
        self.current_k_act = 6
        self.current_beta_kl = 0.05
        self.step = 0

        # История для логов
        self.history = {
            'lambda_L': deque(maxlen=self.history_maxlen),
            'k_act': deque(maxlen=self.history_maxlen),
            'beta_kl': deque(maxlen=self.history_maxlen),
            'u': deque(maxlen=self.history_maxlen),
        }

    def update(self, e_total: float) -> dict:
        """
        Обновить состояние хаоса на основе нового E_total.
        Возвращает dict с текущими управляющими параметрами.
        """
        lL = self.lyapunov.update(e_total)
        self.current_lambda = lL
        self._lam_accum.append(lL)

        # PID раз в pid_interval шагов (не перегружать шумом)
        u = 0.0
        if self.step % self.pid_interval == 0:
            if self.pid_integrate:
                # [Ш2.2] среднее за интервал: медленный контур интегрирует
                # быстрый, а не сэмплирует его. Это и есть требование
                # разделения масштабов (η_PID << η_generation).
                lam_in = float(np.mean(self._lam_accum)) if self._lam_accum else lL
            else:
                lam_in = lL          # прежнее поведение: мгновенный отсчёт
            k_act, beta_kl, u = self.pid.update(lam_in)
            self.current_k_act = k_act
            self.current_beta_kl = beta_kl
            self._lam_accum = []

        self.step += 1

        # Логирование
        self.history['lambda_L'].append(lL)
        self.history['k_act'].append(self.current_k_act)
        self.history['beta_kl'].append(self.current_beta_kl)
        self.history['u'].append(u)

        return {
            'lambda_L': lL,
            'k_act': self.current_k_act,
            'beta_kl': self.current_beta_kl,
            'u': u,
            'regime': self._regime(lL)
        }

    def _regime(self, lL: float) -> str:
        if lL > 0.1:
            return 'CHAOS'
        elif lL < -0.1:
            return 'ORDER'
        else:
            return 'EDGE'
