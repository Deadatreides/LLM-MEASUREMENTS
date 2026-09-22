"""
core/ppr.py — Personalized PageRank + Shannon Entropy
Σ_v8.9  (ТЗ §4-6)

Строгая реализация без апроксимаций:
  π_{t+1} = α·Pᵀ·π_t + (1-α)·v
  H(π) = -Σ π_i·log(π_i)

Гарантии:
  π_i ≥ 0,  Σ π_i = 1   (на входе И на выходе)
  H ≥ 0
  H ≤ log(n)
  H = 0 при n ≤ 1
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

EPS = 1e-12          # защита от log(0)
DEFAULT_ALPHA = 0.85
DEFAULT_TOL = 1e-8
ITER_MARGIN = 20     # запас поверх теоретически необходимого числа итераций
ITER_CAP = 10000     # потолок на случай alpha → 1


def required_iterations(alpha: float = DEFAULT_ALPHA,
                        tol: float = DEFAULT_TOL) -> int:
    """Сколько итераций нужно, чтобы степенная итерация дошла до tol.

    Ошибка степенной итерации убывает как alpha^k, значит
        k >= log(tol) / log(alpha).
    При alpha=0.85 и tol=1e-8 это 113.3 — а фиксированный лимит был 100.
    Порог и лимит были рассогласованы МЕЖДУ СОБОЙ, поэтому предупреждение
    «PPR did not converge» выписывалось всегда, на любом графе: 1891 раз за
    прогон sh2_200, все ложные (невязка медианно 2.1e-8 при пороге 1e-8,
    вклад в H — 1.5e-8 против самого малого различимого |ΔH| 8e-4).

    Считается из alpha и tol, а не подставляется константой: константа 150
    работает до alpha ≈ 0.88, и при подъёме alpha до 0.9 рассогласование
    вернулось бы молча. Формула снимает класс ошибки, а не один её случай.
    """
    if not (0.0 < alpha < 1.0) or not (0.0 < tol < 1.0):
        return 100
    need = math.ceil(math.log(tol) / math.log(alpha)) + ITER_MARGIN
    return int(min(max(need, 10), ITER_CAP))


DEFAULT_MAX_ITER = required_iterations()


@dataclass
class PPRResult:
    """Результат PPR-вычисления."""
    pi: np.ndarray          # стационарное распределение, Σ=1, все ≥ 0
    converged: bool         # достигнута ли сходимость
    iterations: int         # число итераций
    residual: float         # финальная ‖π_{t+1} - π_t‖₁

    # Энтропийные метрики
    entropy: float          # H(π) = -Σ π_i·log(π_i)
    entropy_norm: float     # H_norm = H / log(n), 0 если n ≤ 1


def run_ppr(
    P: np.ndarray,
    v: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
    tol: float = DEFAULT_TOL,
    max_iter: Optional[int] = None,
) -> PPRResult:
    """
    Personalized PageRank — итерационное решение (ТЗ §4).

    Аргументы:
      P       — матрица переходов (n×n), row-stochastic (Σ_j P_ij = 1)
      v       — personalization vector (n,), v_i ≥ 0, Σ v_i = 1
      alpha   — damping factor ∈ (0,1)
      tol     — порог сходимости по L1-норме
      max_iter — макс. итераций; None → считается из alpha и tol
                 (см. required_iterations)

    Возвращает PPRResult с π, H, сходимостью.
    """
    if max_iter is None:
        max_iter = required_iterations(alpha, tol)
    n = P.shape[0]

    # ── Защита входных данных ────────────────────────────────────────────────
    if n == 0:
        return PPRResult(
            pi=np.array([]), converged=True,
            iterations=0, residual=0.0,
            entropy=0.0, entropy_norm=0.0,
        )

    if n == 1:
        return PPRResult(
            pi=np.array([1.0]), converged=True,
            iterations=0, residual=0.0,
            entropy=0.0, entropy_norm=0.0,
        )

    # Нормализовать v на случай численных ошибок
    v = np.asarray(v, dtype=np.float64)
    v = np.maximum(v, 0.0)
    v_sum = v.sum()
    if v_sum < EPS:
        v = np.ones(n) / n
    else:
        v /= v_sum

    # Инициализация π = v
    pi = v.copy()

    # Pᵀ вычислять один раз
    PT = P.T  # (n×n), используем @

    converged = False
    residual = float("inf")

    for k in range(max_iter):
        pi_new = alpha * (PT @ pi) + (1.0 - alpha) * v

        # Проецируем на симплекс (численная стабильность)
        pi_new = np.maximum(pi_new, 0.0)
        s = pi_new.sum()
        pi_new /= s if s > EPS else 1.0

        residual = float(np.abs(pi_new - pi).sum())
        pi = pi_new

        if residual < tol:
            converged = True
            break

    if not converged:
        log.warning(
            "PPR did not converge in %d iterations (residual=%.2e, tol=%.2e)",
            max_iter, residual, tol,
        )

    # Финальная нормализация
    pi = np.maximum(pi, 0.0)
    pi /= pi.sum() if pi.sum() > EPS else 1.0

    # ── Shannon entropy H(π) ─────────────────────────────────────────────────
    H, H_norm = shannon_entropy(pi)

    return PPRResult(
        pi=pi,
        converged=converged,
        iterations=k + 1,
        residual=residual,
        entropy=H,
        entropy_norm=H_norm,
    )


def shannon_entropy(pi: np.ndarray, eps: float = EPS) -> tuple[float, float]:
    """
    H(π) = -Σ π_i · log(π_i)   (натуральный логарифм)
    Численная защита: π_i ← max(π_i, ε) перед log.

    Возвращает (H, H_norm):
      H_norm = H / log(n), если n > 1, иначе 0.
    """
    n = len(pi)
    if n <= 1:
        return 0.0, 0.0

    safe = np.maximum(pi, eps)
    H = float(-np.sum(safe * np.log(safe)))
    H = max(H, 0.0)          # гарантируем H ≥ 0

    H_max = math.log(n)
    H_norm = H / H_max if H_max > 0 else 0.0

    return H, H_norm
