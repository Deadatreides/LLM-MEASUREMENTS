"""
core/blackboard.py — Semantic Blackboard Σ_v8.9
Основной путь: G_B → PPR → Shannon entropy H(π)

АРХИТЕКТУРА:
  G_B (core/block_graph.py) — структурный граф блока
  PPR (core/ppr.py)         — Personalized PageRank → π
  H   = -Σ π_i·log(π_i)    — Shannon entropy (основной путь)
  Fallback: embedding-based entropy (только аварийный)

G_B != G_M:
  G_B — локальный граф текущего блока (AST / предложения)
  G_M — глобальный Physarum-граф памяти (core/memory.py)
  Они не пересекаются.

Внешний API (не менять — используется orchestrator.py):
  blackboard.upsert(block_id, text, task='', task_type='general')
  blackboard.select(blocks=None, lambda_L=0.0)
  blackboard.compute_thermodynamics(block_id, final_answer, task,
                                     task_type, q_env, tokens_used)
  selected_block.entropy  → H(π) через PPR
  selected_block.add_version(candidate, task=task, task_type=task_type)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from core.block_graph import BlockGraph, GraphStats
from core.ppr import PPRResult, run_ppr, shannon_entropy

log = logging.getLogger(__name__)


# ─── ThermodynamicState ────────────────────────────────────────────────────────

@dataclass
class ThermodynamicState:
    """
    Термодинамическое состояние за один шаг (Пригожин).
    dS/dt = d_eS + d_iS;  Φ = |d_eS| - d_iS > 0 = система жива.
    """
    H_old:   float = 0.0
    H_new:   float = 0.0
    delta_H: float = 0.0
    pi_r:    float = 0.0
    rho:     float = 1.0
    d_eS:    float = 0.0
    d_iS:    float = 0.0
    Phi:     float = 0.0
    c_r:     float = 1.0
    L_r:     float = 1.0
    alive:   bool  = True


# ─── SemanticBlock ─────────────────────────────────────────────────────────────

class SemanticBlock:
    """
    Один семантический блок с историей версий и PPR-энтропией.

    Открытые поля (ТЗ «у блока должны появиться поля»):
      graph_type, graph_nodes, graph_edges, ppr, entropy,
      structure_score, fallback_used
    """

    def __init__(self, block_id: str, max_versions: int = 5,
                 ppr_alpha: float = 0.85, entropy_norm: bool = False):
        self.id           = block_id
        self.max_versions = max_versions
        self.ppr_alpha    = ppr_alpha
        # [Ш2.1] Хранить H в [0,1] вместо натов.
        #
        # ppr.run_ppr возвращает и entropy (H = -Σπ·logπ, диапазон [0, log n]),
        # и entropy_norm (H/log n ∈ [0,1]). Код везде брал первое, а пороги
        # писались под второе:
        #   depth.entropy_threshold = 0.08 против наблюдавшихся 0.89..5.26 нат
        #     → гейт не существовал, депт-луп запускался 202/202 раза;
        #   depth.epsilon = 0.01 против |ΔH| в натах → критерий остановки не в
        #     том масштабе.
        # Все пять матописаний независимо утверждают, что в термодинамику идёт
        # H_norm. Переключаем сам источник, чтобы потребители не правились.
        self.use_entropy_norm = bool(entropy_norm)
        self.versions: List[str] = []
        self.rho: float = 1.0

        # ── Поля, требуемые ТЗ ──────────────────────────────────────────────
        self.graph_type: str   = "none"
        self.graph_nodes: int  = 0
        self.graph_edges: int  = 0
        self.ppr: Optional[np.ndarray] = None   # π-распределение
        self.entropy: float    = 0.0             # H(π) — основной
        self.structure_score: float = 0.0
        self.fallback_used: bool = False

        # Внутреннее: граф и PPR-результат последней версии
        self._graph: Optional[BlockGraph]  = None
        self._ppr_result: Optional[PPRResult] = None
        self._last_task: str = ""
        self._last_type: str = "general"

    def add_version(self, text: str, task: str = "",
                    task_type: str = "general") -> float:
        """
        Добавить новую версию текста.
        Пересчитать G_B, PPR, H(π).
        Возвращает H_new.
        """
        if not text:
            return self.entropy

        self.versions.append(text)
        if len(self.versions) > self.max_versions:
            self.versions = self.versions[-self.max_versions:]

        self._last_task = task
        self._last_type = task_type
        self._recompute(text, task, task_type)
        return self.entropy

    def _recompute(self, text: str, task: str, task_type: str):
        """Построить G_B, запустить PPR, вычислить H."""
        # ── Шаг 1: построить граф ─────────────────────────────────────────
        g = BlockGraph().build(text, task_type)
        self._graph = g
        stats: GraphStats = g.stats

        # ── Шаг 2: если граф вырожден (n<=1) → fallback ───────────────────
        if g.n <= 1:
            self._apply_fallback(text, stats)
            return

        # ── Шаг 3: матрица переходов P ────────────────────────────────────
        P = g.transition_matrix()

        # ── Шаг 4: personalization vector v ──────────────────────────────
        v = g.personalization_vector(task, task_type)

        # ── Шаг 5: PPR → π ───────────────────────────────────────────────
        result = run_ppr(P, v, alpha=self.ppr_alpha)
        self._ppr_result = result

        # ── Шаг 6: H(π) — ОСНОВНОЙ путь ─────────────────────────────────
        self.entropy       = (result.entropy_norm if self.use_entropy_norm
                              else result.entropy)
        self.ppr           = result.pi
        self.graph_type    = stats.graph_type
        self.graph_nodes   = stats.number_of_nodes
        self.graph_edges   = stats.number_of_edges
        self.structure_score = stats.structure_score
        self.fallback_used = stats.fallback_used

        log.info(
            "  Block [%s] type=%s n=%d edges=%d fallback=%s "
            "PPR_conv=%s iters=%d H=%.4f H_norm=%.4f score=%.3f",
            self.id, self.graph_type, self.graph_nodes, self.graph_edges,
            self.fallback_used, result.converged, result.iterations,
            self.entropy, result.entropy_norm, self.structure_score,
        )

    def _apply_fallback(self, text: str, stats: GraphStats):
        """Аварийный fallback: embedding-based entropy (только при n<=1)."""
        self.graph_type    = stats.graph_type
        self.graph_nodes   = stats.number_of_nodes
        self.graph_edges   = stats.number_of_edges
        self.structure_score = stats.structure_score
        self.fallback_used = True
        self.ppr = None
        # Используем старую cosine-proxy энтропию из embedding fallback
        # Но без embeddings используем текстовую эвристику
        self.entropy = _text_entropy_heuristic(
            text, normalize=self.use_entropy_norm)
        self._ppr_result = None
        log.debug(
            "  Block [%s] FALLBACK (n=%d): H_heuristic=%.4f",
            self.id, stats.number_of_nodes, self.entropy,
        )

    def complexity(self) -> float:
        """L_r = graph_edges для термодинамики d_iS."""
        return float(max(1.0, self.graph_edges))

    def entropy_norm(self) -> float:
        """Нормированная энтропия H/log(n)."""
        if self._ppr_result is not None:
            return self._ppr_result.entropy_norm
        return 0.0


# ─── SemanticBlackboard ────────────────────────────────────────────────────────

class SemanticBlackboard:
    """
    Семантическая доска с PPR-энтропией как основным механизмом.

    Публичный API (совместим с orchestrator.py):
      upsert(block_id, text, task='', task_type='general')
      select(blocks=None, lambda_L=0.0)
      compute_thermodynamics(...)
      get_entropy(block_id) → float
      max_versions, ppr_alpha
    """

    def __init__(self, cfg: dict):
        bb = cfg.get("blackboard", {})
        self.max_versions:   int   = bb.get("max_versions", 5)
        self.temp_min:       float = bb.get("temperature_min", 0.25)
        self.temp_max:       float = bb.get("temperature_max", 1.50)
        self.lyapunov_scale: float = bb.get("lyapunov_scale", 3.0)
        self.ppr_alpha:      float = bb.get("ppr_alpha", 0.85)
        # [Ш2.1] см. SemanticBlock.use_entropy_norm
        self.use_entropy_norm: bool = bool(
            (cfg.get("fix", {}) or {}).get("entropy_norm", False))
        self.replicator_D:   float = bb.get("replicator_diffusion", 0.05)

        self.blocks: Dict[str, SemanticBlock] = {}
        self._pi_bar: float = 0.0
        self._topology_bias: Dict[str, float] = {}
        # [Ш5] Блок создаётся на каждый шаг (block_id = f"step-{step}") и не
        # вытеснялся никогда: каждый держит до max_versions текстов, граф и
        # вектор PPR. Плюс rho_bar = 1/len(blocks) — по мере накопления
        # блоков rho всех сползает к нижнему клипу 0.01.
        self.max_blocks: int = int(bb.get("max_blocks", 500))

    def set_topology_bias(self, bias: Dict[str, float]):
        self._topology_bias = {
            str(k): float(np.clip(v, 0.0, 1.0))
            for k, v in (bias or {}).items()
            if np.isfinite(float(v))
        }

    # ── upsert ───────────────────────────────────────────────────────────────

    def upsert(self, block_id: str, text: str,
               task: str = "", task_type: str = "general") -> SemanticBlock:
        """
        Добавить/обновить блок. Вычислить G_B → PPR → H.
        Возвращает обновлённый SemanticBlock.
        """
        block = self.blocks.get(block_id)
        if block is None:
            block = SemanticBlock(
                block_id,
                max_versions=self.max_versions,
                ppr_alpha=self.ppr_alpha,
                entropy_norm=self.use_entropy_norm,
            )
            self.blocks[block_id] = block
            self._evict_if_needed()
        block.add_version(text, task=task, task_type=task_type)
        return block

    def _evict_if_needed(self):
        """[Ш5] FIFO-вытеснение самых старых блоков сверх max_blocks."""
        excess = len(self.blocks) - self.max_blocks
        if excess <= 0:
            return
        for key in list(self.blocks.keys())[:excess]:
            self.blocks.pop(key, None)

    def get_entropy(self, block_id: str) -> float:
        b = self.blocks.get(block_id)
        return b.entropy if b else 0.0

    def effective_entropy(self, block_id: str, entropy: Optional[float] = None) -> float:
        base = self.get_entropy(block_id) if entropy is None else float(entropy)
        bias = self._topology_bias.get(str(block_id), None)
        if bias is None and self._topology_bias:
            bias = float(np.mean(list(self._topology_bias.values())))
        bias = float(np.clip(bias or 0.0, 0.0, 1.0))
        return float(np.clip(base, 0.0, 20.0) * (1.0 + 0.1 * bias))

    # ── compute_thermodynamics ───────────────────────────────────────────────

    def compute_thermodynamics(
        self,
        block_id: str,
        final_answer: str,
        task: str,
        task_type: str,
        q_env: float,
        tokens_used: int,
        h_old: Optional[float] = None,
    ) -> ThermodynamicState:
        """
        Строгий каузальный порядок:
          1. H_old — ДО обновления
          2. upsert → G_B → PPR → H_new
          3. ΔH = H_old - H_new
          4. π_r, d_eS, d_iS, Φ
          5. обновить ρ (репликатор)

        h_old: [lab, обратимо] явный снимок H_old. По умолчанию None —
        поведение ровно прежнее. Нужен потому, что depth loop успевает
        положить final_answer в тот же блок ДО этого вызова, и тогда
        H_old ≡ H_new ⇒ ΔH ≡ 0 ⇒ Φ ≡ 0. См. depth.thermo_causal_fix.
        """
        H_old = self.effective_entropy(block_id) if h_old is None else float(h_old)

        block = self.upsert(block_id, final_answer, task, task_type)
        H_new = self.effective_entropy(block_id, block.entropy)
        L_r   = block.complexity()
        rho   = block.rho

        delta_H = H_old - H_new
        c_r     = float(max(1.0, tokens_used))
        pi_r    = (max(0.0, delta_H) / c_r) * float(q_env)

        d_eS = -float(rho) * float(delta_H) * float(q_env)
        self._pi_bar = 0.9 * self._pi_bar + 0.1 * pi_r
        d_iS = float(max(0.0, L_r * (pi_r - self._pi_bar) ** 2))
        Phi  = abs(d_eS) - d_iS

        rho_bar = 1.0 / max(1, len(self.blocks))
        block.rho = float(np.clip(
            rho + rho * (pi_r - self._pi_bar) + self.replicator_D * (rho_bar - rho),
            0.01, 10.0,
        ))

        state = ThermodynamicState(
            H_old=H_old, H_new=H_new, delta_H=delta_H,
            pi_r=pi_r, rho=rho,
            d_eS=d_eS, d_iS=d_iS, Phi=Phi,
            c_r=c_r, L_r=L_r,
            alive=(abs(d_eS) > d_iS),
        )
        log.info(
            "  Thermo %s: H %.4f→%.4f ΔH=%.4f π_r=%.5f "
            "d_eS=%.5f d_iS=%.5f Φ=%.5f alive=%s",
            block_id, H_old, H_new, delta_H, pi_r,
            d_eS, d_iS, Phi, state.alive,
        )
        return state

    def system_entropy(self) -> float:
        """S = -Σ ρ·log(ρ) + Σ ρ·H(r)"""
        if not self.blocks:
            return 0.0
        rhos = np.array([b.rho for b in self.blocks.values()])
        Hs   = np.array([b.entropy for b in self.blocks.values()])
        rhos = np.maximum(rhos / rhos.sum(), 1e-300)
        return float(-np.sum(rhos * np.log(rhos)) + np.sum(rhos * Hs))

    # ── temperature-softmax выбор ─────────────────────────────────────────────

    def temperature(self, lambda_L: float) -> float:
        p = float(np.tanh(self.lyapunov_scale * abs(lambda_L)))
        return self.temp_min + (self.temp_max - self.temp_min) * p

    def probabilities(self, blocks: List[SemanticBlock],
                      lambda_L: float) -> np.ndarray:
        if not blocks:
            return np.array([], dtype=np.float64)
        temp = max(self.temperature(lambda_L), self.temp_min)
        scores = np.array([b.entropy for b in blocks], dtype=np.float64)
        topology = np.array([
            self._topology_bias.get(str(b.id), 0.0)
            for b in blocks
        ], dtype=np.float64)
        scores = np.clip(scores, 0.0, 20.0) * (1.0 + 0.1 * np.clip(topology, 0.0, 1.0))
        s_max = scores.max()
        if s_max > 0:
            scores = scores / s_max
        return _softmax(scores / temp)

    def select(self, blocks=None, lambda_L: float = 0.0):
        candidates = blocks or list(self.blocks.values())
        if not candidates:
            return None
        probs = self.probabilities(candidates, lambda_L)
        return candidates[int(np.random.choice(len(candidates), p=probs))]


# ─── Вспомогательные ──────────────────────────────────────────────────────────

def _softmax(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    e = np.exp(values - values.max())
    s = e.sum()
    return e / s if s > 1e-12 else np.ones_like(values) / values.size


def _text_entropy_heuristic(text: str, normalize: bool = False) -> float:
    """
    Аварийный fallback (только при n<=1).
    Грубая оценка по длине и разнообразию слов.
    НЕ является основным путём.

    normalize — делить на log(число уникальных слов), чтобы аварийный путь
    жил в той же шкале, что основной. Иначе при включённом Ш2.1 fallback
    выдавал бы наты там, где всё остальное в [0,1], и гейты снова поехали бы.
    """
    if not text.strip():
        return 0.0
    import re, math
    words = re.findall(r"\w+", text.lower())
    if len(words) < 2:
        return 0.0
    unique = len(set(words))
    total  = len(words)
    # Crude: normalized word entropy
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    probs = np.array([v / total for v in freq.values()])
    probs = np.maximum(probs, 1e-12)
    h = float(-np.sum(probs * np.log(probs)))
    if normalize:
        import math
        hmax = math.log(len(freq)) if len(freq) > 1 else 0.0
        return (h / hmax) if hmax > 0 else 0.0
    return h


# ─── Backward-compat (тесты v8.8.1) ──────────────────────────────────────────

def entropy_from_embeddings(embeddings) -> float:
    """DEPRECATED: cosine-proxy. Сохранена для обратной совместимости тестов."""
    n = len(embeddings)
    if n < 2:
        return 0.0
    alpha = 4.0
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            a = np.asarray(embeddings[i], dtype=float)
            b = np.asarray(embeddings[j], dtype=float)
            d = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
            sims.append(float(np.dot(a, b) / d))
    sa = np.array(sims)
    e  = np.exp(alpha * (sa - sa.max()))
    p  = e / e.sum()
    h  = -float(np.sum(p * np.log(p + 1e-9)))
    hm = np.log(len(sims)) if len(sims) > 1 else 1.0
    return float(h / hm) if hm > 0 else 0.0
