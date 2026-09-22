"""
core/block_graph.py — Структурный граф блока G_B
Σ_v8.9  (ТЗ §1-3)

G_B = (V, E, W) — ориентированный взвешенный граф.
  V — узлы (AST-сущности или предложения)
  E — рёбра (зависимости)
  W: E → R_{>0} — веса

Основные правила:
  code  → AST-граф (ast.parse → узлы + рёбра иерархии + call→def)
  text  → параграфно-предложенный граф (sequential + Jaccard)
  AST-сбой → fallback на text-граф
  граф пустой → fallback на аварийный single-node

Матрица переходов P:
  P_ij = A_ij / Σ_k A_ik
  Dangling node → uniform row (1/n), НЕ нули.
"""

from __future__ import annotations

import ast as ast_mod
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# Типы graph_type
GRAPH_CODE = "code_ast"
GRAPH_TEXT = "text_paragraph"
GRAPH_SINGLE = "single_node_fallback"


@dataclass
class GraphStats:
    """Структурные характеристики G_B (ТЗ §7)."""
    graph_type: str = GRAPH_SINGLE
    number_of_nodes: int = 0
    number_of_edges: int = 0
    density: float = 0.0
    reachability_ratio: float = 0.0
    component_count: int = 1
    branch_factor: float = 0.0
    fallback_used: bool = False

    @property
    def structure_score(self) -> float:
        """
        Scalar score по полям структуры.
        score = 0.4·density + 0.3·reachability_ratio + 0.2·(1-1/max(branch_factor,1)) + 0.1·log1p(edges)/log1p(100)
        Диапазон: [0, 1].
        """
        import math
        s = (0.4 * min(self.density, 1.0)
             + 0.3 * self.reachability_ratio
             + 0.2 * (1.0 - 1.0 / max(self.branch_factor, 1.0))
             + 0.1 * math.log1p(self.number_of_edges) / math.log1p(100))
        return float(np.clip(s, 0.0, 1.0))


class BlockGraph:
    """
    Структурный граф одного блока G_B = (V, E, W).

    Открытые атрибуты:
      n          — число узлов
      node_labels — метки узлов (str)
      adj        — adjacency matrix (n×n, float64)
      stats      — GraphStats

    Внутренний метод build() — единственная точка входа.
    """

    def __init__(self):
        self.n: int = 0
        self.node_labels: List[str] = []
        self.adj: Optional[np.ndarray] = None
        self.stats: GraphStats = GraphStats()

    # ─── Точка входа ────────────────────────────────────────────────────────

    def build(self, text: str, task_type: str) -> "BlockGraph":
        """
        Построить G_B из текста.
        Cascading fallback:
          1. code → AST-граф
          2. → text-граф
          3. → single-node
        """
        if task_type == "code" or _looks_like_code(text):
            code = _extract_python(text)
            if self._try_build_code(code):
                return self
            log.debug("BlockGraph: AST failed → text fallback")
            self.stats.fallback_used = True

        if self._build_text(text):
            return self

        log.debug("BlockGraph: text failed → single-node fallback")
        self._build_single(text)
        return self

    # ─── AST-граф ────────────────────────────────────────────────────────────

    # Узлы, которые включаем в граф
    _AST_INTERESTING = (
        ast_mod.FunctionDef, ast_mod.AsyncFunctionDef, ast_mod.ClassDef,
        ast_mod.Return, ast_mod.Call,
        ast_mod.If, ast_mod.For, ast_mod.While,
        ast_mod.Try, ast_mod.With,
        ast_mod.Import, ast_mod.ImportFrom,
        ast_mod.Assign, ast_mod.AnnAssign,
        ast_mod.Name,
    )

    def _try_build_code(self, code: str) -> bool:
        """Попытка построить AST-граф. False при любой ошибке."""
        try:
            tree = ast_mod.parse(code)
        except SyntaxError as exc:
            log.debug("AST parse failed: %s", exc)
            return False

        nodes_raw: List[Tuple[object, str]] = []
        id_map: Dict[int, int] = {}

        for node in ast_mod.walk(tree):
            if isinstance(node, self._AST_INTERESTING):
                idx = len(nodes_raw)
                id_map[id(node)] = idx
                nodes_raw.append((node, _ast_label(node)))

        if len(nodes_raw) < 2:
            log.debug("AST graph too small (%d nodes)", len(nodes_raw))
            return False

        self.n = len(nodes_raw)
        self.node_labels = [lbl for _, lbl in nodes_raw]
        A = np.zeros((self.n, self.n), dtype=np.float64)

        # ── Рёбра 1: parent → child (AST-иерархия) ──────────────────────────
        for ast_node, _ in nodes_raw:
            pi = id_map.get(id(ast_node))
            if pi is None:
                continue
            for child in ast_mod.iter_child_nodes(ast_node):
                ci = id_map.get(id(child))
                if ci is not None and ci != pi:
                    A[pi, ci] += 1.0     # forward (strong)
                    A[ci, pi] += 0.3     # backward (weak feedback)

        # ── Рёбра 2: call → def (definition → usage) ────────────────────────
        defs: Dict[str, int] = {}
        for i, lbl in enumerate(self.node_labels):
            if lbl.startswith("def:") or lbl.startswith("class:"):
                name = lbl.split(":", 1)[1]
                defs[name] = i

        for i, lbl in enumerate(self.node_labels):
            if lbl.startswith("call:"):
                fname = lbl[5:]
                j = defs.get(fname)
                if j is not None and j != i:
                    A[i, j] += 2.0   # strong call-dependency
                    A[j, i] += 0.5   # weak reverse

        # ── Рёбра 3: Name usage → enclosing definition ───────────────────────
        for i, lbl in enumerate(self.node_labels):
            if lbl.startswith("name:"):
                varname = lbl[5:]
                # find assign node with this name
                for j, lbl2 in enumerate(self.node_labels):
                    if j != i and (lbl2.startswith(f"assign:{varname}")
                                   or lbl2.startswith(f"ann:{varname}")):
                        A[i, j] += 1.0

        self.adj = A
        self.stats = _compute_stats(GRAPH_CODE, A, self.node_labels,
                                    self.stats.fallback_used)
        log.debug(
            "BlockGraph [%s]: n=%d edges=%d density=%.3f",
            GRAPH_CODE, self.n, self.stats.number_of_edges, self.stats.density,
        )
        return True

    # ─── Текстовый граф ──────────────────────────────────────────────────────

    def _build_text(self, text: str) -> bool:
        """Построить граф предложений/абзацев. False если текст пуст."""
        # Сначала попробуем разбить по абзацам
        paras = [p.strip() for p in re.split(r"\n{2,}", text) if len(p.strip()) >= 15]
        if len(paras) < 2:
            # Fallback: по предложениям
            paras = re.split(r"(?<=[.!?])\s+", text.strip())
            paras = [p.strip() for p in paras if len(p.strip()) >= 10]

        if len(paras) < 2:
            return False

        self.n = len(paras)
        self.node_labels = paras
        A = np.zeros((self.n, self.n), dtype=np.float64)

        # ── Рёбра 1: последовательные ───────────────────────────────────────
        for i in range(self.n - 1):
            A[i, i + 1] += 1.0    # forward
            A[i + 1, i] += 0.4    # backward

        # ── Рёбра 2: лексическое перекрытие ─────────────────────────────────
        kws = [_keywords(p) for p in paras]
        for i in range(self.n):
            for j in range(i + 2, min(i + 8, self.n)):
                if kws[i] and kws[j]:
                    union = len(kws[i] | kws[j])
                    if union > 0:
                        jac = len(kws[i] & kws[j]) / union
                        if jac > 0.07:
                            A[i, j] += jac
                            A[j, i] += jac * 0.4

        self.adj = A
        gt = GRAPH_TEXT if not self.stats.fallback_used else GRAPH_TEXT
        self.stats = _compute_stats(gt, A, self.node_labels,
                                    self.stats.fallback_used)
        log.debug(
            "BlockGraph [%s]: n=%d edges=%d density=%.3f",
            gt, self.n, self.stats.number_of_edges, self.stats.density,
        )
        return True

    # ─── Single-node fallback ────────────────────────────────────────────────

    def _build_single(self, text: str):
        self.n = 1
        self.node_labels = [text[:200] if text.strip() else "empty"]
        self.adj = np.zeros((1, 1), dtype=np.float64)
        self.stats = GraphStats(
            graph_type=GRAPH_SINGLE,
            number_of_nodes=1, number_of_edges=0,
            density=0.0, reachability_ratio=1.0,
            component_count=1, branch_factor=0.0,
            fallback_used=True,
        )

    # ─── Матрица переходов (ТЗ §3) ───────────────────────────────────────────

    def transition_matrix(self) -> np.ndarray:
        """
        P_ij = A_ij / Σ_k A_ik
        Dangling node (нулевая строка) → uniform row 1/n.
        Гарантии: P_ij ≥ 0, Σ_j P_ij = 1 для всех i.
        """
        if self.n == 0 or self.adj is None:
            return np.array([[1.0]])

        P = self.adj.copy()
        for i in range(self.n):
            row_sum = P[i].sum()
            if row_sum > 0.0:
                P[i] /= row_sum
            else:
                # dangling node → uniform (not zeros)
                P[i] = np.ones(self.n) / self.n
        return P

    # ─── Personalization vector (ТЗ §персонализация) ─────────────────────────

    def personalization_vector(self, task: str,
                               task_type: str = "general",
                               tau_v: float = 1.0,
                               embeddings: Optional[np.ndarray] = None
                               ) -> np.ndarray:
        """
        v_i ≥ 0,  Σ v_i = 1.

        Для code: базовый вес по типу узла + лексическое совпадение с task.
        Для text: лексическое совпадение с task (+ embeddings если переданы).
        Если task пустой → uniform fallback.
        """
        if self.n <= 1:
            return np.array([1.0])

        scores = np.zeros(self.n, dtype=np.float64)
        task_kw = _keywords(task) if task else set()

        if task_type == "code":
            # Базовый вес по типу узла (ТЗ: повышенный вес у key-узлов)
            type_weights = {
                "def:": 2.0, "async_def:": 2.0, "class:": 2.0,
                "return": 1.5, "call:": 1.5,
                "if": 1.2, "for": 1.2, "while": 1.2,
                "try": 1.1, "with": 1.1,
                "import:": 1.0, "from:": 1.0,
                "assign:": 0.8, "ann:": 0.8,
                "name:": 0.6,
            }
            for i, lbl in enumerate(self.node_labels):
                for prefix, w in type_weights.items():
                    if lbl.startswith(prefix) or lbl == prefix.rstrip(":"):
                        scores[i] = max(scores[i], w)
                        break
                else:
                    scores[i] = 0.5

            # Дополнительный бонус за лексическое совпадение с task
            if task_kw:
                for i, lbl in enumerate(self.node_labels):
                    lbl_kw = _keywords(lbl)
                    if lbl_kw:
                        u = len(lbl_kw | task_kw)
                        if u > 0:
                            jac = len(lbl_kw & task_kw) / u
                            scores[i] += jac * 1.5
        else:
            # text: лексическое совпадение
            if task_kw:
                for i, lbl in enumerate(self.node_labels):
                    lbl_kw = _keywords(str(lbl))
                    if lbl_kw:
                        u = len(lbl_kw | task_kw)
                        scores[i] = len(lbl_kw & task_kw) / u if u > 0 else 0.0
            # + embedding similarity если предоставлены
            if embeddings is not None and len(embeddings) == self.n:
                task_emb = None  # нет task embedding — пропускаем
                # (embeddings используются только в v, не в entropy)

        # Если нет сигнала → uniform
        if scores.max() < 1e-10:
            return np.ones(self.n) / self.n

        # softmax с температурой tau_v
        scores = scores / scores.max()  # numerical stability
        exp_s = np.exp(scores / tau_v)
        total = exp_s.sum()
        v = exp_s / total if total > 0 else np.ones(self.n) / self.n
        return v.astype(np.float64)


# ─── Вспомогательные функции ──────────────────────────────────────────────────

def _extract_python(text: str) -> str:
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return m.group(1) if m else text


def _looks_like_code(text: str) -> bool:
    code = _extract_python(text)
    if "```" in text and re.search(r"\b(def|class|import|from)\b", code):
        return True
    lines = code.splitlines()
    signals = 0
    signals += len(re.findall(r"^\s*(def|class)\s+\w+", code, re.MULTILINE)) * 3
    signals += len(re.findall(r"^\s*(import|from)\s+\w+", code, re.MULTILINE)) * 2
    signals += len(re.findall(r"^\s{4,}\S+", code, re.MULTILINE))
    signals += code.count("{") + code.count("}")
    signals += code.count("):")
    long_enough = len(lines) >= 4 or len(code) >= 160
    if signals >= 4 and long_enough:
        try:
            ast_mod.parse(code)
            return True
        except SyntaxError:
            return signals >= 8
    return False


def _keywords(text: str) -> set:
    return set(w.lower() for w in re.findall(r"\b[a-zA-Z_]\w{2,}\b", text))


def _ast_label(node) -> str:
    if isinstance(node, (ast_mod.FunctionDef, ast_mod.AsyncFunctionDef)):
        prefix = "def" if isinstance(node, ast_mod.FunctionDef) else "async_def"
        return f"{prefix}:{node.name}"
    if isinstance(node, ast_mod.ClassDef):
        return f"class:{node.name}"
    if isinstance(node, ast_mod.Call):
        if isinstance(node.func, ast_mod.Name):
            return f"call:{node.func.id}"
        if isinstance(node.func, ast_mod.Attribute):
            return f"call:{node.func.attr}"
        return "call:?"
    if isinstance(node, ast_mod.Import):
        return f"import:{','.join(a.name for a in node.names)}"
    if isinstance(node, ast_mod.ImportFrom):
        return f"from:{node.module or '?'}"
    if isinstance(node, ast_mod.Return):
        return "return"
    if isinstance(node, ast_mod.If):    return "if"
    if isinstance(node, ast_mod.For):   return "for"
    if isinstance(node, ast_mod.While): return "while"
    if isinstance(node, ast_mod.Try):   return "try"
    if isinstance(node, ast_mod.With):  return "with"
    if isinstance(node, ast_mod.Assign):
        try:    return f"assign:{','.join(ast_mod.unparse(t) for t in node.targets)}"
        except: return "assign"
    if isinstance(node, ast_mod.AnnAssign):
        try:    return f"ann:{ast_mod.unparse(node.target)}"
        except: return "annassign"
    if isinstance(node, ast_mod.Name):
        return f"name:{node.id}"
    return type(node).__name__.lower()


def _compute_stats(graph_type: str, A: np.ndarray,
                   labels: List[str], fallback: bool) -> GraphStats:
    """Вычислить GraphStats по adjacency matrix."""
    n = A.shape[0]
    nnz = int(np.count_nonzero(A > 0))

    # density = |E| / (|V|·(|V|-1)) для ориентированного графа
    if n > 1:
        density = nnz / (n * (n - 1))
    else:
        density = 0.0

    # branch_factor = средняя out-degree по ненулевым строкам
    out_degrees = (A > 0).sum(axis=1).astype(float)
    nonzero_rows = out_degrees[out_degrees > 0]
    branch_factor = float(nonzero_rows.mean()) if len(nonzero_rows) > 0 else 0.0

    # reachability_ratio через BFS от узлов с высшим типом (def/class/root)
    if n > 0:
        start = 0
        visited = set()
        queue = [start]
        visited.add(start)
        while queue:
            cur = queue.pop(0)
            for nb in range(n):
                if A[cur, nb] > 0 and nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        reachability = len(visited) / n
    else:
        reachability = 0.0

    # component_count (слабосвязные через undirected adjacency)
    sym = (A + A.T > 0).astype(int)
    components = _count_components(sym, n)

    return GraphStats(
        graph_type=graph_type,
        number_of_nodes=n,
        number_of_edges=nnz,
        density=float(density),
        reachability_ratio=float(reachability),
        component_count=components,
        branch_factor=float(branch_factor),
        fallback_used=fallback,
    )


def _count_components(adj_sym: np.ndarray, n: int) -> int:
    """Union-Find для подсчёта слабосвязных компонент."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if adj_sym[i, j]:
                union(i, j)

    return len(set(find(i) for i in range(n)))
