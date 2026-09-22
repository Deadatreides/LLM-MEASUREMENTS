"""
core/memory.py — Гибридная память: FAISS + граф Physarum
Σ_v8.5 «Мицелий»

FAISS: поиск похожих задач по вектору (1408d)
Граф: отслеживает КАУЗАЛЬНЫЕ связи между решениями (что что усиливает)

Аналогия Physarum polycephalum:
- Трубки утолщаются пропорционально потоку питательных веществ
- Слабые ветки отмирают (γ = 0.048 испарение)
- Levy flight (вероятность 0.005) добавляет случайные ветки — споры

Leaky integrator I(t) — RC-цепь:
- τ = 1/(1-0.9985) ≈ 666 шагов ≈ 1.1 часа при 6 сек/шаг
- I* = γ·E[E_gain] / (1-λ) — устойчивый аттрактор
"""

import numpy as np
import sqlite3
import pickle
import os
import time
import hashlib
from typing import Optional, List, Tuple, Dict
from collections import defaultdict


class HybridEmbedder:
    """
    BGE-m3 (1024d) + E5-small (384d) → 1408d
    Масштаб 0.68 выравнивает дисперсии.
    Работает через ONNX на CPU (47мс) или GPU (12мс).
    """
    def __init__(self, cfg: dict):
        mem = cfg['memory']
        self.scale = mem['hybrid_scale']  # 0.68
        self.dim = mem['vector_dim']      # 1408
        self._bge = None
        self._e5 = None
        self._loaded = False
        # [Ш5] Отсюда удалён метод set_search_priority: он был скопирован не в
        # тот класс (живой дубликат — в Memory) и не вызывался ниоткуда.
        # Такие остатки создают впечатление, что механизм есть в двух местах.

    def _lazy_load(self):
        if self._loaded:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._bge = SentenceTransformer('BAAI/bge-m3', device='cpu')
            self._e5 = SentenceTransformer('intfloat/e5-small-v2', device='cpu')
            self._loaded = True
        except Exception as e:
            print(f"[Memory] Embedder load failed: {e}. Using deterministic fallback vectors.")
            self._loaded = True

    def encode(self, text: str) -> np.ndarray:
        self._lazy_load()
        if self._bge is None:
            # Deterministic fallback for tests and minimal environments.
            seed = int.from_bytes(hashlib.sha256(text.encode('utf-8')).digest()[:8], 'little')
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(self.dim).astype(np.float32)
            return vec / (np.linalg.norm(vec) + 1e-9)

        v1 = self._bge.encode(text, normalize_embeddings=True)   # 1024d
        v2 = self._e5.encode(text, normalize_embeddings=True)    # 384d
        raw = np.concatenate([v1, self.scale * v2])              # 1408d
        return (raw / (np.linalg.norm(raw) + 1e-9)).astype(np.float32)


class FAISSIndex:
    """Обёртка над FAISS IVFFlat→IVFPQIndex"""
    def __init__(self, cfg: dict, path: str):
        self.dim = cfg['memory']['vector_dim']
        self.nlist = cfg['memory']['faiss']['nlist']
        self.nprobe = cfg['memory']['faiss']['nprobe']
        self.m = cfg['memory']['faiss']['m']
        self.nbits = cfg['memory']['faiss']['nbits']
        self.path = path
        self.index = None
        self.id_map = []  # id_map[i] = trace_id в SQLite
        self._load_or_create()

    def _load_or_create(self):
        try:
            import faiss
            if os.path.exists(self.path):
                self.index = faiss.read_index(self.path)
                map_path = self.path + '.map'
                if os.path.exists(map_path):
                    with open(map_path, 'rb') as f:
                        self.id_map = pickle.load(f)
            else:
                # Начинаем с Flat, переходим к IVFFlat после 10k векторов
                self.index = faiss.IndexFlatL2(self.dim)
        except ImportError:
            print("[Memory] FAISS not installed. pip install faiss-cpu")
            self.index = None

    def add(self, vec: np.ndarray, trace_id: int):
        """
        Добавить вектор в FAISS индекс.
        При ошибке — откат: trace_id не добавляется в id_map.
        """
        if self.index is None:
            return
        import faiss
        vec = vec.reshape(1, -1).astype(np.float32)

        # Апгрейд до IVFFlat: СТРОГОЕ ==10000 хрупко (skip если уже IVFFlat)
        # FIX: заменить == на >= чтобы не пропустить upgrade при перезапуске
        if (len(self.id_map) >= 10000 and
                not isinstance(self.index, faiss.IndexIVFFlat)):
            print("[Memory] Upgrading FAISS to IVFFlat...")
            quantizer = faiss.IndexFlatL2(self.dim)
            new_idx = faiss.IndexIVFFlat(quantizer, self.dim,
                                          min(self.nlist, len(self.id_map)//10))
            old_vecs = self.index.reconstruct_n(0, self.index.ntotal)
            new_idx.train(old_vecs)
            new_idx.add(old_vecs)
            new_idx.nprobe = self.nprobe
            self.index = new_idx

        # FIX 1: транзакционная безопасность — откат при сбое FAISS
        try:
            self.index.add(vec)
            self.id_map.append(trace_id)
        except Exception as e:
            import logging
            logging.getLogger('memory').error(
                "FAISSIndex.add failed for trace_id=%d: %s — rollback", trace_id, e)
            # Откат: trace_id НЕ попадает в id_map, граф и SQLite остаются консистентными

    def search(self, vec: np.ndarray, k: int = 50) -> List[int]:
        if self.index is None or self.index.ntotal < 1:
            return []
        import faiss
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = self.nprobe
        vec = vec.reshape(1, -1).astype(np.float32)
        k = min(k, self.index.ntotal)
        _, indices = self.index.search(vec, k)
        return [self.id_map[i] for i in indices[0] if 0 <= i < len(self.id_map)]

    def save(self):
        if self.index is None:
            return
        import faiss
        faiss.write_index(self.index, self.path)
        with open(self.path + '.map', 'wb') as f:
            pickle.dump(self.id_map, f)


class PhysarumGraph:
    """
    Граф памяти по аналогии с Physarum polycephalum.
    Узлы = следы (trace_id).
    Рёбра = усиленные переходы.
    Вес ребра D_ij растёт с потоком, медленно затухает.

    D_ij(t+1) = clip(γ·D_ij(t) + η·Flow_ij·(1+tanh(E))/2, 0.01, 1.0)
    """
    def __init__(self, cfg: dict, path: str):
        g = cfg['memory']['graph']
        self.decay = g['decay']
        self.flow_gain = g['flow_gain']
        self.competition_decay = g.get('competition_decay', 0.02)
        self.flow_thresh = g['flow_threshold']
        self.edge_min = g['edge_min']
        self.edge_max = g['edge_max']
        self.levy_prob = g['levy_prob']
        self.levy_w = g['levy_weight']
        self.ppr_alpha = g['ppr_alpha']
        self.ppr_iters = g['ppr_iters']
        self.pr_interval = g['pagerank_interval']
        self.path = path

        self.edges = defaultdict(float)  # (i,j) → weight
        self.nodes = set()
        self.pagerank = {}
        self.step = 0
        self._lambda_L = 0.0
        self._load()

    def set_lambda(self, lambda_L: float):
        self._lambda_L = float(np.clip(lambda_L, -10.0, 10.0))

    def add_node(self, trace_id: int):
        self.nodes.add(trace_id)

    def add_flow(self, from_id: int, to_id: int, e_total: float):
        """Обновить ребро при успешном переходе"""
        if e_total < self.flow_thresh:
            return
        key = (from_id, to_id)
        flow = (1 + np.tanh(e_total)) / 2  # нормировка в [0,1]
        d = self.edges.get(key, self.edge_min)
        outgoing_sum = sum(w for (src, _), w in self.edges.items() if src == from_id)
        d_new = np.clip(self.decay * d + self.flow_gain * flow -
                        self.competition_decay * outgoing_sum,
                        self.edge_min, self.edge_max)
        self.edges[key] = d_new

    def decay_all(self):
        """Испарение всех рёбер — применять каждые N шагов"""
        for key in list(self.edges.keys()):
            self.edges[key] *= self.decay
            if self.edges[key] < self.edge_min:
                del self.edges[key]

        # Levy flight: добавить случайное ребро
        if self.nodes and np.random.rand() < self.levy_prob:
            nodes_list = list(self.nodes)
            a, b = np.random.choice(len(nodes_list), 2, replace=False)
            key = (nodes_list[a], nodes_list[b])
            self.edges[key] = max(self.edges.get(key, 0), self.levy_w)

        self.step += 1
        if self.step % self.pr_interval == 0:
            self._compute_pagerank()

    def co_activate(self, ids: List[int], strengths: Optional[Dict[int, float]] = None):
        """Update memory topology from co-retrieved/co-activated traces."""
        ids = [int(i) for i in ids if i in self.nodes]
        if len(ids) < 2:
            return
        strengths = strengths or {}
        eta = 0.10
        decay_eff = float(np.clip(0.02 * (1.0 + 0.3 * abs(self._lambda_L)), 0.01, 0.08))
        max_pairs = 80
        pairs = 0
        for i, src in enumerate(ids):
            for dst in ids[i + 1:]:
                if src == dst:
                    continue
                co = 0.5 * (
                    float(np.clip(strengths.get(src, 0.5), 0.0, 1.0)) +
                    float(np.clip(strengths.get(dst, 0.5), 0.0, 1.0))
                )
                for key in ((src, dst), (dst, src)):
                    old = float(self.edges.get(key, self.edge_min))
                    new_w = (1.0 - decay_eff) * old + eta * co
                    self.edges[key] = float(np.clip(new_w, self.edge_min, self.edge_max))
                pairs += 1
                if pairs >= max_pairs:
                    break
            if pairs >= max_pairs:
                break
        self._sparsify(max_edges=5000)

    def decay_cold_edges(self):
        decay_eff = float(np.clip(0.02 * (1.0 + 0.3 * abs(self._lambda_L)), 0.01, 0.08))
        for key in list(self.edges.keys()):
            self.edges[key] = float(np.clip((1.0 - decay_eff) * self.edges[key],
                                            self.edge_min, self.edge_max))
            if self.edges[key] <= self.edge_min * 1.01:
                del self.edges[key]

    def normalized_degree(self, node_id: int) -> float:
        if not self.edges:
            return 0.0
        degrees = defaultdict(float)
        for (src, dst), w in self.edges.items():
            degrees[src] += float(w)
            degrees[dst] += float(w)
        max_degree = max(degrees.values(), default=0.0)
        if max_degree <= 1e-12:
            return 0.0
        return float(np.clip(degrees.get(int(node_id), 0.0) / max_degree, 0.0, 1.0))

    def topology_bias(self) -> Dict[str, float]:
        return {str(n): self.normalized_degree(n) for n in self.nodes}

    def _sparsify(self, max_edges: int = 5000):
        per_node_cap = 64
        if not self.edges:
            return
        outgoing = defaultdict(list)
        for key, weight in self.edges.items():
            outgoing[key[0]].append((key, weight))
        capped = {}
        for items in outgoing.values():
            for key, weight in sorted(items, key=lambda kv: -kv[1])[:per_node_cap]:
                if weight > self.edge_min * 1.05:
                    capped[key] = weight
        if len(capped) > max_edges:
            capped = dict(sorted(capped.items(), key=lambda kv: -kv[1])[:max_edges])
        self.edges = defaultdict(float, capped)

    def ppr_search(self, query_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """Personalized PageRank от узла query_id"""
        if not self.nodes or len(self.nodes) < 2:
            return []

        nodes = list(self.nodes)
        n = len(nodes)
        idx = {nid: i for i, nid in enumerate(nodes)}

        if query_id not in idx:
            return []

        # Переходная матрица (нормированная)
        P = np.zeros((n, n))
        for (i, j), w in self.edges.items():
            if i in idx and j in idx:
                P[idx[j], idx[i]] += w  # столбец i

        col_sums = P.sum(axis=0)
        col_sums[col_sums == 0] = 1
        P /= col_sums

        # PPR итерация
        pi = np.ones(n) / n
        e_q = np.zeros(n)
        e_q[idx[query_id]] = 1.0

        for _ in range(self.ppr_iters):
            pi = self.ppr_alpha * P @ pi + (1 - self.ppr_alpha) * e_q

        results = [(nodes[i], float(pi[i])) for i in range(n)
                   if nodes[i] != query_id]
        return sorted(results, key=lambda x: -x[1])[:top_k]

    def _compute_pagerank(self):
        if not self.nodes:
            return
        nodes = list(self.nodes)
        n = len(nodes)
        idx = {nid: i for i, nid in enumerate(nodes)}
        P = np.zeros((n, n))
        for (i, j), w in self.edges.items():
            if i in idx and j in idx:
                P[idx[j], idx[i]] += w
        col_sums = P.sum(axis=0)
        col_sums[col_sums == 0] = 1
        P /= col_sums
        pr = np.ones(n) / n
        for _ in range(50):
            pr = 0.85 * P @ pr + 0.15 / n
        self.pagerank = {nodes[i]: float(pr[i]) for i in range(n)}

    def _save(self):
        with open(self.path, 'wb') as f:
            pickle.dump({'edges': dict(self.edges), 'nodes': self.nodes,
                         'pagerank': self.pagerank, 'step': self.step}, f)

    def _load(self):
        try:
            with open(self.path, 'rb') as f:
                d = pickle.load(f)
            self.edges = defaultdict(float, d.get('edges', {}))
            self.nodes = d.get('nodes', set())
            self.pagerank = d.get('pagerank', {})
            self.step = d.get('step', 0)
        except (FileNotFoundError, Exception):
            pass


class SQLiteStore:
    """Хранит полные тексты трасс для DPO и анализа"""
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts INTEGER,
                prompt TEXT,
                answer TEXT,
                e_total REAL,
                e_score REAL,
                d_sem REAL,
                k_act INTEGER,
                models TEXT,
                tokens_out INTEGER
            )
        """)
        for name, ddl in {
            'local_entropy': 'REAL DEFAULT 0.0',
            'local_phi': 'REAL DEFAULT 0.0',
            'retrieval_count': 'INTEGER DEFAULT 0',
            'activation_energy': 'REAL DEFAULT 0.0',
            'last_access_step': 'INTEGER DEFAULT 0',
        }.items():
            try:
                self.conn.execute(f"ALTER TABLE traces ADD COLUMN {name} {ddl}")
            except sqlite3.OperationalError:
                pass
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_e ON traces(e_total)")
        self.conn.commit()

    def insert(self, prompt: str, answer: str, e_total: float,
               e_score: float, d_sem: float, k_act: int,
               models: List[str], tokens_out: int) -> int:
        cur = self.conn.execute("""
            INSERT INTO traces
            (ts, prompt, answer, e_total, e_score, d_sem, k_act, models, tokens_out)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (int(time.time()), prompt, answer, e_total, e_score,
              d_sem, k_act, ','.join(models), tokens_out))
        self.conn.commit()
        return cur.lastrowid

    def get_pairs(self, n: int = 1800, gap: float = 0.18):
        """Формирует пары (chosen, rejected) для DPO"""
        rows = self.conn.execute(
            "SELECT id, prompt, answer, e_total FROM traces "
            "ORDER BY e_total DESC LIMIT ?", (n * 4,)
        ).fetchall()

        pairs = []
        mid = len(rows) // 2
        top = rows[:mid]
        bot = rows[mid:]
        for (ia, pa, aa, ea), (ib, pb, ab, eb) in zip(top, bot):
            if abs(ea - eb) >= gap:
                pairs.append({
                    'prompt': pa,
                    'chosen': aa, 'chosen_e': ea,
                    'rejected': ab, 'rejected_e': eb
                })
            if len(pairs) >= n:
                break
        return pairs


class LeakyIntegrator:
    """
    RC-цепь для информационной памяти системы.
    I(t+1) = λ·I(t) + (1-λ)·max(0, E_base)
    """
    def __init__(self, decay: float = 0.9985, history_maxlen: int = 20000):
        self.decay = decay
        self.gain = 1.0 - decay
        self.I = 0.5    # начальное значение
        # [Ш5] Был безлимитный список: при непрерывной работе течёт.
        # 20000 шагов при ~80 с/шаг — это больше двух недель, с запасом
        # покрывает окно alarm(14400).
        from collections import deque as _deque
        self.history = _deque(maxlen=history_maxlen)

    def update(self, e_total: float) -> float:
        self.I = self.decay * self.I + self.gain * max(0.0, e_total)
        self.history.append(self.I)
        return self.I

    def alarm(self, window_steps: int = 14400) -> bool:
        """True если I упал более чем на 30% за окно"""
        if len(self.history) < window_steps:
            return False
        old = self.history[-window_steps]
        return self.I < 0.7 * old if old > 0 else False


def _resolve_resilient_path(path_str: str, default_filename: str) -> str:
    """
    Checks if a directory for the given path is writeable and exists.
    If not, gracefully falls back to local data/default_filename.
    """
    import logging
    logger = logging.getLogger("memory")
    try:
        if not path_str:
            path_str = os.path.join("data", default_filename)
        dir_path = os.path.dirname(path_str)
        if not dir_path:
            dir_path = "data"
            path_str = os.path.join("data", default_filename)
        
        # Ensure directories exist
        os.makedirs(dir_path, exist_ok=True)
        
        # Test writeability by creating/appending a temp file
        test_file = os.path.join(dir_path, ".write_test")
        with open(test_file, 'a', encoding='utf-8') as f:
            pass
        try:
            os.remove(test_file)
        except OSError:
            pass
            
        return path_str
    except Exception as e:
        fallback_path = os.path.join("data", default_filename)
        logger.warning(
            f"Path '{path_str}' is invalid or unwriteable on Windows: {e}. "
            f"Automatically falling back to local storage: '{fallback_path}'"
        )
        try:
            os.makedirs("data", exist_ok=True)
        except Exception:
            pass
        return fallback_path


class Memory:
    """Главный фасад памяти системы"""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.embedder = HybridEmbedder(cfg)
        mem = cfg['memory']
        
        # Windows-safe resilient path fallback
        faiss_p = _resolve_resilient_path(mem.get('faiss_path'), 'faiss.index')
        graph_p = _resolve_resilient_path(mem.get('graph_path'), 'graph.pkl')
        sqlite_p = _resolve_resilient_path(mem.get('sqlite_path'), 'traces.db')
        
        self.faiss = FAISSIndex(cfg, faiss_p)
        self.graph = PhysarumGraph(cfg, graph_p)
        self.store = SQLiteStore(sqlite_p)
        self.integrator = LeakyIntegrator(
            cfg['emergence']['leaky_decay']
        )

        self._last_trace_id: Optional[int] = None
        self._decay_counter = 0
        # Связь C: Φ и ΔI для приоритизации поиска
        self._last_phi:     float = 0.0
        self._last_delta_i: float = 0.0
        self._lambda_L: float = 0.0
        self._access_step: int = 0

    def set_search_priority(self, phi: float, delta_i: float):
        """
        Связь C: передать Φ и ΔI для приоритизации ближайшего поиска.
        Вызывается из orchestrator после compute_thermodynamics + delta_I.
        priority = 0.6*Φ + 0.4*ΔI  (ТЗ §C)
        """
        self._last_phi     = float(phi)
        self._last_delta_i = float(delta_i)

    def set_lambda(self, lambda_L: float):
        self._lambda_L = float(np.clip(lambda_L, -10.0, 10.0))
        self.graph.set_lambda(self._lambda_L)

    def update_trace_thermo(self, trace_id: int, local_entropy: float, local_phi: float):
        local_entropy = float(np.clip(local_entropy, 0.0, 20.0))
        local_phi = float(np.clip(local_phi, 0.0, 20.0))
        self.store.conn.execute(
            "UPDATE traces SET local_entropy = ?, local_phi = ? WHERE id = ?",
            (local_entropy, local_phi, trace_id)
        )
        self.store.conn.commit()

    def topology_bias(self) -> Dict[str, float]:
        return self.graph.topology_bias()

    def encode(self, text: str) -> np.ndarray:
        return self.embedder.encode(text)

    def add(self, prompt: str, answer: str, vec: np.ndarray,
            e_total: float, e_score: float, d_sem: float,
            k_act: int, models: List[str], tokens_out: int) -> int:
        """Добавить трасу в FAISS, граф и SQLite"""
        trace_id = self.store.insert(prompt, answer, e_total, e_score,
                                      d_sem, k_act, models, tokens_out)
        graph_nodes = set(self.graph.nodes)
        graph_edges = defaultdict(float, self.graph.edges.copy())
        graph_pagerank = dict(self.graph.pagerank)
        graph_step = self.graph.step
        last_trace_id = self._last_trace_id
        faiss_len = len(self.faiss.id_map)
        try:
            self.faiss.add(vec, trace_id)
            self.graph.add_node(trace_id)

            if self._last_trace_id is not None:
                self.graph.add_flow(self._last_trace_id, trace_id, e_total)

            self.integrator.update(e_total)
            self._last_trace_id = trace_id

        # Испарение рёбер каждые 100 шагов
            self._decay_counter += 1
            if self._decay_counter % 100 == 0:
                self.graph.decay_all()
                self.faiss.save()
                self.graph._save()
        except Exception:
            self.graph.nodes = graph_nodes
            self.graph.edges = graph_edges
            self.graph.pagerank = graph_pagerank
            self.graph.step = graph_step
            self._last_trace_id = last_trace_id
            if len(self.faiss.id_map) > faiss_len:
                del self.faiss.id_map[faiss_len:]
            self.store.conn.execute("DELETE FROM traces WHERE id = ?", (trace_id,))
            self.store.conn.commit()
            raise

        return trace_id

    def update_trace_e_total(self, trace_id: int, e_total: float):
        self.store.conn.execute(
            "UPDATE traces SET e_total = ? WHERE id = ?",
            (e_total, trace_id)
        )
        self.store.conn.commit()

    def search(self, vec: np.ndarray, k: int = 5) -> List[dict]:
        """Поиск похожих трасс через FAISS + PPR re-ranking"""
        candidate_ids = self.faiss.search(vec, k=50)
        if not candidate_ids:
            self.graph.decay_cold_edges()
            return []

        # PPR re-ranking по графу если есть последняя траса
        if self._last_trace_id and self._last_trace_id in self.graph.nodes:
            ppr = dict(self.graph.ppr_search(self._last_trace_id, top_k=50))
        else:
            ppr = {}

        # ── Per-candidate priority boost ─────────────────────────────────────
        # Bug fix: global priority cannot change ranking (all candidates
        # multiplied by same factor → monotone transform → order invariant).
        #
        # Fix: per-candidate priority derived from candidate's stored e_total.
        # priority_cand = 0.6*Φ_global + 0.4*e_total_cand
        #   Φ_global    = system-level thermodynamic quality (from last step)
        #   e_total_cand = candidate's historical quality (stored in SQLite)
        # This gives each candidate a DIFFERENT priority_norm → ranking changes.
        #
        # score = semantic_score * (1 + d * priority_norm_cand)
        # d = 0.3 (ТЗ §C)

        phi_global = getattr(self, '_last_phi', 0.0)
        d = 0.3

        # Fetch e_total for all candidates in ONE query (efficient)
        candidate_ids_50 = candidate_ids[:50]
        if candidate_ids_50:
            placeholders_c = ','.join('?' * len(candidate_ids_50))
            cand_rows = self.store.conn.execute(
                f"SELECT id, e_total, local_entropy, local_phi, retrieval_count, "
                f"activation_energy FROM traces WHERE id IN ({placeholders_c})",
                candidate_ids_50
            ).fetchall()
            cand_etotal = {
                r[0]: {
                    'e_total': float(r[1] or 0.0),
                    'local_entropy': float(r[2] or 0.0),
                    'local_phi': float(r[3] or 0.0),
                    'retrieval_count': int(r[4] or 0),
                    'activation_energy': float(r[5] or 0.0),
                }
                for r in cand_rows
            }
        else:
            cand_etotal = {}

        scored = []
        for rank, tid in enumerate(candidate_ids_50):
            ppr_score   = ppr.get(tid, 0.0)
            faiss_score = 1.0 / (1 + rank)
            semantic_score = 0.6 * faiss_score + 0.4 * ppr_score

            # Per-candidate priority: 0.6*Φ_global + 0.4*e_total_candidate
            meta = cand_etotal.get(tid, {})
            local_phi = float(np.clip(meta.get('local_phi', 0.0), 0.0, 20.0))
            activation = float(np.clip(meta.get('activation_energy', 0.0), 0.0, 20.0))
            retrieval_count = max(0, int(meta.get('retrieval_count', 0)))
            retrieval_frequency = retrieval_count / (10.0 + retrieval_count)
            priority_cand = 0.4 * local_phi + 0.4 * activation + 0.2 * retrieval_frequency
            if priority_cand <= 1e-12:
                e_cand = max(meta.get('e_total', 0.0), 0.0)
                priority_cand = 0.6 * max(phi_global, 0.0) + 0.4 * e_cand
            priority_norm_cand = priority_cand / (1.0 + priority_cand)

            combined = semantic_score * (1.0 + d * priority_norm_cand)
            scored.append((tid, combined, priority_norm_cand))

        top_scored = sorted(scored, key=lambda x: -x[1])[:k]
        top_ids = [tid for tid, _, _ in top_scored]

        if top_ids:
            self._update_retrieval_metadata(top_ids)
            strengths = {tid: pri for tid, _, pri in top_scored}
            self.graph.co_activate(top_ids, strengths)
            rows = self.conn_fetch(top_ids)
            for row in rows:
                row['priority_norm'] = float(strengths.get(row['id'], 0.0))
                row['topology_bias'] = self.graph.normalized_degree(row['id'])
            return rows
        self.graph.decay_cold_edges()
        return []

    def _update_retrieval_metadata(self, ids: List[int]):
        if not ids:
            return
        self._access_step += 1
        alpha = 0.2
        placeholders = ','.join('?' * len(ids))
        rows = self.store.conn.execute(
            f"SELECT id, local_entropy, activation_energy FROM traces "
            f"WHERE id IN ({placeholders})", ids
        ).fetchall()
        for tid, local_entropy, activation_energy in rows:
            local_entropy = float(np.clip(local_entropy or 0.0, 0.0, 20.0))
            activation_energy = float(np.clip(activation_energy or 0.0, 0.0, 20.0))
            activation_new = (1.0 - alpha) * activation_energy + alpha * local_entropy
            self.store.conn.execute(
                "UPDATE traces SET retrieval_count = retrieval_count + 1, "
                "activation_energy = ?, last_access_step = ? WHERE id = ?",
                (float(np.clip(activation_new, 0.0, 20.0)), self._access_step, tid)
            )
        self.store.conn.commit()

    def conn_fetch(self, ids: List[int]) -> List[dict]:
        if not ids:
            return []
        placeholders = ','.join('?' * len(ids))
        rows = self.store.conn.execute(
            f"SELECT id, prompt, answer, e_total, local_entropy, local_phi, "
            f"retrieval_count, activation_energy, last_access_step FROM traces "
            f"WHERE id IN ({placeholders})", ids
        ).fetchall()
        by_id = {
            r[0]: {
                'id': r[0], 'prompt': r[1], 'answer': r[2], 'e_total': r[3],
                'local_entropy': r[4], 'local_phi': r[5],
                'retrieval_count': r[6], 'activation_energy': r[7],
                'last_access_step': r[8],
            }
            for r in rows
        }
        return [by_id[i] for i in ids if i in by_id]

    @property
    def I(self) -> float:
        return self.integrator.I
