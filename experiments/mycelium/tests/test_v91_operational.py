import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import yaml

sys.path.insert(0, ".")

from core.bandit import SwarmBandit
from core.blackboard import SemanticBlackboard
from core.block_graph import BlockGraph, GRAPH_CODE
from core.memory import Memory, SQLiteStore, PhysarumGraph, LeakyIntegrator
from core.scorer import QualityScorer


PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
failures = []


def check(label, cond, detail=""):
    if cond:
        print(f"  {PASS} {label}")
    else:
        print(f"  {FAIL} {label}" + (f" | {detail}" if detail else ""))
        failures.append(label)


def cfg():
    with open("config/settings.yaml", encoding="utf-8") as f:
        c = yaml.safe_load(f)
    with open("config/models.yaml", encoding="utf-8") as f:
        c["models"] = yaml.safe_load(f)["models"]
    return c


class FakeFAISS:
    def __init__(self, fail_add=False, fail_save=False):
        self.id_map = []
        self.fail_add = fail_add
        self.fail_save = fail_save

    def add(self, vec, trace_id):
        if self.fail_add:
            raise RuntimeError("faiss add failed")
        self.id_map.append(trace_id)

    def save(self):
        if self.fail_save:
            raise RuntimeError("faiss save failed")


def make_memory(c, td, faiss):
    m = object.__new__(Memory)
    m.cfg = c
    m.faiss = faiss
    m.graph = PhysarumGraph(c, os.path.join(td, "graph.pkl"))
    m.store = SQLiteStore(os.path.join(td, "traces.db"))
    m.integrator = LeakyIntegrator(c["emergence"]["leaky_decay"])
    m._last_trace_id = None
    m._decay_counter = 0
    m._last_phi = 0.0
    m._last_delta_i = 0.0
    m._lambda_L = 0.0
    m._access_step = 0
    return m


c = cfg()

print("\n[A] transactional memory consistency")
with tempfile.TemporaryDirectory() as td:
    m = make_memory(c, td, FakeFAISS())
    def fail_insert(*args, **kwargs):
        raise RuntimeError("sqlite insert failed")
    m.store.insert = fail_insert
    try:
        m.add("p", "a", np.zeros(4), 0.5, 0.0, 0.0, 1, ["m"], 1)
        ok = False
    except RuntimeError:
        ok = True
    rows = m.store.conn.execute("SELECT COUNT(*) FROM traces").fetchone()[0]
    check("A0 SQLite failure leaves graph/index/store unchanged", ok and rows == 0 and not m.graph.nodes and not m.faiss.id_map)
    m.store.conn.close()

with tempfile.TemporaryDirectory() as td:
    m = make_memory(c, td, FakeFAISS(fail_add=True))
    try:
        m.add("p", "a", np.zeros(4), 0.5, 0.0, 0.0, 1, ["m"], 1)
        ok = False
    except RuntimeError:
        ok = True
    rows = m.store.conn.execute("SELECT COUNT(*) FROM traces").fetchone()[0]
    check("A1 FAISS add failure rolls back SQLite and graph", ok and rows == 0 and not m.graph.nodes and not m.faiss.id_map)
    m.store.conn.close()

with tempfile.TemporaryDirectory() as td:
    m = make_memory(c, td, FakeFAISS(fail_save=True))
    m._decay_counter = 99
    try:
        m.add("p", "a", np.zeros(4), 0.5, 0.0, 0.0, 1, ["m"], 1)
        ok = False
    except RuntimeError:
        ok = True
    rows = m.store.conn.execute("SELECT COUNT(*) FROM traces").fetchone()[0]
    check("A2 interrupted save rolls back topology/index/store", ok and rows == 0 and not m.graph.nodes and not m.faiss.id_map)
    m.store.conn.close()


print("\n[B] provider circuit breaker")
models = [
    {"id": "bad", "provider": "badp", "family": "f", "roles": ["G"], "max_rpm": 100000, "mu": 0.9, "sigma": 0.01},
    {"id": "good", "provider": "goodp", "family": "f", "roles": ["G"], "max_rpm": 100000, "mu": 0.6, "sigma": 0.01},
]
b = SwarmBandit(c, models)
for _ in range(3):
    b.record_provider_result("badp", ok=False, latency=30)
for bucket in b.buckets.values():
    bucket.tokens = bucket.max_tokens
selected = b.select(1, np.ones(32), allowed_roles=["G"])
check("B1 failing provider disabled, healthy continues", selected == ["good"], f"selected={selected}")
b.provider_health["badp"].cooldown_until = 0.0
b.record_provider_result("badp", ok=True, latency=1)
check("B2 provider recovery clears infinite disable", b.provider_available("badp"))
b.record_provider_result("goodp", ok=True, latency=90)
penalty = b._provider_latency_penalty("goodp")
check("B3 latency penalty bounded, not starvation", 0.65 <= penalty < 1.0, f"penalty={penalty:.3f}")


print("\n[C] topology affects thermodynamics")
bb0 = SemanticBlackboard(c)
bb1 = SemanticBlackboard(c)
text = "This block has multiple sentences. It should form a non-zero entropy graph. Another sentence adds structure."
s0 = bb0.compute_thermodynamics("blk", text, "task", "general", q_env=0.8, tokens_used=100)
bb1.set_topology_bias({"blk": 1.0})
s1 = bb1.compute_thermodynamics("blk", text, "task", "general", q_env=0.8, tokens_used=100)
check("C1 topology bias changes actual thermodynamic H_new", s1.H_new > s0.H_new,
      f"H0={s0.H_new:.4f} H1={s1.H_new:.4f}")


print("\n[D] semantic code autodetection")
code = """
def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)

class Solver:
    def run(self):
        return fib(5)
"""
g = BlockGraph().build(code, task_type="general")
check("D1 general task with code builds AST graph", g.stats.graph_type == GRAPH_CODE,
      f"type={g.stats.graph_type}")


print("\n[E] Windows sandbox guard")
sc = QualityScorer(c)
score = sc._q_code_safe("def f():\n    return 1\n")
if sys.platform == "win32":
    check("E1 Windows code execution is not silently enabled", score <= 0.3, f"score={score}")
else:
    check("E1 non-Windows sandbox still available", score >= 0.2, f"score={score}")


print("\n[F] long-run bounded topology")
with tempfile.TemporaryDirectory() as td:
    g = PhysarumGraph(c, os.path.join(td, "g.pkl"))
    g.nodes.update(range(120))
    rng = np.random.default_rng(91)
    for step in range(1200):
        ids = rng.choice(120, size=6, replace=False).tolist()
        g.set_lambda(float(rng.normal()))
        g.co_activate(ids, {int(i): float(rng.random()) for i in ids})
        if step % 4 == 0:
            g.decay_cold_edges()
    weights = np.array(list(g.edges.values()), dtype=float)
    max_out = {}
    for src, _ in g.edges:
        max_out[src] = max_out.get(src, 0) + 1
    check("F1 no NaN propagation", weights.size > 0 and np.isfinite(weights).all())
    check("F2 edge count remains bounded", len(g.edges) <= 5000, f"edges={len(g.edges)}")
    check("F3 per-node density cap active", max(max_out.values(), default=0) <= 64)


print()
print("=" * 60)
if failures:
    print(f"\033[31mFAILED {len(failures)}: {failures}\033[0m")
    sys.exit(1)
print("\033[32mALL V9.1 OPERATIONAL TESTS PASSED\033[0m")
print("=" * 60)
