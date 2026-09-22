import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import yaml

sys.path.insert(0, ".")

from core.bandit import SwarmBandit
from core.blackboard import SemanticBlackboard
from core.memory import PhysarumGraph, SQLiteStore


PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
failures = []


def check(label, cond, detail=""):
    if cond:
        print(f"  {PASS} {label}")
    else:
        print(f"  {FAIL} {label}" + (f" | {detail}" if detail else ""))
        failures.append(label)


def load_cfg():
    with open("config/settings.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open("config/models.yaml", encoding="utf-8") as f:
        cfg["models"] = yaml.safe_load(f)["models"]
    return cfg


cfg = load_cfg()


print("\n[A] memory graph co-evolution")
with tempfile.TemporaryDirectory() as td:
    g = PhysarumGraph(cfg, os.path.join(td, "graph.pkl"))
    g.nodes.update({1, 2, 3})
    w0 = g.edges.get((1, 2), g.edge_min)
    for _ in range(5):
        g.co_activate([1, 2], {1: 0.8, 2: 0.8})
    w1 = g.edges[(1, 2)]
    check("A1 repeated co-activation increases edge weight", w1 > w0,
          f"w0={w0:.5f} w1={w1:.5f}")
    g.decay_cold_edges()
    w2 = g.edges.get((1, 2), 0.0)
    check("A2 no activation decays edge weight", w2 < w1,
          f"w1={w1:.5f} w2={w2:.5f}")


print("\n[B] contextual bandit changes selection dynamics")
models = [
    {"id": "strong", "provider": "p1", "family": "f", "roles": ["G"], "max_rpm": 100000, "mu": 0.60, "sigma": 0.05},
    {"id": "risky", "provider": "p2", "family": "f", "roles": ["G"], "max_rpm": 100000, "mu": 0.50, "sigma": 0.05},
]
cfg_b = dict(cfg)
cfg_b["bandit"] = dict(cfg["bandit"])
cfg_b["bandit"]["max_per_provider"] = 10


def risky_count(context_complexity):
    np.random.seed(7)
    b = SwarmBandit(cfg_b, models)
    b.set_memory_context(context_complexity=context_complexity)
    count = 0
    task_vec = np.ones(32, dtype=np.float32)
    for _ in range(600):
        for bucket in b.buckets.values():
            bucket.tokens = bucket.max_tokens
        selected = b.select(1, task_vec, allowed_roles=["G"])
        count += int(selected == ["risky"])
    return count


low_ctx = risky_count(0.0)
high_ctx = risky_count(1.0)
check("B1 same rewards, different context changes selection dynamics",
      high_ctx != low_ctx, f"low={low_ctx} high={high_ctx}")


print("\n[C] lambda changes memory topology evolution")
with tempfile.TemporaryDirectory() as td:
    g_low = PhysarumGraph(cfg, os.path.join(td, "low.pkl"))
    g_high = PhysarumGraph(cfg, os.path.join(td, "high.pkl"))
    g_low.nodes.update({1, 2})
    g_high.nodes.update({1, 2})
    g_low.set_lambda(0.0)
    g_high.set_lambda(3.0)
    for _ in range(10):
        g_low.co_activate([1, 2], {1: 0.7, 2: 0.7})
        g_high.co_activate([1, 2], {1: 0.7, 2: 0.7})
    w_low = g_low.edges[(1, 2)]
    w_high = g_high.edges[(1, 2)]
    check("C1 same activation, different lambda -> different edge weight",
          abs(w_low - w_high) > 1e-6, f"low={w_low:.6f} high={w_high:.6f}")


print("\n[D] topology affects blackboard selection entropy")
bb = SemanticBlackboard(cfg)
blocks = [SimpleNamespace(id="1", entropy=1.0), SimpleNamespace(id="2", entropy=1.0)]
p0 = bb.probabilities(blocks, lambda_L=0.0)
bb.set_topology_bias({"1": 0.0, "2": 1.0})
p1 = bb.probabilities(blocks, lambda_L=0.0)
check("D1 higher topology bias increases selection probability",
      p1[1] > p0[1], f"p0={p0.tolist()} p1={p1.tolist()}")


print("\n[E] per-block thermodynamics and stability")
with tempfile.TemporaryDirectory() as td:
    store = SQLiteStore(os.path.join(td, "traces.db"))
    trace_id = store.insert("p", "a", 0.2, 0.0, 0.0, 1, ["m"], 1)
    cols = store.conn.execute(
        "SELECT local_entropy, local_phi, retrieval_count, activation_energy, last_access_step "
        "FROM traces WHERE id = ?", (trace_id,)
    ).fetchone()
    check("E1 metadata columns persist per block", len(cols) == 5)
    store.conn.close()

with tempfile.TemporaryDirectory() as td:
    g = PhysarumGraph(cfg, os.path.join(td, "stable.pkl"))
    g.nodes.update(range(20))
    rng = np.random.default_rng(11)
    for step in range(1000):
        ids = rng.choice(20, size=3, replace=False).tolist()
        strength = {int(i): float(rng.random()) for i in ids}
        g.set_lambda(float(rng.normal(0.0, 1.0)))
        g.co_activate(ids, strength)
        if step % 3 == 0:
            g.decay_cold_edges()
    weights = np.array(list(g.edges.values()), dtype=float)
    check("E2 long-run graph has finite weights",
          weights.size > 0 and np.isfinite(weights).all())
    check("E3 long-run graph stays clipped and sparse",
          len(g.edges) <= 5000 and weights.max() <= g.edge_max + 1e-9,
          f"edges={len(g.edges)} max={weights.max() if weights.size else 0}")
    check("E4 long-run graph does not collapse",
          weights.size > 0 and weights.sum() > 0.0)


print()
print("=" * 60)
if failures:
    print(f"\033[31mFAILED {len(failures)}: {failures}\033[0m")
    sys.exit(1)
print("\033[32mALL STRUCTURAL LOOP TESTS PASSED\033[0m")
print("=" * 60)
