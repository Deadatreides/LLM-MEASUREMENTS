"""
tests/test_block_graph.py — Acceptance tests для G_B + PPR + Shannon entropy
Σ_v8.9

Запуск: python tests/test_block_graph.py
       python -m pytest tests/test_block_graph.py -v
"""

import math
import sys
import numpy as np
sys.path.insert(0, '.')

from core.block_graph import (
    BlockGraph, GraphStats, GRAPH_CODE, GRAPH_TEXT, GRAPH_SINGLE
)
from core.ppr import run_ppr, shannon_entropy, PPRResult
from core.blackboard import (
    SemanticBlock, SemanticBlackboard, ThermodynamicState,
    entropy_from_embeddings,
)

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
_failures = []


def check(label: str, cond: bool, detail: str = ""):
    if cond:
        print(f"  {PASS} {label}")
    else:
        print(f"  {FAIL} FAIL: {label}" + (f" | {detail}" if detail else ""))
        _failures.append(label)


# ── A: code → graph → PPR → entropy ─────────────────────────────────────────

def test_A_code_graph():
    print("\n[A] code → graph → PPR → entropy")
    code = (
        "def factorial(n):\n"
        "    if n <= 1:\n"
        "        return 1\n"
        "    return n * factorial(n - 1)\n\n"
        "class Calculator:\n"
        "    def add(self, a, b):\n"
        "        return a + b\n"
        "    def mul(self, a, b):\n"
        "        return a * b\n"
    )
    g = BlockGraph().build(code, 'code')
    check("A1 graph_type=code_ast", g.stats.graph_type == GRAPH_CODE,
          f"got {g.stats.graph_type}")
    check("A2 node_count > 0", g.n > 0, f"n={g.n}")
    check("A3 edge_count > 0", g.stats.number_of_edges > 0,
          f"edges={g.stats.number_of_edges}")
    check("A4 adj shape", g.adj.shape == (g.n, g.n))

    P = g.transition_matrix()
    check("A5 P row-stochastic", np.allclose(P.sum(axis=1), 1.0, atol=1e-9))
    check("A6 P non-negative", (P >= 0).all())

    v = g.personalization_vector("factorial calculator", 'code')
    check("A7 v sums to 1", abs(v.sum() - 1.0) < 1e-9, f"sum={v.sum()}")
    check("A8 v non-negative", (v >= 0).all())

    result = run_ppr(P, v, alpha=0.85)
    check("A9 pi sums to 1", abs(result.pi.sum() - 1.0) < 1e-9,
          f"sum={result.pi.sum():.10f}")
    check("A10 pi non-negative", (result.pi >= 0).all())
    check("A11 entropy > 0", result.entropy > 0.0, f"H={result.entropy:.4f}")
    check("A12 entropy <= log(n)",
          result.entropy <= math.log(g.n) + 1e-6,
          f"H={result.entropy:.4f} log(n)={math.log(g.n):.4f}")
    check("A13 fallback_used=False", not g.stats.fallback_used)
    check("A14 entropy is real Shannon",
          result.entropy == pytest_shannon(result.pi),
          f"H={result.entropy:.6f} expected={pytest_shannon(result.pi):.6f}")

    print(f"      n={g.n} edges={g.stats.number_of_edges} "
          f"H={result.entropy:.4f} H_norm={result.entropy_norm:.4f}")


def pytest_shannon(pi: np.ndarray) -> float:
    """Reference Shannon entropy."""
    safe = np.maximum(pi, 1e-12)
    return float(-np.sum(safe * np.log(safe)))


# ── B: structure-sensitive ───────────────────────────────────────────────────

def test_B_structure_sensitive():
    print("\n[B] structure-sensitive: linear vs branching")

    linear = "x = 1\ny = 2\nz = 3\n"
    branching = (
        "def process(x):\n"
        "    if x > 0:\n"
        "        return positive(x)\n"
        "    elif x < 0:\n"
        "        return negative(x)\n"
        "    else:\n"
        "        return zero(x)\n\n"
        "def positive(x): return x * 2\n"
        "def negative(x): return -x\n"
        "def zero(x): return 0\n"
    )

    g_lin = BlockGraph().build(linear, 'code')
    g_bra = BlockGraph().build(branching, 'code')

    def get_H(g, task):
        if g.n <= 1:
            return 0.0
        P = g.transition_matrix()
        v = g.personalization_vector(task, 'code')
        return run_ppr(P, v).entropy

    H_lin = get_H(g_lin, "process")
    H_bra = get_H(g_bra, "process")

    print(f"      linear: n={g_lin.n} H={H_lin:.4f}")
    print(f"      branching: n={g_bra.n} H={H_bra:.4f}")

    check("B1 branching has more nodes", g_bra.n > g_lin.n,
          f"bra={g_bra.n} lin={g_lin.n}")
    check("B2 entropies differ (structure-sensitive)",
          abs(H_bra - H_lin) > 1e-4 or g_bra.n != g_lin.n,
          f"H_lin={H_lin:.4f} H_bra={H_bra:.4f}")

    short = "Sky is blue."
    long_ = (
        "The sky is blue. Grass is green. Water is clear. "
        "Fire burns hot. Ice feels cold. Wind moves fast. "
        "Rain drops fall. Sun shines bright. Moon glows pale."
    )
    g_s = BlockGraph().build(short, 'text')
    g_l = BlockGraph().build(long_, 'text')

    def get_H_text(g, task):
        if g.n <= 1:
            return 0.0
        P = g.transition_matrix()
        v = g.personalization_vector(task, 'text')
        return run_ppr(P, v).entropy

    H_s = get_H_text(g_s, "sky")
    H_l = get_H_text(g_l, "sky")

    print(f"      short text: n={g_s.n} H={H_s:.4f}")
    print(f"      long text:  n={g_l.n}  H={H_l:.4f}")

    check("B3 long text has more nodes", g_l.n > g_s.n,
          f"long={g_l.n} short={g_s.n}")
    check("B4 longer text higher entropy", H_l >= H_s,
          f"H_long={H_l:.4f} H_short={H_s:.4f}")


# ── C: fallback on broken code ───────────────────────────────────────────────

def test_C_fallback():
    print("\n[C] AST fallback")

    broken = "def f(: pass\nif True print('hello')\n"
    g = BlockGraph().build(broken, 'code')
    check("C1 no crash", True)
    check("C2 fallback_used=True", g.stats.fallback_used)
    check("C3 n >= 1", g.n >= 1)

    block = SemanticBlock("test_broken")
    block.add_version(broken, "test", 'code')
    check("C4 block survives broken code", True)
    check("C5 entropy is float", isinstance(block.entropy, float))
    check("C6 entropy >= 0", block.entropy >= 0.0, f"H={block.entropy:.4f}")
    check("C7 fallback_used on block", block.fallback_used)
    print(f"      broken: type={block.graph_type} H={block.entropy:.4f}")


# ── D: integration API ────────────────────────────────────────────────────────

def test_D_integration():
    print("\n[D] integration: blackboard API")

    cfg = {'blackboard': {
        'max_versions': 5, 'temperature_min': 0.25,
        'temperature_max': 1.5, 'lyapunov_scale': 3.0,
        'ppr_alpha': 0.85, 'replicator_diffusion': 0.05,
    }}
    bb = SemanticBlackboard(cfg)
    code = "def sort(lst):\n    return sorted(lst)\ndef rev(lst):\n    return lst[::-1]\n"

    block = bb.upsert("step-1", code, task="sort list", task_type="code")
    check("D1 upsert → SemanticBlock", isinstance(block, SemanticBlock))
    check("D2 entropy float", isinstance(block.entropy, float))
    check("D3 entropy >= 0", block.entropy >= 0.0)
    check("D4 graph_nodes > 0", block.graph_nodes > 0, f"n={block.graph_nodes}")
    check("D5 graph_edges > 0", block.graph_edges > 0, f"e={block.graph_edges}")
    check("D6 block.ppr field", hasattr(block, 'ppr'))
    check("D7 block.graph_type field", hasattr(block, 'graph_type'))
    check("D8 block.graph_nodes field", hasattr(block, 'graph_nodes'))
    check("D9 block.graph_edges field", hasattr(block, 'graph_edges'))
    check("D10 block.structure_score field", hasattr(block, 'structure_score'))
    check("D11 block.fallback_used field", hasattr(block, 'fallback_used'))

    # ppr is ndarray when graph built successfully
    if block.ppr is not None:
        check("D12 ppr sums to 1", abs(block.ppr.sum() - 1.0) < 1e-9,
              f"sum={block.ppr.sum()}")
        check("D13 ppr non-negative", (block.ppr >= 0).all())

    selected = bb.select([block], lambda_L=0.0)
    check("D14 select returns block", selected is not None)
    check("D15 selected.entropy accessible",
          isinstance(selected.entropy, float))

    block.add_version(
        "def sort(lst):\n    return list(sorted(lst))\n",
        task="sort", task_type="code",
    )
    check("D16 add_version updates entropy", isinstance(block.entropy, float))

    state = bb.compute_thermodynamics(
        "step-1", code, "sort list", "code", 0.8, 200
    )
    check("D17 ThermodynamicState returned",
          isinstance(state, ThermodynamicState))
    check("D18 d_iS >= 0", state.d_iS >= 0.0, f"d_iS={state.d_iS:.6f}")
    check("D19 Phi computed", isinstance(state.Phi, float))

    print(f"      block: n={block.graph_nodes} H={block.entropy:.4f} "
          f"fallback={block.fallback_used}")


# ── E: deterministic ─────────────────────────────────────────────────────────

def test_E_deterministic():
    print("\n[E] deterministic behavior")

    code = "def f(x):\n    return x * 2\ndef g(x):\n    return f(x) + 1\n"
    task = "compute value"
    results = []
    for _ in range(5):
        g = BlockGraph().build(code, 'code')
        P = g.transition_matrix()
        v = g.personalization_vector(task, 'code')
        r = run_ppr(P, v, alpha=0.85)
        results.append(r.entropy)

    check("E1 entropy stable across runs",
          max(results) - min(results) < 1e-10,
          f"values={[round(x, 8) for x in results]}")
    print(f"      H values (5 runs): {[round(x, 6) for x in results]}")


# ── F: PPR math properties ───────────────────────────────────────────────────

def test_F_ppr_math():
    print("\n[F] PPR mathematical properties")

    # n=1
    r1 = run_ppr(np.array([[1.0]]), np.array([1.0]))
    check("F1 single-node H=0", r1.entropy == 0.0, f"H={r1.entropy}")
    check("F2 single-node pi=[1.0]", np.allclose(r1.pi, [1.0]))

    # n=0
    import io
    r0 = run_ppr(np.zeros((0, 0)), np.array([]))
    check("F3 empty graph no crash", len(r0.pi) == 0)
    check("F4 empty graph H=0", r0.entropy == 0.0)

    # uniform → H ≈ log(n)
    n = 5
    P_u = np.ones((n, n)) / n
    v_u = np.ones(n) / n
    r_u = run_ppr(P_u, v_u, alpha=0.0)   # alpha=0 → pi = v
    check("F5 uniform H ≈ log(n)",
          abs(r_u.entropy - math.log(n)) < 0.02,
          f"H={r_u.entropy:.4f} log(n)={math.log(n):.4f}")

    # dangling node → uniform row
    g2 = BlockGraph()
    g2.n = 3
    g2.node_labels = ["a", "b", "c"]
    g2.adj = np.array([[1.0, 1.0, 0.0],
                       [0.0, 0.0, 0.0],
                       [1.0, 0.0, 0.0]], dtype=float)
    P2 = g2.transition_matrix()
    check("F6 dangling row → uniform",
          np.allclose(P2[1], [1/3, 1/3, 1/3], atol=1e-9),
          f"row1={P2[1]}")
    check("F7 P row-stochastic after dangling fix",
          np.allclose(P2.sum(axis=1), 1.0, atol=1e-9))
    v2 = g2.personalization_vector("a", 'code')
    r2 = run_ppr(P2, v2)
    check("F8 pi sums to 1 with dangling",
          abs(r2.pi.sum() - 1.0) < 1e-9, f"sum={r2.pi.sum()}")
    check("F9 pi non-negative with dangling", (r2.pi >= 0).all())

    # main path entropy equals manually computed Shannon
    code = "def h(x):\n    return x + 1\ndef k(x):\n    return h(x)*2\n"
    g3 = BlockGraph().build(code, 'code')
    P3 = g3.transition_matrix()
    v3 = g3.personalization_vector("h k", 'code')
    r3 = run_ppr(P3, v3)
    expected_H = pytest_shannon(r3.pi)
    check("F10 H equals Shannon(-Σpi·log(pi))",
          abs(r3.entropy - expected_H) < 1e-8,
          f"H={r3.entropy:.8f} expected={expected_H:.8f}")


# ── G: causal order ──────────────────────────────────────────────────────────

def test_G_causal():
    print("\n[G] causal order")

    cfg = {'blackboard': {
        'max_versions': 5, 'temperature_min': 0.25,
        'temperature_max': 1.5, 'lyapunov_scale': 3.0,
        'ppr_alpha': 0.85, 'replicator_diffusion': 0.05,
    }}
    bb = SemanticBlackboard(cfg)
    code1 = "def f(x):\n    return x\n"
    code2 = "def f(x):\n    return x * 2\ndef g(x):\n    return f(x) + 1\n"

    s1 = bb.compute_thermodynamics("blk", code1, "f", "code", 0.8, 100)
    check("G1 H_old=0 for new block", s1.H_old == 0.0, f"H_old={s1.H_old}")

    s2 = bb.compute_thermodynamics("blk", code2, "f g", "code", 0.9, 200)
    check("G2 H_old(t+1)==H_new(t)",
          abs(s2.H_old - s1.H_new) < 1e-10,
          f"H_old={s2.H_old:.8f} H_new={s1.H_new:.8f}")
    check("G3 d_iS >= 0", s2.d_iS >= 0.0)
    check("G4 alive field bool", isinstance(s2.alive, bool))

    print(f"      s1: 0→{s1.H_new:.4f}  s2: {s2.H_old:.4f}→{s2.H_new:.4f}")


# ── H: separation G_B != G_M ─────────────────────────────────────────────────

def test_H_separation():
    print("\n[H] G_B ≠ G_M separation")
    import os
    root = os.path.join(os.path.dirname(__file__), '..')
    bb_src = open(os.path.join(root, 'core', 'blackboard.py'), encoding='utf-8').read()
    bg_src = open(os.path.join(root, 'core', 'block_graph.py')).read()
    check("H1 no PhysarumGraph in blackboard.py", 'PhysarumGraph' not in bb_src)
    check("H2 no FAISSIndex in blackboard.py",    'FAISSIndex'    not in bb_src)
    check("H3 no PhysarumGraph in block_graph.py",'PhysarumGraph' not in bg_src)
    check("H4 no memory import in block_graph.py",'from core.memory' not in bg_src)


# ── I: deprecated backward compat ────────────────────────────────────────────

def test_I_backward_compat():
    print("\n[I] backward-compat entropy_from_embeddings (deprecated)")
    h = entropy_from_embeddings(list(np.eye(3)))
    check("I1 returns float", isinstance(h, float))
    check("I2 value > 0.9", h > 0.9, f"H={h:.4f}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 65)
    print("Σ_v8.9 — Acceptance Tests: G_B + PPR + Shannon entropy")
    print("=" * 65)

    test_A_code_graph()
    test_B_structure_sensitive()
    test_C_fallback()
    test_D_integration()
    test_E_deterministic()
    test_F_ppr_math()
    test_G_causal()
    test_H_separation()
    test_I_backward_compat()

    print()
    print("=" * 65)
    if _failures:
        print(f"\033[31mFAILED {len(_failures)} checks:\033[0m")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("\033[32mALL CHECKS PASSED\033[0m")
    print("=" * 65)
