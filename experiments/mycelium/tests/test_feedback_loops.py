"""
tests/test_feedback_loops.py — Sanity checks: 3 новые обратные связи
"""
import sys, numpy as np
sys.path.insert(0, '.')

PASS="\033[32m✓\033[0m"; FAIL="\033[31m✗\033[0m"; _failures=[]

def check(label, cond, detail=""):
    if cond: print(f"  {PASS} {label}")
    else:
        print(f"  {FAIL} {label}" + (f" | {detail}" if detail else ""))
        _failures.append(label)


# ── #1: R_mem → bandit reward ─────────────────────────────────────────────────
print("\n[#1] R_mem → bandit reward")

# Replicate _compute_r_mem logic
def compute_r_mem(past):
    if not past: return 0.0
    vals = [e/(1.+abs(e)) for p in past for e in [float(p.get('e_total',0))] if e>0]
    return float(np.mean(vals)) if vals else 0.0

def effective_reward(base, H, R_mem, a=0.3, b=0.2):
    H_hat = H/(1.+H)
    reward_adj = base*(1.+a*H_hat)
    return float(np.clip(reward_adj*(1.+b*R_mem), 0., 1.))

past_empty = []
past_low   = [{'e_total': 0.05}]*3
past_high  = [{'e_total': 0.50}]*3

R_empty = compute_r_mem(past_empty)
R_low   = compute_r_mem(past_low)
R_high  = compute_r_mem(past_high)

check("#1.1 empty retrieval → R_mem=0", R_empty == 0.0)
check("#1.2 low-quality retrieval → R_mem small", 0 < R_low < 0.1)
check("#1.3 high-quality retrieval → R_mem larger", R_high > R_low)

r = 0.5; H = 1.0
eff_empty = effective_reward(r, H, R_empty)
eff_high  = effective_reward(r, H, R_high)
check("#1.4 high R_mem → higher reward", eff_high > eff_empty,
      f"eff_high={eff_high:.4f} eff_empty={eff_empty:.4f}")
check("#1.5 reward stays in [0,1]",
      all(0. <= effective_reward(r, H, R) <= 1. for R in [0, 0.3, 1.0]))

src = open('orchestrator.py', encoding='utf-8').read()
check("#1.6 _compute_r_mem in source", '_compute_r_mem' in src)
check("#1.7 R_mem used in _build_bandit_rewards", 'R_mem = getattr(self, \'_r_mem\'' in src)
check("#1.8 b=0.2 coefficient present", '+ b * R_mem' in src)


# ── #2: λ → hypercycle alpha_eff ─────────────────────────────────────────────
print("\n[#2] λ → hypercycle structural coordination")

alpha_base = 0.0207; c_link = 0.5
def alpha_eff(lL):
    return float(np.clip(alpha_base*(1.+c_link*abs(lL)), 0.001, 0.5))

check("#2.1 λ=0 → alpha_eff = alpha_base", abs(alpha_eff(0.0)-alpha_base)<1e-9)
check("#2.2 λ=1 → alpha_eff > alpha_base", alpha_eff(1.0) > alpha_base)
check("#2.3 λ=-1 → alpha_eff > alpha_base (abs)", alpha_eff(-1.0) > alpha_base)
check("#2.4 alpha_eff bounded ≤ 0.5", alpha_eff(100.) <= 0.5)
check("#2.5 larger |λ| → more coordination",
      alpha_eff(2.0) > alpha_eff(1.0) > alpha_eff(0.5))

import yaml
cfg = yaml.safe_load(open('config/settings.yaml', encoding='utf-8'))
cfg['models'] = yaml.safe_load(open('config/models.yaml', encoding='utf-8'))['models']
from core.hypercycle import Hypercycle
hc = Hypercycle(cfg)
check("#2.6 set_lambda exists", hasattr(hc, 'set_lambda'))
# #2.7: λ changes hypercycle dynamics when mean_e is asymmetric.
# With G=0.9 > C=0.1, higher λ amplifies alpha_eff → G's share grows faster.
hc_lo = Hypercycle(cfg); hc_hi = Hypercycle(cfg)
hc_lo.set_lambda(0.0); hc_hi.set_lambda(2.0)
asymm = {'G': 0.9, 'C': 0.1, 'S': 0.3}
for _ in range(10):
    hc_lo.update(asymm); hc_hi.update(asymm)
diff_27 = sum(abs(hc_hi.x[c]-hc_lo.x[c]) for c in ['G','C','S'])
check("#2.7 λ changes hypercycle state with asymmetric rewards",
      diff_27 > 1e-4, f"diff={diff_27:.6f}")

hc_src = open('core/hypercycle.py', encoding='utf-8').read()
check("#2.8 alpha_eff in hypercycle source", 'alpha_eff' in hc_src)
check("#2.9 set_lambda in orchestrator", 'hypercycle.set_lambda' in open('orchestrator.py', encoding='utf-8').read())


# ── #3: priority → retrieval ranking ─────────────────────────────────────────
print("\n[#3] memory priority → retrieval ranking")

d = 0.3
def retrieval_score(faiss_rank, ppr, phi, e_cand):
    priority = 0.6*max(phi,0.) + 0.4*max(e_cand,0.)
    priority_norm = priority/(1.+priority)
    faiss_score = 1./(1+faiss_rank)
    semantic = 0.6*faiss_score + 0.4*ppr
    return semantic*(1.+d*priority_norm)

# Same semantic score, different priority
s_low  = retrieval_score(0, 0.0, phi=0.0, e_cand=0.0)
s_high = retrieval_score(1, 0.75, phi=0.0, e_cand=10.0)
check("#3.1 higher Φ/ΔI → higher retrieval score", s_high > s_low,
      f"s_high={s_high:.4f} s_low={s_low:.4f}")

semantic_low = 0.6 * 1.0 + 0.4 * 0.0
semantic_high = 0.6 * 0.5 + 0.4 * 0.75
check("#3.1b test candidates have equal semantic score",
      abs(semantic_low - semantic_high) < 1e-12,
      f"low={semantic_low:.4f} high={semantic_high:.4f}")

# Same FAISS rank: higher priority → higher score
# (Priority cannot override a large FAISS rank difference — by design:
#  d=0.3 is a gentle boost, not a dominant signal. That's correct.)
s_same_rank_hi = retrieval_score(3, 0.5, phi=0.0, e_cand=10.0)
s_same_rank_lo = retrieval_score(3, 0.5, phi=0.0, e_cand=0.0)
check("#3.2 same rank + high priority > same rank + no priority",
      s_same_rank_hi > s_same_rank_lo,
      f"hi={s_same_rank_hi:.4f} lo={s_same_rank_lo:.4f}")

check("#3.3 priority_norm ∈ [0,1)",
      all(0<=0.6*p/(1.+0.6*p)<1 for p in [0.,0.5,1.,5.,100.]))

mem_src = open('core/memory.py', encoding='utf-8').read()
check("#3.4 new formula in memory.py",
      'semantic_score * (1.0 + d * priority_norm_cand)' in mem_src)
check("#3.5 priority/(1.+priority) formula",
      'priority_cand / (1.0 + priority_cand)' in mem_src)
check("#3.6 degenerate formula removed",
      '(0.4 - 0.2 * priority_norm)' not in mem_src)

try:
    import sqlite3
    from types import SimpleNamespace
    from core.memory import Memory

    class FakeFAISS:
        def search(self, vec, k=50):
            return [1, 2]

    class FakeGraph:
        nodes = {99, 1, 2}

        def ppr_search(self, query_id, top_k=50):
            return [(1, 0.0), (2, 0.75)]

        def co_activate(self, ids, strengths=None):
            pass

        def decay_cold_edges(self):
            pass

        def normalized_degree(self, node_id):
            return 0.0

    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE traces (
            id INTEGER PRIMARY KEY,
            prompt TEXT,
            answer TEXT,
            e_total REAL,
            local_entropy REAL DEFAULT 0.0,
            local_phi REAL DEFAULT 0.0,
            retrieval_count INTEGER DEFAULT 0,
            activation_energy REAL DEFAULT 0.0,
            last_access_step INTEGER DEFAULT 0
        )
    """)
    conn.executemany(
        "INSERT INTO traces(id, prompt, answer, e_total) VALUES (?,?,?,?)",
        [(1, "low", "low priority", 0.0), (2, "high", "high priority", 10.0)],
    )

    mem = object.__new__(Memory)
    mem.faiss = FakeFAISS()
    mem.graph = FakeGraph()
    mem.store = SimpleNamespace(conn=conn)
    mem._last_trace_id = 99
    mem._last_phi = 0.0
    mem._last_delta_i = 0.0
    mem._access_step = 0

    ranked = mem.search(np.zeros(4, dtype=np.float32), k=2)
    check("#3.7 Memory.search returns high-priority candidate first",
          [r["id"] for r in ranked] == [2, 1],
          f"ids={[r['id'] for r in ranked]}")
except Exception as e:
    check("#3.7 Memory.search returns high-priority candidate first", False, str(e))


# ── Old tests still pass ──────────────────────────────────────────────────────
print("\n[old tests] existing connections still intact")
src_o = open('orchestrator.py', encoding='utf-8').read()
src_b = open('core/bandit.py', encoding='utf-8').read()
check("old A: H_hat in reward",      'H_hat = H / (1.0 + H)' in src_o)
check("old B: exploration_boost",    'exploration_boost' in src_b)
check("old B: depth_scale",          'depth_scale' in src_o)
check("old C: set_search_priority",  'set_search_priority' in src_o)
check("no NaN risk: all clips present",
      src_o.count('np.clip') >= 3)


print()
print("="*60)
if _failures:
    print(f"\033[31mFAILED {len(_failures)}: {_failures}\033[0m")
    sys.exit(1)
else:
    print("\033[32mALL SANITY CHECKS PASSED\033[0m")
print("="*60)
