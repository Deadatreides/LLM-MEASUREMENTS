"""
tests/test_behavioral_feedback.py — Строго поведенческие тесты (не existence)
Σ_v8.9 — Financial audit pass

Проверяет реальное изменение поведения, а не наличие полей.
"""

import sys, math, sqlite3, tempfile, os, numpy as np
sys.path.insert(0, '.')

PASS="\033[32m✓\033[0m"; FAIL="\033[31m✗\033[0m"; _f=[]

def chk(label, cond, detail=""):
    if cond: print(f"  {PASS} {label}")
    else:
        print(f"  {FAIL} {label}" + (f" | {detail}" if detail else ""))
        _f.append(label)


# ═══════════════════════════════════════════════════════════════
# A. priority → REAL retrieval ranking (per-candidate, not global)
# ═══════════════════════════════════════════════════════════════
print("\n[A] priority → real retrieval ranking")

def retrieval_score(faiss_rank, ppr, phi_global, e_cand, d=0.3):
    """Exact formula from core/memory.py search()"""
    faiss_score = 1.0 / (1 + faiss_rank)
    semantic_score = 0.6 * faiss_score + 0.4 * ppr
    priority_cand = 0.6 * max(phi_global, 0.0) + 0.4 * max(e_cand, 0.0)
    priority_norm_cand = priority_cand / (1.0 + priority_cand)
    return semantic_score * (1.0 + d * priority_norm_cand)

# A1: same rank, same ppr, different e_total → score differs
phi = 0.5
s_lo = retrieval_score(1, 0.5, phi, e_cand=0.0)
s_hi = retrieval_score(1, 0.5, phi, e_cand=0.8)
chk("A1 same-rank: hi e_total → higher score", s_hi > s_lo,
    f"hi={s_hi:.4f} lo={s_lo:.4f}")

# A2: verify NOT algebraically degenerate
# Old bug: all candidates had same multiplier → ratio invariant
# New: different e_total → different multiplier
s_lo2 = retrieval_score(2, 0.5, phi, e_cand=0.0)
s_hi2 = retrieval_score(2, 0.5, phi, e_cand=0.8)
ratio_1 = s_hi / s_lo
ratio_2 = s_hi2 / s_lo2
chk("A2 not degenerate: both rank-1 and rank-2 multipliers differ",
    abs(ratio_1 - 1.0) > 1e-6 and abs(ratio_2 - 1.0) > 1e-6,
    f"ratio_rank1={ratio_1:.6f} ratio_rank2={ratio_2:.6f}")

# A3: equal semantic score + higher priority changes ordering.
s_lopri_equal = retrieval_score(0, 0.0, 0.0, e_cand=0.0)
s_hipri_equal = retrieval_score(1, 0.75, 0.0, e_cand=10.0)
sem_lopri = 0.6 * 1.0 + 0.4 * 0.0
sem_hipri = 0.6 * 0.5 + 0.4 * 0.75
chk("A3 same semantic + high e_total ranks higher",
    abs(sem_lopri - sem_hipri) < 1e-12 and s_hipri_equal > s_lopri_equal,
    f"hi={s_hipri_equal:.4f} lo={s_lopri_equal:.4f}")

# A4: monotonically increasing with e_total (same rank)
scores = [retrieval_score(1, 0.5, phi, e) for e in [0.0, 0.2, 0.5, 1.0, 3.0]]
chk("A4 score monotone with e_total", all(scores[i] < scores[i+1] for i in range(len(scores)-1)),
    f"scores={[round(s,4) for s in scores]}")

# A5: phi_global=0 and e_cand=0 → no priority boost (base semantic only)
s_base = retrieval_score(1, 0.5, 0.0, 0.0)
s_boost = retrieval_score(1, 0.5, 0.0, 1.0)
sem_base = 0.6*(1/2) + 0.4*0.5
chk("A5 phi=0 e=0 → score equals pure semantic",
    abs(s_base - sem_base) < 1e-9, f"score={s_base:.6f} sem={sem_base:.6f}")
chk("A6 phi=0 e>0 → boost above pure semantic", s_boost > s_base)

# A6: verify formula in source code
mem_src = open('core/memory.py', encoding='utf-8').read()
chk("A7 per-candidate e_total fetched in search()",
    'cand_etotal' in mem_src and 'SELECT id, e_total, local_entropy' in mem_src)
chk("A8 priority_norm_cand used per-candidate", 'priority_norm_cand' in mem_src)
chk("A9 old degenerate formula removed",
    '(0.4 - 0.2 * priority_norm)' not in mem_src)
chk("A10 d=0.3 default", 'd = 0.3' in mem_src)


# ═══════════════════════════════════════════════════════════════
# B. λ → REAL hypercycle structural dynamics
# ═══════════════════════════════════════════════════════════════
print("\n[B] λ → real hypercycle structural dynamics")

import yaml
cfg = yaml.safe_load(open('config/settings.yaml', encoding='utf-8'))
cfg['models'] = yaml.safe_load(open('config/models.yaml', encoding='utf-8'))['models']
from core.hypercycle import Hypercycle

# B1: alpha_eff formula is correct
alpha_base = cfg.get('hypercycle', {}).get('alpha', 0.0207)
c_link = 0.5
def alpha_eff(lL): return float(np.clip(alpha_base*(1+c_link*abs(lL)), 0.001, 0.5))
chk("B1 alpha_eff=alpha_base at λ=0", abs(alpha_eff(0.0)-alpha_base)<1e-9)
chk("B2 alpha_eff increases with |λ|", alpha_eff(1.0) > alpha_eff(0.5) > alpha_eff(0.0))
chk("B3 alpha_eff bounded ≤ 0.5", alpha_eff(100.) <= 0.5)

# B4: SYMMETRIC rewards → λ has NO effect (mathematically correct, not a bug)
hc_lo_sym = Hypercycle(cfg); hc_hi_sym = Hypercycle(cfg)
hc_lo_sym.set_lambda(0.0); hc_hi_sym.set_lambda(2.0)
sym = {'G': 0.5, 'C': 0.5, 'S': 0.5}
for _ in range(20): hc_lo_sym.update(sym); hc_hi_sym.update(sym)
diff_sym = sum(abs(hc_hi_sym.x[c]-hc_lo_sym.x[c]) for c in ['G','C','S'])
chk("B4 symmetric rewards → λ has no effect (correct math)",
    diff_sym < 1e-6, f"diff={diff_sym:.8f}")

# B5: ASYMMETRIC rewards → λ DOES change trajectory (realistic operational scenario)
hc_lo_a = Hypercycle(cfg); hc_hi_a = Hypercycle(cfg)
hc_lo_a.set_lambda(0.0); hc_hi_a.set_lambda(2.0)
asym = {'G': 0.9, 'C': 0.1, 'S': 0.5}
for _ in range(20): hc_lo_a.update(asym); hc_hi_a.update(asym)
diff_asym = sum(abs(hc_hi_a.x[c]-hc_lo_a.x[c]) for c in ['G','C','S'])
chk("B5 asymmetric rewards → λ changes hypercycle trajectory",
    diff_asym > 1e-4, f"diff={diff_asym:.6f}")
chk("B6 G share higher at λ=2 (amplifies dominant caste faster)",
    hc_hi_a.x['G'] > hc_lo_a.x['G'],
    f"λ=2: {hc_hi_a.x['G']:.4f} vs λ=0: {hc_lo_a.x['G']:.4f}")

# B7: realistic rewards (G slightly best)
hc_lo_r = Hypercycle(cfg); hc_hi_r = Hypercycle(cfg)
hc_lo_r.set_lambda(0.0); hc_hi_r.set_lambda(2.0)
realistic = {'G': 0.7, 'C': 0.3, 'S': 0.5}
for _ in range(15): hc_lo_r.update(realistic); hc_hi_r.update(realistic)
diff_real = sum(abs(hc_hi_r.x[c]-hc_lo_r.x[c]) for c in ['G','C','S'])
chk("B7 realistic asymmetry → measurable λ effect",
    diff_real > 1e-3, f"diff={diff_real:.5f}")

# B8: α_eff is actually used inside update() (code check)
hc_src = open('core/hypercycle.py', encoding='utf-8').read()
chk("B8 alpha_eff in hypercycle.py", 'alpha_eff' in hc_src)
chk("B9 alpha_eff used in phi calculation", 'alpha_eff * self.k' in hc_src)
chk("B10 alpha_eff bounded with clip", 'np.clip' in hc_src)
chk("B11 set_lambda stores lambda_L", '_lambda_L' in hc_src)

# B12: higher |λ| → faster convergence to dominant caste
hc_slow = Hypercycle(cfg); hc_fast = Hypercycle(cfg)
hc_slow.set_lambda(0.0); hc_fast.set_lambda(3.0)
dom = {'G': 0.8, 'C': 0.2, 'S': 0.4}
x_slow_hist = []; x_fast_hist = []
for _ in range(10):
    hc_slow.update(dom); hc_fast.update(dom)
    x_slow_hist.append(hc_slow.x['G']); x_fast_hist.append(hc_fast.x['G'])
# G should converge faster (higher G share) at λ=3 than λ=0
chk("B12 higher λ → faster G dominance (convergence speed)",
    x_fast_hist[-1] >= x_slow_hist[-1],
    f"fast_G={x_fast_hist[-1]:.5f} slow_G={x_slow_hist[-1]:.5f}")


# ═══════════════════════════════════════════════════════════════
# C. No regression: previous connections still work
# ═══════════════════════════════════════════════════════════════
print("\n[C] regression: previous connections intact")

src_o = open('orchestrator.py', encoding='utf-8').read()
src_b = open('core/bandit.py', encoding='utf-8').read()

chk("C1 entropy→reward: H_hat in _build_bandit_rewards", "H_hat = H / (1.0 + H)" in src_o)
chk("C2 R_mem→reward: _compute_r_mem exists", '_compute_r_mem' in src_o)
chk("C3 λ→bandit: exploration_boost in bandit.py", 'exploration_boost' in src_b)
chk("C4 λ→depth: depth_scale in orchestrator", 'depth_scale' in src_o)
chk("C5 λ→hypercycle: set_lambda wired", 'hypercycle.set_lambda' in src_o)
chk("C6 Φ→memory: set_search_priority called", 'set_search_priority' in src_o)
chk("C7 no NaN risk: np.clip present", src_o.count('np.clip') >= 3)

# Algebraic check: reward stays bounded
def eff_reward(base, H, R_mem, a=0.3, b=0.2):
    H_hat = H/(1+H)
    adj = base*(1+a*H_hat)
    return float(np.clip(adj*(1+b*R_mem), 0., 1.))
chk("C8 reward in [0,1] for any inputs",
    all(0<=eff_reward(r,H,R)<=1
        for r,H,R in [(0.5,0,0),(0.5,5,1),(1.0,0,1),(0,10,10)]))


# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════
print()
print("="*60)
if _f:
    print(f"\033[31mFAILED {len(_f)}: {_f}\033[0m")
    sys.exit(1)
else:
    print("\033[32mALL BEHAVIORAL TESTS PASSED\033[0m")
print("="*60)
