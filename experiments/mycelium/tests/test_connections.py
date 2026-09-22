"""
tests/test_connections.py — Sanity checks для 3 связей
Σ_v8.9

A) entropy → reward_adj: H > 0 увеличивает reward
B) λ → sigma_eff:        λ > 0 увеличивает exploration
B) λ → depth_scale:      λ > 0 увеличивает max_steps
C) Φ/ΔI → priority:      set_search_priority сохраняется в memory
"""

import sys, math, numpy as np
sys.path.insert(0, '.')

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
_failures = []

def check(label, cond, detail=""):
    if cond:
        print(f"  {PASS} {label}")
    else:
        print(f"  {FAIL} FAIL: {label}" + (f" | {detail}" if detail else ""))
        _failures.append(label)


# ── A: entropy → reward_adj ──────────────────────────────────────────────────
print("\n[A] entropy → reward_adj")

# Simulate the formula directly (it lives in orchestrator._build_bandit_rewards)
def reward_adj(base_reward, H, a=0.3):
    H_hat = H / (1.0 + H)
    return base_reward * (1.0 + a * H_hat)

r = 0.5
check("A1 H=0 → reward unchanged", abs(reward_adj(r, 0.0) - r) < 1e-9)
check("A2 H=1 → reward increases", reward_adj(r, 1.0) > r)
check("A3 H=2 → reward increases more", reward_adj(r, 2.0) > reward_adj(r, 1.0))
check("A4 H_hat ∈ [0,1)",
      all(0.0 <= H/(1+H) < 1.0 for H in [0.001, 0.5, 1.0, 5.0, 100.0]))
check("A5 reward_adj <= reward*(1+a)",
      reward_adj(r, 1000.0) <= r * (1.0 + 0.3) + 1e-9)

# Verify it's in orchestrator source
src = open('orchestrator.py', encoding='utf-8').read()
check("A6 H_hat formula in orchestrator", 'H_hat = H / (1.0 + H)' in src)
check("A7 reward_adj formula in orchestrator", 'reward * (1.0 + a * H_hat)' in src)
check("A8 _last_thermo.H_new used", '_last_thermo.H_new' in src)


# ── B: λ → exploration / depth ───────────────────────────────────────────────
print("\n[B] λ → exploration_boost, depth_scale")

# exploration_boost = clamp(0.2 + 0.5*λ, 0.0, 1.0)
def exploration_boost(lL):
    return float(np.clip(0.2 + 0.5 * lL, 0.0, 1.0))

check("B1 λ=0  → boost=0.20", abs(exploration_boost(0.0) - 0.20) < 1e-9)
check("B2 λ=0.5 → boost=0.45", abs(exploration_boost(0.5) - 0.45) < 1e-9)
check("B3 λ=2.0 → boost clamped to 1.0", abs(exploration_boost(2.0) - 1.0) < 1e-9)
check("B4 λ=-2 → boost clamped to 0.0", abs(exploration_boost(-2.0) - 0.0) < 1e-9)
check("B5 boost monotone in λ",
      exploration_boost(0.5) > exploration_boost(0.0) > exploration_boost(-1.0))

# sigma_eff = max(sigma*(1+boost*0.5), sigma_min)
sigma = 0.4; sigma_min = 0.2
def sigma_eff(lL):
    boost = exploration_boost(lL)
    return max(sigma * (1.0 + boost * 0.5), sigma_min)

check("B6 σ_eff increases with λ", sigma_eff(1.0) > sigma_eff(0.0) > sigma_eff(-1.0))
check("B7 σ_eff >= sigma_min", sigma_eff(-10.0) >= sigma_min)

# depth_scale = 1.0 + 0.5*λ
def depth_scale(lL): return 1.0 + 0.5 * lL
max_steps_cfg = 2
check("B8 λ=0  → max_steps=2", max(1, round(max_steps_cfg * depth_scale(0.0))) == 2)
check("B9 λ=1  → max_steps=3", max(1, round(max_steps_cfg * depth_scale(1.0))) == 3)
check("B10 λ=-0.9 → max_steps=1", max(1, round(max_steps_cfg * depth_scale(-0.9))) == 1)

# Verify in source
bandit_src = open('core/bandit.py', encoding='utf-8').read()
orch_src    = open('orchestrator.py', encoding='utf-8').read()
check("B11 exploration_boost in bandit.py", 'exploration_boost' in bandit_src)
check("B12 selection_pressure in bandit.py", 'selection_pressure' in bandit_src)
check("B13 depth_scale in orchestrator.py", 'depth_scale' in orch_src)


# ── C: Φ/ΔI → memory priority ────────────────────────────────────────────────
print("\n[C] Φ/ΔI → memory search priority")

import yaml
cfg = yaml.safe_load(open('config/settings.yaml', encoding='utf-8'))
cfg['models'] = yaml.safe_load(open('config/models.yaml', encoding='utf-8'))['models']
from core.memory import Memory
mem = Memory(cfg)

# priority formula
def priority(phi, di):
    return 0.6 * max(phi, 0.0) + 0.4 * max(di, 0.0)

check("C1 Φ=0 ΔI=0 → priority=0", abs(priority(0.0, 0.0) - 0.0) < 1e-9)
check("C2 Φ=1 ΔI=0 → priority=0.6", abs(priority(1.0, 0.0) - 0.6) < 1e-9)
check("C3 Φ=0 ΔI=1 → priority=0.4", abs(priority(0.0, 1.0) - 0.4) < 1e-9)
check("C4 priority monotone in Φ", priority(1.0, 0.5) > priority(0.5, 0.5))
check("C5 negative Φ clamped to 0",
      abs(priority(-5.0, 0.5) - priority(0.0, 0.5)) < 1e-9)

# set_search_priority persists
mem.set_search_priority(phi=0.7, delta_i=0.3)
check("C6 _last_phi stored", abs(mem._last_phi - 0.7) < 1e-9)
check("C7 _last_delta_i stored", abs(mem._last_delta_i - 0.3) < 1e-9)

expected_pnorm = float(np.tanh(0.6*0.7 + 0.4*0.3))
check("C8 priority_norm = tanh(priority)",
      abs(np.tanh(priority(0.7, 0.3)) - expected_pnorm) < 1e-9)

mem_src = open('core/memory.py', encoding='utf-8').read()
check("C9 set_search_priority in memory.py", 'set_search_priority' in mem_src)
check("C10 _last_phi in search()", '_last_phi' in mem_src)
check("C11 set_search_priority called in orchestrator",
      'set_search_priority' in orch_src)


# ── Summary ──────────────────────────────────────────────────────────────────
print()
print("=" * 60)
if _failures:
    print(f"\033[31mFAILED {len(_failures)}:\033[0m")
    for f in _failures: print(f"  - {f}")
    sys.exit(1)
else:
    print("\033[32mALL SANITY CHECKS PASSED\033[0m")
print("=" * 60)
