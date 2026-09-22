"""End-to-end rehearsal of the MoE path, without the real weights.

The dense half of the probe has been run; the MoE half has not, so every MoE
branch (capture -> trace file -> simulator -> §7 gap -> report) is exercised
here on a granitemoe of the *real* routing geometry (24 layers, 32 experts,
top-8) with random weights and the Qwen3 tokenizer that is already on disk.

Catches shape and plumbing errors before the 2.67 GB download is spent.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, GraniteMoeConfig, GraniteMoeForCausalLM

ROOT = Path(__file__).parent
TAG = "GSMOKE"
DIR = ROOT / "models" / "granite-moe-smoke"


def build():
    tok_src = ROOT / "models" / "Qwen3-1.7B"
    if not (tok_src / "tokenizer.json").exists():
        raise SystemExit(f"need a tokenizer; {tok_src} not found")
    tok = AutoTokenizer.from_pretrained(tok_src)
    cfg = GraniteMoeConfig(
        vocab_size=len(tok),          # reuse the tokenizer already on disk
        hidden_size=128, intermediate_size=256,
        num_hidden_layers=24,         # real granite-3.1-1b-a400m geometry
        num_attention_heads=8, num_key_value_heads=4,
        num_local_experts=32, num_experts_per_tok=8,
        max_position_embeddings=4096,
    )
    DIR.mkdir(parents=True, exist_ok=True)
    GraniteMoeForCausalLM(cfg).half().save_pretrained(DIR)
    tok.save_pretrained(DIR)
    print(f"built {DIR} : 24 layers, 32 experts, top-8 "
          f"-> {24*32} units, {24*8} active per token")


def run(*args):
    print(f"\n$ {' '.join(str(a) for a in args)}", flush=True)
    r = subprocess.run([sys.executable, *[str(a) for a in args]])
    if r.returncode != 0:
        raise SystemExit(f"failed: {args}")


if __name__ == "__main__":
    build()
    trace = ROOT / "traces" / f"{TAG}.npz"
    for p in (trace, trace.with_suffix(".seq.json")):
        p.unlink(missing_ok=True)
    shutil.rmtree(ROOT / "traces" / f"{TAG}.parts", ignore_errors=True)

    run(ROOT / "probe.py", TAG, "--introspect")
    run(ROOT / "probe.py", TAG, "--limit", 12, "--batch", 4, "--checkpoint", 6)
    run(ROOT / "sim.py", trace, ROOT / "out" / f"{TAG}.analysis.json")
    run(ROOT / "probe.py", f"{TAG}2", "--limit", 12, "--batch", 4,
        "--replay", trace.with_suffix(".seq.json"),
        "--out", ROOT / "traces" / f"{TAG}2.npz")
    run(ROOT / "sim.py", ROOT / "traces" / f"{TAG}2.npz",
        ROOT / "out" / f"{TAG}2.analysis.json")
    run(ROOT / "compare.py", TAG, f"{TAG}2")
    run(ROOT / "report.py", TAG, f"{TAG}2")

    res = json.loads((ROOT / "out" / f"{TAG}.analysis.json").read_text("utf-8"))
    print("\n=== MoE path exercised end to end ===")
    print(f"  units {res['trace']['units_total']}, active/token "
          f"{res['trace']['mean_active_per_token']}, "
          f"experts {res['trace']['experts']}, top_k {res['trace']['top_k']}")
    print(f"  §7 gap section present: {'gap' in res} "
          f"({len(res.get('gap', {}))} layers)")
    print(f"  iso-ratio M = {res['iso_ratio_point']['M_percent']} % "
          f"(clamped: {res['iso_ratio_point']['clamped']})")
    print("  NOTE: weights are random, so the numbers are meaningless; only the "
          "shapes and the plumbing are being checked.")
