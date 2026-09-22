"""Driver for the whole probe, in the order of §10.

Every stage is skipped if its output already exists, so the run can be resumed
after an interruption; probe.py itself also checkpoints inside a trace, so a
killed run continues from the last shard rather than from the start.

  python run_all.py                    # dense pair B / B' end to end
  python run_all.py --only introspect  # step 1 only, nothing downloaded
  python run_all.py --pairs B:B2 A:A2  # add the MoE pair once a model is present

The MoE pair is not in the default list: on this machine bitsandbytes cannot
quantise fused MoE experts, so OLMoE needs ~14 GB and does not fit.  See
README.md for the options.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
PY = sys.executable
TRACES = ROOT / "traces"
OUT = ROOT / "out"

# (reference model, quantised counterpart) -- the second replays the first's tokens
DEFAULT_PAIRS = [("B", "B2")]
NAMED = {"dense": [("B", "B2")], "moe": [("G", "G2")],
         "all": [("B", "B2"), ("G", "G2")]}


def sh(*args, **kw):
    print(f"\n$ {' '.join(str(a) for a in args)}", flush=True)
    t0 = time.time()
    r = subprocess.run([str(a) for a in args], **kw)
    print(f"  ({time.time()-t0:.0f} s, exit {r.returncode})", flush=True)
    if r.returncode != 0:
        raise SystemExit(f"stage failed: {args}")


def trace(tag, replay=None, force=False, batch=8):
    npz = TRACES / f"{tag}.npz"
    if npz.exists() and not force:
        print(f"[skip] trace {tag} exists")
        return npz
    cmd = [PY, ROOT / "probe.py", tag, "--batch", batch]
    if force:
        cmd += ["--no-resume"]
    if replay:
        cmd += ["--replay", replay]
    sh(*cmd)
    return npz


def analyse(tag, force=False):
    js = OUT / f"{tag}.analysis.json"
    if js.exists() and not force:
        print(f"[skip] analysis {tag} exists")
        return js
    sh(PY, ROOT / "sim.py", TRACES / f"{tag}.npz", js)
    return js


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--pairs", nargs="*", default=None,
                    help='"dense", "moe", "all", or explicit REF:QUANT pairs')
    a = ap.parse_args()
    if a.pairs and len(a.pairs) == 1 and a.pairs[0] in NAMED:
        pairs = NAMED[a.pairs[0]]
    elif a.pairs:
        pairs = [tuple(p.split(":")) for p in a.pairs]
    else:
        pairs = DEFAULT_PAIRS
    TRACES.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)

    if a.only in (None, "introspect"):
        # §10 step 1 -- this is the point where the whole probe can be
        # invalidated silently, so it runs first and its output is read.
        # hooktest.py checks the same machinery on tiny random-weight models,
        # which fails fast and without a download if something is wrong.
        sh(PY, ROOT / "hooktest.py")
        sh(PY, ROOT / "archcheck.py")
        for tag in {p[0] for p in pairs}:
            sh(PY, ROOT / "probe.py", tag, "--introspect")
        if a.only == "introspect":
            return

    if not (ROOT / "corpus.json").exists():
        sh(PY, ROOT / "make_corpus.py")

    done = []
    for ref, quant in pairs:
        if a.only not in (None, ref):
            continue
        trace(ref, force=a.force, batch=a.batch)
        analyse(ref, force=a.force)
        seq = TRACES / f"{ref}.seq.json"
        trace(quant, replay=str(seq), force=a.force, batch=a.batch)
        analyse(quant, force=a.force)
        sh(PY, ROOT / "compare.py", ref, quant)
        done += [ref, quant]

    if done:
        sh(PY, ROOT / "report.py", *done)


if __name__ == "__main__":
    main()
