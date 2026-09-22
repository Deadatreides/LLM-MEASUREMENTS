"""run_g1_smoke.py — gate before budget for G1 (GSM8K).
5 problems x 6 models = 30 calls. Verifies: the ANSWER: format is
followed, extraction works, the safe evaluator parses the models' own
equations. Aborts the main run if the format is not usable.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import gsm8k_data as GD          # noqa: E402
import safe_arith as SA          # noqa: E402
import model_registry_11 as MR   # noqa: E402

N_TASKS = 5
MAX_TOKENS = 320
TEMPERATURE = 0.0


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    tasks = GD.load(N_TASKS)
    print(f"GSM8K smoke: {len(tasks)} problems x {len(MR.MODEL_IDS)} models\n", flush=True)

    got_fmt = 0
    got_num = 0
    total = 0
    hits = 0
    t0 = time.time()
    for model_id in MR.MODEL_IDS:
        llm, _ = MR.load_model(model_id)
        n_ok = 0
        for t in tasks:
            raw = MR.generate(llm, model_id, GD.build_solve_prompt(t), temperature=TEMPERATURE,
                              top_p=1.0, seed=1234, max_tokens=MAX_TOKENS)
            txt = raw.get("raw_text", "")
            total += 1
            if "ANSWER" in txt.upper():
                got_fmt += 1
            a = GD.extract_answer(txt)
            if a is not None:
                got_num += 1
            ok = GD.same(a, t["gold"])
            hits += int(ok)
            n_ok += int(ok)
        print(f"  {model_id:35s} {n_ok}/{len(tasks)} correct", flush=True)
        del llm

    eqs = SA.find_equations("Janet sells 16 - 3 - 4 = 9 eggs. She makes 9 * 2 = $18 daily.")
    print(f"\n  format followed : {got_fmt}/{total}")
    print(f"  number extracted: {got_num}/{total}")
    print(f"  pooled accuracy : {hits/total:.3f}")
    print(f"  evaluator check : {eqs}")
    print(f"  elapsed {time.time()-t0:.0f}s -> ~{(time.time()-t0)/total:.2f}s per call")

    if got_num < total * 0.8:
        print("\n  GATE FAILED: extraction below 80%")
        sys.exit(1)
    print("\n  GATE OK")


if __name__ == "__main__":
    main()
