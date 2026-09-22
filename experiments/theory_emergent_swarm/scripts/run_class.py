"""run_class.py — «класс роя»: где 0.726 лежит на кривой размер->качество,
измеренной ОДНИМ И ТЕМ ЖЕ стендом (тот же промпт, та же экстракция, те же
200 задач GSM8K, temp=0).

Сравнивать свои числа с опубликованными нельзя: протоколы разные (0-shot
против 8-shot CoT, maj@k, свои промпты). Поэтому кривая строится только
на локальных моделях. Phi-3.5-mini (3.8B) НЕ грузится (квантизация
Q4_0_4_8 удалена из формата), поэтому верхний конец кривой -- 2B, и
локализация 0.726 будет ЭКСТРАПОЛЯЦИЕЙ за пределы замеренного диапазона.
Это указывается в отчёте явно.

Добавляет две модели ниже 1B (0.36B, 0.5B) к шести уже замеренным.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import gsm8k_data as GD          # noqa: E402
import model_registry_11 as MR   # noqa: E402
import call_log                  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
METRICS = Path(__file__).resolve().parents[1] / "metrics"

# package-local extension of the registry -- experiment11/ is never edited
EXTRA = {
    "smollm2-360m-instruct-q5_k_m": {
        "path": "bartowskiSmolLM2-360M-Instruct-GGUF/SmolLM2-360M-Instruct-Q5_K_M.gguf",
        "params_b": 0.36, "stop": ["<|im_end|>"],
    },
    "qwen2.5-0.5b-instruct-q5_0": {
        "path": "Qwen2.5-0.5B-Instruct/qwen2.5-0.5b-instruct-q5_0.gguf",
        "params_b": 0.49, "stop": ["<|im_end|>"],
    },
}

PARAMS_B = {            # published parameter counts of the registry models
    "llama-3.2-1b-instruct-q4_0": 1.24,
    "qwen2.5-coder-1.5b-instruct-q4_0": 1.54,
    "qwen3-1.7b-q4_0-unsloth": 1.72,
    "internvl3-2b-q4_k_m": 2.00,
    "gemma-3-it-1b-q5_k_s": 1.00,
    "smollm2-1.7b-instruct-q4_k_m": 1.71,
}

N_TASKS = 200
MAX_TOKENS = 320
SEED = 4242


def load_extra(model_id: str):
    sys.path.insert(0, str(ROOT / "experiment9" / "configs"))
    from env_fix import fix_cuda_dll_path
    fix_cuda_dll_path()
    from llama_cpp import Llama
    n = os.cpu_count() or 4
    return Llama(model_path=str(ROOT / "models" / EXTRA[model_id]["path"]),
                 n_gpu_layers=-1, n_ctx=2048, verbose=False,
                 n_threads=max(n - 1, 1), n_threads_batch=n)


def gen_extra(llm, model_id: str, prompt: str) -> str:
    import chat_render
    rendered = chat_render.render_prompt(llm, model_id, [{"role": "user", "content": prompt}],
                                         enable_thinking=None)
    out = llm.create_completion(prompt=rendered, max_tokens=MAX_TOKENS, temperature=0.0,
                                top_p=1.0, top_k=40, seed=SEED, stop=EXTRA[model_id]["stop"])
    return out["choices"][0]["text"]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    tasks = GD.load(N_TASKS)
    print(f"class curve: {len(tasks)} GSM8K problems, same harness as G1\n", flush=True)

    scores = {}
    for model_id in EXTRA:
        llm = load_extra(model_id)
        t0 = time.time()
        hit = 0
        for t in tasks:
            try:
                txt = gen_extra(llm, model_id, GD.build_solve_prompt(t))
            except Exception:
                txt = ""
            a = GD.extract_answer(txt)
            hit += int(GD.same(a, t["gold"]))
            call_log.log_filter_call(model=model_id, task_id=t["task_id"], family="CLASS-solve",
                                     seed=SEED, n_in=0, n_out=0, ms=0.0, failed=False,
                                     extracted=[a], raw_text=txt)
        scores[model_id] = hit / len(tasks)
        print(f"  {model_id:35s} {EXTRA[model_id]['params_b']:.2f}B  "
             f"acc={scores[model_id]:.3f}  ({time.time()-t0:.0f}s)", flush=True)
        del llm

    # merge with G1's already-measured six
    g1 = json.loads((METRICS / "g1_phase1.json").read_text(encoding="utf-8"))["per_model"]
    curve = [(EXTRA[m]["params_b"], scores[m], m) for m in EXTRA]
    curve += [(PARAMS_B[m], g1[m], m) for m in g1]
    curve.sort()

    print("\n--- measured scaling curve (same harness, same 200 problems) ---")
    for p, a, m in curve:
        print(f"  {p:5.2f}B  {a:.3f}  {m}")

    # log-linear fit + R^2, so a bad fit is visible rather than hidden
    import math
    xs = [math.log10(p) for p, _a, _m in curve]
    ys = [a for _p, a, _m in curve]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sxy / sxx if sxx else 0.0
    icept = my - slope * mx
    ss_res = sum((y - (slope * x + icept)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1 - ss_res / ss_tot if ss_tot else 0.0

    swarm = 0.726
    implied = 10 ** ((swarm - icept) / slope) if slope else float("nan")
    print(f"\n  log-linear fit: acc = {slope:.3f}*log10(params_B) + {icept:.3f}   R^2 = {r2:.3f}")
    print(f"  swarm = {swarm:.3f}  ->  implied single-model size = {implied:.1f}B")
    print(f"  {'FIT IS WEAK -- treat the implied size as unreliable' if r2 < 0.5 else 'fit usable'}")

    (METRICS / "class_curve.json").write_text(json.dumps({
        "n_tasks": len(tasks), "curve": [{"params_b": p, "acc": a, "model": m} for p, a, m in curve],
        "slope": slope, "intercept": icept, "r2": r2,
        "swarm_gsm8k": swarm, "implied_params_b": implied,
        "fit_usable": r2 >= 0.5,
        "note": "extrapolation beyond 2B; Phi-3.5-mini 3.8B could not be loaded (Q4_0_4_8 removed)",
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
