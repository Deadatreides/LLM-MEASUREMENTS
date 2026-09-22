"""
Thin wrapper around llama-cpp-python for Experiment 2, reusing the same
GPU-DLL fix and chat-template rendering approach validated in Experiment 1
(configs/env_fix.py, configs/chat_render.py copied verbatim from
experiment/configs/).
"""
import json
import os
import time

from env_fix import fix_cuda_dll_path

fix_cuda_dll_path()
from llama_cpp import Llama  # noqa: E402
from chat_render import render_prompt, STOP_SEQUENCES  # noqa: E402

MODELS_BASE = r"<PROJECT_ROOT>\trace-probe\models"
CONFIGS_DIR = os.path.dirname(__file__)
N_CTX = 4096  # pipeline prompts carry more context (contract + prior artifacts) than experiment 1


def load_registry():
    with open(os.path.join(CONFIGS_DIR, "models.json"), encoding="utf-8") as f:
        return json.load(f)


def get_model_entry(model_id):
    reg = load_registry()
    entry = next((m for m in reg["models"] if m["model_id"] == model_id), None)
    if entry is None:
        raise SystemExit(f"unknown model_id {model_id}")
    return entry


def load_model(model_id):
    entry = get_model_entry(model_id)
    path = os.path.join(MODELS_BASE, entry["path"])
    t0 = time.time()
    llm = Llama(model_path=path, n_gpu_layers=-1, n_ctx=N_CTX, verbose=False)
    load_time_sec = time.time() - t0
    return llm, entry, load_time_sec


def generate(llm, model_id, model_entry, user_content, temperature=0.0, top_p=1.0, top_k=40, seed=1, max_tokens=700):
    """One role/baseline call: renders the chat prompt via the model's own
    template (or fallback) and runs a single completion. Returns a dict
    with raw_text, usage, timing, and any generation error (kept separate
    from solution-quality signals downstream)."""
    enable_thinking = False if model_entry["family"] == "qwen3" else None
    stop = STOP_SEQUENCES.get(model_id)

    try:
        rendered_prompt = render_prompt(
            llm, model_id, [{"role": "user", "content": user_content}], enable_thinking=enable_thinking
        )
    except Exception as e:
        return {
            "raw_text": "",
            "rendered_prompt": None,
            "generation_failed": True,
            "generation_error": f"{type(e).__name__}: {e}",
            "input_tokens": None,
            "output_tokens": None,
            "generation_time_sec": 0.0,
        }

    t0 = time.time()
    try:
        out = llm.create_completion(
            prompt=rendered_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            stop=stop,
        )
        gen_time = time.time() - t0
        usage = out.get("usage", {}) or {}
        return {
            "raw_text": out["choices"][0]["text"],
            "rendered_prompt": rendered_prompt,
            "generation_failed": False,
            "generation_error": None,
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "generation_time_sec": gen_time,
        }
    except Exception as e:
        return {
            "raw_text": "",
            "rendered_prompt": rendered_prompt,
            "generation_failed": True,
            "generation_error": f"{type(e).__name__}: {e}",
            "input_tokens": None,
            "output_tokens": None,
            "generation_time_sec": time.time() - t0,
        }
