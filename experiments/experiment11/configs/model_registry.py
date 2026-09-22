"""model_registry.py — 6 моделей эксперимента 11.

experiment9/configs/models.json заморожен под протокол experiment9 (3
модели). Эксперимент 11 использует 6, включая 3 не зарегистрированных
там, поэтому здесь собственный, независимый реестр — не правка чужого
замороженного протокола.

Переиспользуется только НИЗКОУРОВНЕВЫЙ стенд: env_fix.fix_cuda_dll_path,
chat_render.render_prompt, llama_cpp.Llama — тот же слой, которым
пользуется llm_client.py, но с собственным путём/стоп-словарём (та же
степень переиспользования, что у model_adapter.py на arch1, без
Mycelium-обёртки поверх).

Все 6 моделей проверены реальной загрузкой + одной генерацией ДО
включения в реестр. `Phi-3.5-mini-instruct-Q4_0_4_8` исключена:
квантизация `TYPE_Q4_0_4_8` удалена из формата, который читает
llama-cpp-python 0.3.34 ("REMOVED, use Q4_0 with runtime repacking") —
не грузится ни на GPU, ни на CPU. Починка требует перекачать другой
квант той же модели — внешняя загрузка требует разрешения пользователя
и не входит в этот эксперимент.

`internvl3-2b-q4_k_m`: на диске лежит только текстовый бэкбон (файла
mmproj/визуального проектора нигде в дереве models/ нет) — мультимодальность
физически недоступна, модель используется как обычная текстовая.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Callable, Optional

MODELS_BASE = Path(__file__).resolve().parents[2] / "models"
_EXPERIMENT9_CONFIGS = Path(__file__).resolve().parents[2] / "experiment9" / "configs"

N_CTX = 2048  # короткие retry-промпты и короткие ответы -- запас достаточен


def _stand():
    """Ленивый импорт низкоуровневого стенда. Не Mycelium, не arch1."""
    path_str = str(_EXPERIMENT9_CONFIGS)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    from env_fix import fix_cuda_dll_path

    fix_cuda_dll_path()
    from llama_cpp import Llama
    import chat_render

    return Llama, chat_render


MODEL_REGISTRY = {
    "llama-3.2-1b-instruct-q4_0": {
        "path": "Llama-3.2-1B-Instruct-Q4_0/Llama-3.2-1B-Instruct-Q4_0.gguf",
        "family": "llama-3.2",
        "size_label": "1B",
    },
    "qwen2.5-coder-1.5b-instruct-q4_0": {
        "path": "qwen2.5-coder-1.5b-instruct-q4_0/qwen2.5-coder-1.5b-instruct-q4_0.gguf",
        "family": "qwen2.5-coder",
        "size_label": "1.5B",
    },
    "qwen3-1.7b-q4_0-unsloth": {
        "path": "unslothQwen3-1.7B-GGUF/Qwen3-1.7B-Q4_0.gguf",
        "family": "qwen3",
        "size_label": "1.7B",
    },
    "internvl3-2b-q4_k_m": {
        "path": "InternVL3-2B-Q4_K_M/InternVL3-2B-Q4_K_M.gguf",
        "family": "internvl3",
        "size_label": "2B",
    },
    "gemma-3-it-1b-q5_k_s": {
        "path": "gemma-3-it-1B-Q5_K_S/gemma-3-it-1B-Q5_K_S.gguf",
        "family": "gemma-3",
        "size_label": "1B",
    },
    "smollm2-1.7b-instruct-q4_k_m": {
        "path": "SmolLM2-1.7B-Instruct-Q4_K_M/SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
        "family": "smollm2",
        "size_label": "1.7B",
    },
}

MODEL_IDS = tuple(MODEL_REGISTRY)

# Стандартные end-of-turn маркеры по документированной конвенции шаблона
# семейства -- зафиксированы здесь, до прогона, не подобраны по итогам.
_STOP_SEQUENCES = {
    "llama-3.2-1b-instruct-q4_0": ["<|eot_id|>"],
    "qwen2.5-coder-1.5b-instruct-q4_0": ["<|im_end|>"],
    "qwen3-1.7b-q4_0-unsloth": ["<|im_end|>"],
    "internvl3-2b-q4_k_m": ["<|im_end|>"],  # Qwen2-based backbone, ChatML
    "gemma-3-it-1b-q5_k_s": ["<end_of_turn>"],
    "smollm2-1.7b-instruct-q4_k_m": ["<|im_end|>"],
}


def load_model(model_id: str):
    """-> (llm, load_time_sec).

    `n_threads`/`n_threads_batch` заданы явно (раньше — библиотечные
    умолчания, `cpu_count()//2`/`cpu_count()`) -- на 12-ядерной машине
    умолчание использовало только 6 потоков на генерацию; здесь -- явный
    запас под ОС (cpu_count()-1) на генерацию, все ядра на batch-обработку
    промпта. Не меняет алгоритм/семплирование, только скорость CPU-части
    (проверка/детокенизация), безопасно для уже собранных данных
    Эксп.11/12 (если их когда-нибудь потребуется повторить)."""
    Llama, _chat_render = _stand()
    entry = MODEL_REGISTRY[model_id]
    path = MODELS_BASE / entry["path"]
    n_cpu = os.cpu_count() or 4
    t0 = time.time()
    llm = Llama(
        model_path=str(path), n_gpu_layers=-1, n_ctx=N_CTX, verbose=False,
        n_threads=max(n_cpu - 1, 1), n_threads_batch=n_cpu,
    )
    return llm, time.time() - t0


def generate(
    llm,
    model_id: str,
    user_content: str,
    *,
    temperature: float = 0.5,
    top_p: float = 1.0,
    top_k: int = 40,
    seed: int = 1,
    max_tokens: int = 300,
    stop_predicate: Optional[Callable[[str], bool]] = None,
) -> dict:
    """Возвращает СЫРЬЁ (не парсит) -- та же форма результата, что
    llm_client.generate(), но со своим стоп-словарём: models.json
    experiment9 не знает про наши 6 моделей.

    `stop_predicate` (опционально, по умолчанию None -- поведение ПОЛНОСТЬЮ
    как раньше, для обратной совместимости с уже собранными данными
    Эксп.11/12): если задан, генерация идёт в потоковом режиме
    (`stream=True`) и ПРЕКРАЩАЕТСЯ, как только `stop_predicate(накопленный_текст)`
    становится True -- ДО исчерпания `max_tokens`.

    Зачем: `llama-cpp-python` внутри `_create_completion` на КАЖДОМ новом
    токене передетокенизирует ВЕСЬ накопленный список токенов заново
    (`self.detokenize(completion_tokens, prev_tokens=prompt_tokens)`,
    см. llama_cpp/llama.py) -- стоимость генерации одного ответа растёт
    квадратично от его длины. Для коротких ответов (Эксп.11/12, ~50-150
    токенов) это незаметно; для длинных структурированных ответов
    (Эксп.13, до 400-500 токенов, часто без чистой ранней остановки
    самой моделью) это стало доминирующей, CPU-bound стоимостью --
    найдено и подтверждено на реальном прогоне (см. STATUS.md/отчёт
    эксперимента 13). Уменьшать `max_tokens` (бюджет) не требуется --
    ранняя остановка просто прекращает вытягивание генератора, как
    только механически ясно, что ответ уже получен, что останавливает
    и лишнюю генерацию, и лишнюю передетокенизацию её хвоста."""
    _Llama, chat_render = _stand()
    family = MODEL_REGISTRY[model_id]["family"]
    enable_thinking = False if family == "qwen3" else None  # согласовано с experiment1-6
    stop = _STOP_SEQUENCES.get(model_id)

    try:
        rendered_prompt = chat_render.render_prompt(
            llm, model_id, [{"role": "user", "content": user_content}], enable_thinking=enable_thinking
        )
    except Exception as exc:
        return {
            "raw_text": "",
            "rendered_prompt": None,
            "generation_failed": True,
            "generation_error": f"{type(exc).__name__}: {exc}",
            "input_tokens": None,
            "output_tokens": None,
            "generation_time_sec": 0.0,
        }

    if stop_predicate is None:
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
                "stopped_early": False,
            }
        except Exception as exc:
            return {
                "raw_text": "",
                "rendered_prompt": rendered_prompt,
                "generation_failed": True,
                "generation_error": f"{type(exc).__name__}: {exc}",
                "input_tokens": None,
                "output_tokens": None,
                "generation_time_sec": time.time() - t0,
            }

    t0 = time.time()
    pieces: list = []
    try:
        stream = llm.create_completion(
            prompt=rendered_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            stop=stop,
            stream=True,
        )
        stopped_early = False
        for chunk in stream:
            piece = chunk["choices"][0].get("text", "")
            pieces.append(piece)
            if stop_predicate("".join(pieces)):
                stopped_early = True
                break
        gen_time = time.time() - t0
        raw_text = "".join(pieces)
        try:
            prompt_tokens = len(llm.tokenize(rendered_prompt.encode("utf-8")))
        except Exception:
            prompt_tokens = None
        return {
            "raw_text": raw_text,
            "rendered_prompt": rendered_prompt,
            "generation_failed": False,
            "generation_error": None,
            "input_tokens": prompt_tokens,
            "output_tokens": len(pieces),  # приближение: почти всегда 1 chunk = 1 токен, точный usage в потоковом режиме библиотека не даёт
            "generation_time_sec": gen_time,
            "stopped_early": stopped_early,
        }
    except Exception as exc:
        return {
            "raw_text": "".join(pieces),
            "rendered_prompt": rendered_prompt,
            "generation_failed": True,
            "generation_error": f"{type(exc).__name__}: {exc}",
            "input_tokens": None,
            "output_tokens": None,
            "generation_time_sec": time.time() - t0,
        }
