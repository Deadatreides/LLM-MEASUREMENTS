"""
lab/gguf_boot.py — надёжный импорт llama_cpp на этой машине.

Импортировать ДО llama_cpp:

    from lab.gguf_boot import llama_cpp, Llama      # или
    from lab.gguf_boot import ensure_llama_cpp; llama_cpp = ensure_llama_cpp()

Две вещи ломают `import llama_cpp` в этом окружении, и обе не в колесе.

1. CUDA_PATH указывает в никуда.
   На машине задана МАШИННАЯ переменная CUDA_PATH = 'E:\\софт\\cuda' (и
   CUDA_PATH_V12_8 туда же) — остаток от снесённой CUDA 12.8, каталога нет.
   llama_cpp/_ctypes_extensions.py:64 делает
       os.add_dll_directory(os.path.join(os.environ["CUDA_PATH"], "bin"))
   безусловно, без проверки существования, и падает с
       FileNotFoundError: [WinError 3] ... 'E:\\софт\\cuda'
   Снимаем переменную ТОЛЬКО в текущем процессе. Машинные переменные не
   трогаем: это системная настройка, и её правка — отдельное решение
   пользователя, а не побочный эффект запуска прогона.

2. Колесо не везёт CUDA-рантайм.
   В llama_cpp/lib лежат только ggml-*.dll (ggml-cuda.dll на 902 МБ) и
   llama.dll. cudart64_12.dll / cublas64_12.dll / cublasLt64_12.dll в нём
   нет — предполагается, что они найдутся через CUDA_PATH\\bin, которого
   здесь нет. Но они уже стоят вместе с torch:
       site-packages/torch/lib/{cudart64_12,cublas64_12,cublasLt64_12}.dll
   Добавляем этот каталог в поиск DLL. cu124-колесо и cu126-torch совместимы:
   мажорная версия SONAME одна (12), CUDA гарантирует minor-совместимость.

Проверено: GPU offload = True, GTX 1660 SUPER (compute 7.5) определяется.
Предупреждение llama.cpp про отсутствие tensor cores для Turing ожидаемо и
на работоспособность не влияет.
"""

from __future__ import annotations

import os
import pathlib

_ready = False


def _strip_broken_cuda_path() -> list:
    """Снять CUDA_* переменные, указывающие на несуществующие каталоги."""
    removed = []
    for var in ('CUDA_PATH', 'CUDA_PATH_V12_8', 'CUDA_HOME'):
        val = os.environ.get(var)
        if val and not os.path.isdir(os.path.join(val, 'bin')):
            os.environ.pop(var, None)
            removed.append(f"{var}={val}")
    return removed


def _add_cuda_runtime_dirs() -> list:
    """Дать загрузчику найти cudart/cublas. Источники по убыванию надёжности."""
    added = []
    candidates = []

    try:
        import torch
        candidates.append(pathlib.Path(torch.__file__).parent / 'lib')
    except Exception:
        pass

    # Пакеты nvidia-*-cu12, если они когда-нибудь появятся
    try:
        import site
        for sp in site.getsitepackages():
            nv = pathlib.Path(sp) / 'nvidia'
            if nv.is_dir():
                candidates.extend(p for p in nv.glob('*/bin') if p.is_dir())
    except Exception:
        pass

    # Настоящая установка CUDA, если она есть
    real = os.environ.get('CUDA_PATH')
    if real and os.path.isdir(os.path.join(real, 'bin')):
        candidates.append(pathlib.Path(real) / 'bin')

    for d in candidates:
        try:
            if d.is_dir() and any(d.glob('cudart64_*.dll')):
                os.add_dll_directory(str(d))
                added.append(str(d))
        except Exception:
            continue
    return added


def ensure_llama_cpp(verbose: bool = False):
    """Подготовить окружение и вернуть импортированный модуль llama_cpp."""
    global _ready
    if not _ready:
        removed = _strip_broken_cuda_path()
        added = _add_cuda_runtime_dirs()
        if verbose:
            for r in removed:
                print(f"[gguf_boot] снята битая переменная {r}")
            for a in added:
                print(f"[gguf_boot] CUDA-рантайм: {a}")
        _ready = True
    import llama_cpp  # noqa: E402
    return llama_cpp


def diagnose() -> dict:
    """Полная диагностика — то, что надо приложить к багрепорту."""
    info = {}
    info['cuda_path_env'] = os.environ.get('CUDA_PATH')
    lc = ensure_llama_cpp(verbose=True)
    info['llama_cpp_version'] = getattr(lc, '__version__', '?')
    info['gpu_offload'] = bool(lc.llama_supports_gpu_offload())
    info['mmap'] = bool(lc.llama_supports_mmap())
    try:
        import torch
        info['torch'] = torch.__version__
        info['torch_cuda'] = torch.version.cuda
        info['gpu'] = torch.cuda.get_device_name(0)
        info['capability'] = torch.cuda.get_device_capability(0)
        free, total = torch.cuda.mem_get_info()
        info['vram_free_mib'] = round(free / 2 ** 20)
        info['vram_total_mib'] = round(total / 2 ** 20)
    except Exception as exc:
        info['torch'] = f'недоступен: {exc}'
    return info


if __name__ == '__main__':
    import json
    d = diagnose()
    print(json.dumps(d, ensure_ascii=False, indent=2))
    print('\nИТОГ:', 'GGUF на GPU готов' if d.get('gpu_offload')
          else 'GPU offload НЕ доступен — GGUF пойдёт на CPU')
