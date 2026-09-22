"""
Fixes GPU (CUDA) DLL loading for llama-cpp-python on this machine.

Root cause: the system env var CUDA_PATH points to a stale/incomplete
directory (no `bin` folder), which makes llama_cpp's loader crash before
it even tries to load the actual shared library. Separately, the CUDA
toolkit install under Program Files is incomplete (missing bin/ entirely).

Fix (process-local only, does not touch system/env settings):
  - drop CUDA_PATH / CUDA_PATH_V12_8 from this process' environment so
    llama_cpp's loader doesn't try to add a nonexistent directory
  - prepend known-good CUDA 12 runtime DLL directories to PATH:
      * LM Studio's bundled cuBLAS/cuBLASLt/cudart vendor dir
      * torch's bundled nvrtc64_120_0.dll (cuBLASLt depends on it)
  - ctypes.CDLL on this build uses winmode=RTLD_GLOBAL==0 on Windows,
    which triggers the legacy DLL search order that honors PATH, so this
    is sufficient without any system-wide changes.

Call fix_cuda_dll_path() before `import llama_cpp`.
"""
import os

_LMSTUDIO_CUDA12_VENDOR = (
    r"<HOME>\.lmstudio\extensions\backends\vendor\win-llama-cuda12-vendor-v2"
)
_TORCH_LIB = (
    r"<HOME>\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\lib"
)

_applied = False


def fix_cuda_dll_path():
    global _applied
    if _applied:
        return
    os.environ.pop("CUDA_PATH", None)
    os.environ.pop("CUDA_PATH_V12_8", None)
    os.environ.pop("CUDA_PATH_V12_6", None)
    extra = [p for p in (_LMSTUDIO_CUDA12_VENDOR, _TORCH_LIB) if os.path.isdir(p)]
    os.environ["PATH"] = os.pathsep.join(extra) + os.pathsep + os.environ.get("PATH", "")
    _applied = True
