"""
smoke_check.py - Mycelium operational dependency and health sanity checker.
"""
import sys
import os

print("=" * 60)
print("MYCELIUM DEVIATION & HARDENING SMOKE CHECKER")
print("=" * 60)

# Check Python Version
print(f"Python Version: {sys.version}")
py_major, py_minor = sys.version_info[:2]
if py_major < 3 or (py_major == 3 and py_minor < 10):
    print(f"[ERROR] Python 3.10+ is required. Found: {py_major}.{py_minor}")
    sys.exit(1)
print("[✓] Python version is compatible (3.10+)")

# Check Required Core Dependencies
deps = [
    ("faiss", "faiss-cpu"),
    ("sentence_transformers", "sentence-transformers"),
    ("aiohttp", "aiohttp"),
    ("yaml", "pyyaml"),
    ("schedule", "schedule"),
    ("numpy", "numpy"),
    ("networkx", "networkx"),
    ("sympy", "sympy"),
    ("tqdm", "tqdm"),
    ("pytest", "pytest"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
]

print("\n--- Core Dependencies ---")
core_ok = True
for module_name, pip_name in deps:
    try:
        __import__(module_name)
        print(f"[✓] {module_name} ({pip_name}) is installed.")
    except ImportError as e:
        print(f"[✗] {module_name} ({pip_name}) is MISSING! Error: {e}")
        core_ok = False

# Check Optional Nightly DPO Stack
print("\n--- Nightly DPO Stack (Optional) ---")
dpo_deps = ["torch", "transformers", "peft", "trl", "accelerate", "datasets"]
dpo_installed_count = 0
for module_name in dpo_deps:
    try:
        mod = __import__(module_name)
        version = getattr(mod, "__version__", "unknown")
        print(f"[✓] {module_name} is installed (Version: {version}).")
        dpo_installed_count += 1
    except ImportError:
        print(f"[!] {module_name} is not installed (DPO mode will be bypassed).")

if dpo_installed_count == len(dpo_deps):
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        cuda_ver = torch.version.cuda if cuda_avail else "N/A"
        print(f"[✓] PyTorch CUDA available: {cuda_avail} (CUDA Version: {cuda_ver})")
    except Exception as e:
        print(f"[!] Failed to check CUDA: {e}")
else:
    print("[!] DPO training dependencies are not fully installed. Nightly DPO will be disabled/bypassed.")

# Environment File Check
print("\n--- Project Environment ---")
if os.path.exists(".env"):
    print("[✓] .env file is present.")
else:
    print("[!] .env file is missing. Please copy .env.example to .env and configure keys.")

print("=" * 60)
if not core_ok:
    print("[ERROR] Core dependencies are missing. Hardening failed.")
    sys.exit(1)
else:
    print("[SUCCESS] All core dependencies are successfully validated.")
    sys.exit(0)
