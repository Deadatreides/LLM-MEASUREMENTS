# SETUP — Swarm Installation & Hardening Guide

This document provides step-by-step instructions for installing, configuring, and starting Mycelium Sigma v8.9.2 on Windows and Linux.

---

## Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|---------------|
| GPU | GTX 1060 6GB | GTX 1660 Super 6GB |
| CPU | 4 cores / 3.0GHz | Ryzen 5 1600 (6c/12t) |
| RAM | 16GB DDR4 | 24GB DDR4 |
| SSD | 50GB free | 100GB+ SATA or NVMe |
| RAM-Disk | Optional | 4GB (speedups FAISS searches) |

### Software Stack

- **Python 3.10+** (Required)
- **Git** (For evolution commits)
- **LM Studio** (For the local Critic-model)
- **CUDA Toolkit 12.1+** (Optional, only for nightly DPO training)

---

## Quick Start on Windows

We have provided a comprehensive automated installation script `install.bat` that performs all necessary steps:

1. Double-click `install.bat` or run it from a cmd prompt.
2. The script will:
   - Check your Python version.
   - Create a Python virtual environment `.venv` inside the project folder.
   - Upgrade pip and install all core pinned dependencies from `requirements.txt`.
   - Setup a default `.env` file from the `.env.example` template.
   - Automatically initialize all database and log folders (`data/`, `logs/`, `swarm/tasks/`, `swarm/context/`, `swarm/evolution/`).
   - Run the custom `smoke_check.py` script.

---

## Manual Python Environment Setup

If you prefer to perform the installation manually, follow these instructions:

### 1. Create and Activate Virtual Environment
```bash
# Create .venv
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate.bat

# Activate (Linux/macOS)
source .venv/bin/activate
```

### 2. Install Pinned Core Dependencies
```bash
pip install -r requirements.txt
```

This will install the following securely pinned packages:
- `aiohttp==3.9.5`
- `faiss-cpu==1.8.0`
- `numpy==1.26.4`
- `python-dotenv==1.0.1`
- `pyyaml==6.0.2`
- `fastapi==0.111.0` (OpenCode HTTP bridge)
- `uvicorn==0.30.1` (OpenCode HTTP bridge)

### 3. Install Optional Nightly DPO Stack
If you have a GPU with CUDA 12.1, you can install the training stack:
```bash
# 1. Install PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Install TRL stack under pinned version constraints
pip install "trl>=0.8" "peft>=0.10" "transformers>=4.40,<4.43" "accelerate>=0.28" "bitsandbytes>=0.43" "datasets>=2.18"
```

### 4. Verify Installation
```bash
python smoke_check.py
```

---

## OpenCode HTTP Bridge Setup

To launch the local FastAPI bridge and status webpage:
```bash
# Double click the VBScript launcher in the project root:
start_opencode.vbs
```
This VBScript automatically starts the FastAPI uvicorn server on `http://127.0.0.1:8000` and opens the status page in your default browser.

---

## Troubleshooting Windows Issues

### 1. HF Hub Symlinks Warning
When downloading models, Hugging Face Hub may emit warnings about missing symlink permissions on Windows.
- **Solution**: Enable Windows Developer Mode:
  `Settings` -> `Update & Security` -> `For developers` -> Toggle **Developer Mode** on.
  Alternatively, run your terminal/CMD/PowerShell as **Administrator**.

### 2. Unicode / Regional Console Crashes
Windows CMD can crash with encoding exceptions when displaying Unicode symbols like `λ`, `Φ`, `→`, `Δ`.
- **Solution**: The system automatically attempts to force UTF-8 on Windows standard streams at startup. If you still encounter issues in external shells, run the following command before launching Python:
  ```cmd
  chcp 65001
  ```
  This changes the active command page of the terminal to UTF-8.
