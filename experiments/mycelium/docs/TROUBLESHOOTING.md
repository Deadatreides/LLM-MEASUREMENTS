# TROUBLESHOOTING — swarms and API networks diagnostics

This guide details resolutions for common Windows compatibility issues, provider outages, and path exceptions in Mycelium Sigma v8.9.2.

---

## 1. Windows Compatibility & Paths

### 1.1 Unicode / Encoding Crash in Windows Console
- **Symptom**: Swarm starts up, but as soon as a Cyrillic character or mathematical symbol (`λ`, `Φ`, `→`, `Δ`) is logged, the terminal crashes with a `UnicodeEncodeError`.
- **Cause**: Windows CMD defaults to active regional code pages (e.g. CP1251, CP866) which do not support UTF-8 characters natively.
- **Resolution**:
  - The system automatically triggers `sys.stdout.reconfigure(encoding='utf-8')` on startup.
  - If your terminal does not support this, change your active code page page to UTF-8 manually before running:
    ```cmd
    chcp 65001
    ```
  - We recommend using modern terminals like **Windows Terminal** or VS Code's integrated console instead of legacy CMD.exe.

### 1.2 Unmounted RAM-Disk or HDD Path Crash
- **Symptom**: Configured paths to FAISS vector index, Physarum graph, or SQLite trace databases are located on a drive (e.g. RAM-disk `R:\` or SSD `D:\`) that is unmounted or write-protected.
- **Resolution**:
  - The system has been hardened with a path fallback mechanism inside `core/memory.py`.
  - If a path is invalid or unwriteable, Mycelium prints a warning in `logs/main.log` and automatically redirects database and index files to `./data/` folder in the project folder. No crash will occur.

### 1.3 Hugging Face Hub Symlink Warning
- **Symptom**: Hugging Face sentence-transformers warns about missing symlink permissions on Windows.
- **Cause**: On Windows, normal user accounts are restricted from creating symlinks for safety reasons.
- **Resolution**:
  - Enable **Developer Mode** on Windows (Go to Windows Settings -> Update & Security -> For developers -> Toggle Developer Mode ON).
  - Alternatively, run CMD or PowerShell as **Administrator** once.

---

## 2. API & Provider Outages

### 2.1 Provider Cooldown / Outage Adaptation
- **Symptom**: Log shows `rate_limit` (429) or `fatal` HTTP errors (400, 401, 402, 403, 404) for some endpoints.
- **Hardened Behavior**:
  - Transient errors (429, 5xx, timeout) are automatically retried up to 3 times with exponential backoff before reporting failure.
  - Non-transient errors (invalid credentials, insufficient balance, missing auth keys) are classified as fatal.
  - Fatal errors instantly place the provider on cooldown for 60 seconds (no new requests will be sent to it during cooldown).
  - You can permanently disable unstable or missing endpoints by setting `alive: false` in `config/models.yaml`.

### 2.2 LM Studio Connection Failures
- **Symptom**: `local_value` or `local_router` model logs a connection exception.
- **Cause**: Local server is not running on port 1234 or the GGUF model is not loaded.
- **Resolution**:
  - Open LM Studio, go to the Local Server tab, load your Qwen3-0.8B model, ensure port is `1234`, and check that the server status is green.
