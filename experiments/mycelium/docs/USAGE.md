# USAGE — Swarm Runtime & Modes

This document explains the runtime modes, commands, and integration interfaces for Mycelium Sigma v8.9.2.

---

## 1. Swarm Execution Modes

### 1.1 Interactive REPL Mode
Standard shell command:
```bash
python main.py
```
Type any question or prompt at the `>>> ` cursor.

**Supported REPL commands:**
- `/quit`: Save state and gracefully exit.
- `/status`: Output a snapshot of the current state variables (`I`, `lambda_L`, `K_ACT`, and `regime`).
- `/save`: Force write the state variables, Physarum graph, and FAISS index immediately.

### 1.2 Single-Task CLI Mode
Allows executing a one-shot query from an external script or terminal:
```bash
python main.py --task "Explain how Prigogine entropy applies to closed systems" --type general
```
Types can be `general`, `code`, or `math`.

### 1.3 Non-Blocking OpenCode CLI Stream Mode
This mode is used by the OpenCode Agent through standard standard input/output streams. The stdin loop runs in a background thread to prevent thread blocks on Windows:
```bash
python main.py --opencode
```
**JSON Input Protocol format:**
```json
{"task": "Write a binary search function", "type": "code"}
```
**JSON Output Protocol format:**
```json
{"answer": "...", "e_total": 0.54, "delta_I": 0.002, "step_time": 6.8, "models": ["model1"]}
```

---

## 2. Local HTTP Bridge Mode (FastAPI)

For a modern integration, launch the FastAPI HTTP Bridge:
```bash
python opencode_bridge.py
```
This boots up an asynchronous web server on `http://127.0.0.1:8000`.

### 2.1 Send Step Task (`POST /run`)
- **URL**: `http://127.0.0.1:8000/run`
- **Payload**:
  ```json
  {
    "task": "Write quicksort function in Python",
    "type": "code"
  }
  ```
- **Response**:
  ```json
  {
    "answer": "def quicksort...",
    "e_total": 0.814,
    "delta_I": 0.0012,
    "step_time": 4.12,
    "models": ["groq_llama33_70b", "deepseek_v3"]
  }
  ```

### 2.2 Get Swarm Status (`GET /status`)
- **URL**: `http://127.0.0.1:8000/status`
- **Response**:
  ```json
  {
    "step": 5,
    "I": 0.5012,
    "lambda_L": 0.0045,
    "k_act": 6,
    "regime": "EDGE",
    "alive_agents": 22,
    "tokens_used": 284000
  }
  ```

---

## 3. Disabling Bad Providers or Models

If specific models are slow, expensive, or experiencing server outages, you can disable them directly by setting `alive: false` in `config/models.yaml`:
```yaml
  - id: google_gemini25_flash
    provider: google
    # ...
    alive: false  # Model will be filtered out from Thompson selection
```
The Swarm will immediately adapt and route queries to active, healthy alternatives.
