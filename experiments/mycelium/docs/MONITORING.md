# MONITORING — Swarm Health, Logging & Telemetry

Mycelium Sigma v8.9.2 features highly decoupled, UTF-8 secure log outputs and structured step-by-step JSONL telemetry to monitor the swarm's edge of chaos dynamics.

---

## 1. Log Files Directory (`logs/`)

Log files are located in the `logs/` folder in the project root:

| Log File | Logger Channel | Focus / Content |
|----------|----------------|-----------------|
| `logs/main.log` | `main` | REPL session start, user commands, task initialization. |
| `logs/orchestrator.log` | `swarm` | Detailed step-by-step orchestrator execution cycles. |
| `logs/thermo.log` | `thermo` | Thermodynamic Prigogine metrics (`H_old`, `H_new`, `Phi`, `d_eS`, `d_iS`). |
| `logs/providers.log` | `providers` | HTTP requests to providers, latencies, retry attempts, rate limits, and fatal failures. |
| `logs/runtime.log` | Root logger | Generic uncaught exceptions, system-level critical errors. |

All handlers use a standard file size rotation threshold of **10MB** with backup rotation parameters to prevent memory exhaustion, and use explicit **UTF-8** encoding.

---

## 2. Structured Telemetry (`logs/telemetry.jsonl`)

The system records step-level parameters inside `logs/telemetry.jsonl` in JSON Lines format.

### 2.1 Telemetry Payload Schema

Each JSON record contains the following keys:
- `step` (int): Sequential index of the step.
- `task_type` (str): Type of task (`general`, `code`, `math`).
- `lambda` (float): Lyapunov chaos exponent (pid target is 0).
- `phi` (float): Thermodynamic local entropy dissipation $\Phi$.
- `entropy` (float): Personalized PageRank entropy $H(\pi)$.
- `delta_h` (float): Change in entropy ($H_{old} - H_{new}$).
- `e_total` (float): Unified Swarm Emergence Metric.
- `regime` (str): Chaos state classification (`ORDER`, `EDGE`, `CHAOS`).
- `k_act` (int): Number of selected models active in this step.
- `retrieval_count` (int): Number of returned context traces from memory.
- `memory_nodes` (int): Nodes in the Physarum memory graph.
- `memory_edges` (int): Edges in the Physarum memory graph.
- `alive_agents` (int): Count of healthy agents in Thompson routing.
- `provider_failures` (int): Sum of consecutive failures across all providers.
- `tokens_used` (int): Tokens consumed by API completions in this step.
- `latency` (float): Step execution time in seconds.

---

## 3. Analysis & Query Snippets

### 3.1 Analyzing Telemetry in Python (Pandas)
```python
import pandas as pd

# Load structured telemetry
df = pd.read_json("logs/telemetry.jsonl", lines=True)

# Calculate aggregate stats
print("Average Emergence (E_total):", df['e_total'].mean())
print("Edge of Chaos Presence (%):", (df['regime'] == 'EDGE').mean() * 100)
print("Average step latency:", df['latency'].mean(), "seconds")
print("Total tokens consumed:", df['tokens_used'].sum())
```

### 3.2 Grep Diagnostics (UTF-8 safe)
- **Check provider failures**:
  `grep "fatal=True" logs/providers.log`
- **Check thermodynamics**:
  `grep "Thermo" logs/thermo.log`
- **Check Lyapunov regime**:
  `grep "Chaos" logs/orchestrator.log`
