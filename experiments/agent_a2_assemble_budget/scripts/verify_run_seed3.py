"""verify_run_seed3.py — честная личная перепроверка пакета A2.

Прогоняет все 8 ячеек `CELLS_CONFIG` (из `scripts/run_all.py`, не дублируется
здесь) на seed=20260903 — сиде, которого не было ни в одном существующем
файле `runs/`/`metrics/summary.json`. Это гарантирует, что резюмируемая
логика `arch2/evolve.py::Evolution.evaluate()` (пропуск уже посчитанных пар
"комплекс-задача", если файл трассы уже на диске) физически не может
подхватить существующий кэш: новый seed -> новые генотипы -> новые
`complex_id` -> новые файлы трасс, которых на диске ещё нет.

Использует `src/orchestrator.py::run_cell_experiment` БЕЗ ИЗМЕНЕНИЙ — тот же
код, что породил исходные 16 прогонов. Если бы исходный прогон был подделкой
с другой (нечестной) логикой, поведение на этом прогоне разошлось бы с
`summary.json`. Совпадение диапазонов -- прямое свидетельство, что механизм
настоящий.

Печатает прогресс по каждой ячейке вживую и логирует фактическое потребление
памяти процесса (через psutil, замеры каждые несколько секунд в фоновом
потоке) -- не апостериорную оценку, а измерение по ходу прогона.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AGENT_DIR = ROOT / "agent_a2_assemble_budget"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import psutil                                    # noqa: E402

from scripts.run_all import CELLS_CONFIG          # noqa: E402
from src.orchestrator import run_cell_experiment  # noqa: E402

SEED = 20260903
RUNS_DIR = AGENT_DIR / "runs"
METRICS_DIR = AGENT_DIR / "metrics"
LOG_PATH = METRICS_DIR / "seed3_memory_log.json"


class _MemWatcher:
    """Фоновый поток: замеряет RSS текущего процесса каждые `interval` сек.
    Наблюдение НЕЗАВИСИМО от логики прогона -- пишет то, что реально видит ОС."""

    def __init__(self, interval: float = 3.0):
        self.interval = interval
        self.samples: list = []
        self._stop = threading.Event()
        self._proc = psutil.Process()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                rss_mb = self._proc.memory_info().rss / (1024 * 1024)
            except Exception:                     # noqa: BLE001
                rss_mb = None
            self.samples.append({"t": round(time.time(), 2), "rss_mb": round(rss_mb, 1) if rss_mb else None})
            self._stop.wait(self.interval)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)


def main() -> int:
    print(f"=== ЧЕСТНАЯ ПЕРЕПРОВЕРКА: seed={SEED}, 8 ячеек, PID={psutil.Process().pid} ===")

    # -- гарантия свежести: ни одна из 8 директорий этого seed не существует заранее --
    pre_existing = [d for cell in CELLS_CONFIG for d in [RUNS_DIR / f"{cell}_s{SEED}"] if d.exists()]
    if pre_existing:
        print(f"[verify_run_seed3] СТОП: уже существуют директории для seed={SEED}: {pre_existing}")
        print("[verify_run_seed3] это означало бы попадание в резюмируемый кэш -- перепроверка невалидна")
        return 1
    print(f"[verify_run_seed3] подтверждено: 0/8 директорий seed={SEED} существуют заранее -- прогон будет свежим")

    watcher = _MemWatcher(interval=3.0)
    watcher.start()

    results = []
    overall_t0 = time.time()
    for cell_name, cfg in CELLS_CONFIG.items():
        t0 = time.time()
        rss_before = psutil.Process().memory_info().rss / (1024 * 1024)
        print(f"\n[{cell_name}] старт (cfg={cfg}), RSS до={rss_before:.1f} MB ...", flush=True)

        res = run_cell_experiment(cell_name, cfg, SEED)

        elapsed = time.time() - t0
        rss_after = psutil.Process().memory_info().rss / (1024 * 1024)
        results.append(res)
        print(f"[{cell_name}] готово за {elapsed:.1f}с | RSS после={rss_after:.1f} MB | "
              f"best_r={res['best_r']:.3f} best_c={res['best_c']:.1f} "
              f"n_gen_run={res['n_generations_run']} extinct={res['extinct']} "
              f"V1={res['v1_pass']} V2={res['v2_pass']} V3={res['v3_pass']} "
              f"Cov_atoms={res['coverage_atoms']:.3f} Cov_pairs={res['coverage_pairs']:.3f} "
              f"P_AB={res['P_A_and_B']:.3f}", flush=True)

    overall_elapsed = time.time() - overall_t0
    watcher.stop()

    rss_values = [s["rss_mb"] for s in watcher.samples if s["rss_mb"] is not None]
    mem_summary = {
        "n_samples": len(watcher.samples),
        "rss_mb_min": min(rss_values) if rss_values else None,
        "rss_mb_max": max(rss_values) if rss_values else None,
        "rss_mb_mean": round(sum(rss_values) / len(rss_values), 1) if rss_values else None,
    }
    print(f"\n=== ИТОГ: 8/8 ячеек, seed={SEED}, {overall_elapsed:.1f}с всего ===")
    print(f"память процесса (psutil, замер каждые 3с): {mem_summary}")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "seed": SEED, "pid": psutil.Process().pid,
            "elapsed_seconds": round(overall_elapsed, 2),
            "memory_summary": mem_summary, "memory_samples": watcher.samples,
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"[verify_run_seed3] лог памяти/времени/результатов записан -> {LOG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
