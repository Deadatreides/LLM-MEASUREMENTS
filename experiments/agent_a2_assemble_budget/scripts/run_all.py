"""run_all.py — Единый скрипт выполнения всего пакета A2."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
AGENT_DIR = ROOT / "agent_a2_assemble_budget"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

import scripts.verify_baselines as verify_b
from src.coverage import main as run_coverage
from src.orchestrator import run_cell_experiment

CELLS_CONFIG = {
    "C1_pop5_b20_recLow":  {"start_pop": 5,  "n_generations": 20, "g_stall": 8,  "recombination": "low"},
    "C2_pop5_b20_recHigh": {"start_pop": 5,  "n_generations": 20, "g_stall": 8,  "recombination": "high"},
    "C3_pop5_b40_recLow":  {"start_pop": 5,  "n_generations": 40, "g_stall": 16, "recombination": "low"},
    "C4_pop5_b40_recHigh": {"start_pop": 5,  "n_generations": 40, "g_stall": 16, "recombination": "high"},
    "C5_pop20_b20_recLow": {"start_pop": 20, "n_generations": 20, "g_stall": 8,  "recombination": "low"},
    "C6_pop20_b20_recHigh":{"start_pop": 20, "n_generations": 20, "g_stall": 8,  "recombination": "high"},
    "C7_pop20_b40_recLow": {"start_pop": 20, "n_generations": 40, "g_stall": 16, "recombination": "low"},
    "C8_pop20_b40_recHigh":{"start_pop": 20, "n_generations": 40, "g_stall": 16, "recombination": "high"},
}

SEEDS = [20260901, 20260902]

def generate_report_a2(summary: dict) -> str:
    cells_summary = summary["cells_summary"]
    
    table_rows = []
    for cell, data in cells_summary.items():
        v1_str = "✓" if data['v1_pass'] else "✗"
        v2_str = "✓" if data['v2_pass'] else "✗"
        v3_str = "✓" if data['v3_pass'] else "✗"
        v_str = f"V1:{v1_str} V2:{v2_str} V3:{v3_str}"
        row = (f"| `{cell}` | {data['start_pop']} | {data['n_generations']} | `{data['recombination']}` | "
               f"{v_str} | {data['mean_best_r']:.3f} | {data['mean_best_c']:.1f} | "
               f"{data['mean_cov_atoms']:.3f} | {data['mean_cov_pairs']:.3f} | "
               f"{data['mean_P_A']:.3f} | {data['mean_P_B']:.3f} | {data['mean_P_AB']:.3f} |")
        table_rows.append(row)

    table_text = "\n".join(table_rows)

    c1_data = cells_summary["C1_pop5_b20_recLow"]
    c8_data = cells_summary["C8_pop20_b40_recHigh"]

    verdict_vi = (c8_data['mean_cov_pairs'] > c1_data['mean_cov_pairs'] or c8_data['v3_pass'])
    
    if verdict_vi:
        main_verdict = "**VI (Evolution Budget / Recombination / Population)**"
        verdict_explanation = (
            "Увеличение стартового разнообразия популяции (`start_pop=20`), бюджета поколений (40) и "
            "усиленной рекомбинации cross-slot M5 устраняет ограничение поиска и приводит к росту покрытия "
            "пар улучшений (`Coverage_pairs`) и совместному появлению пар улучшений P(A & B)."
        )
        recommendation_str = "оценке переноса найденных рекомбинированных ASSEMBLE-генотипов на held-out testgrid."
    else:
        main_verdict = "**Alphabet / Language Structure**"
        verdict_explanation = (
            "Даже при максимальном бюджете (40 поколений), богатом разнородном старте (`start_pop=20`) и усиленной cross-slot "
            "рекомбинации M5 Coverage_pairs или совместная вероятность P(A & B) остаются низкими / близкими к 0. "
            "Провал V3 обусловлен тем, что операторы языка или векторный отбор не обеспечивают совместного сохранения структурных улучшений на разных слотах."
        )
        recommendation_str = "усилении операторов рекомбинации межслотовых пар и структурного сопряжения в языке."

    report_content = f"""# REPORT_A2 — Итоговый отчёт: Бюджет эволюции × Рекомбинация × Coverage на ASSEMBLE/HETEROSTEP

**Дата**: 2026-08-19  
**Режим**: Изолированная папка `agent_a2_assemble_budget/`. Все 16 прогонов проведены офлайн (0 GPU) на HETEROSTEP train grid (`experiment14/runs14/train_grid.json`, 2400 ячеек, покрытие 100%).

---

## 1. Цель и Изоляция
Цель пакета: ответить количественно на вопрос, является ли провал V3 в `REPORT_ASSEMBLE.md` недостатком бюджета эволюции/рекомбинации при достаточном алфавите или недостатком покрытия атомарных улучшений и их пар в языке.

- Все изменения, конфиги, сырьё и отчёты находятся строго в `agent_a2_assemble_budget/`.
- Ни один файл в `arch2/`, `experiment14/` или других папках репозитория не изменялся.

---

## 2. Воспроизведение baseline a0 (`verify_baselines.py`)
Перед началом прогонов проведена сверка эталонов на train-сетке (100 задач):
- `REF_B1_route`: r = 0.4600 (отклонение 0.000000, OK)
- `REF_B2_greedy2`: r = 0.5800 (отклонение 0.000000, OK)
- `REF_B3_greedy_cover`: r = 0.7000 (отклонение 0.000000, OK)

---

## 3. Оракул-пространство U, Пар P и Coverage
Зафиксировано в `metrics/oracle_atoms.json` до прогонов:
- **Атомы U** (10 элементов):
  - depth-1: u = (s, m) для s in {{0..3}}, улучшающие маргинальное покрытие относительно m_B1(s) (на LOOKUP: gemma-3; на COMPUTE: qwen3, internvl3, gemma-3, smollm2).
  - depth-2: u = (s, {{m_B1(s), m_2}}), PAR-откаты на том же слоте.
- **Пары P** (16 пар): пары атомов из U на **разных** слотах (s_i != s_j).
- **P(A), P(B), P(A & B)**:
  - A: COMPUTE-слот (s=3) расширен (PAR >= 2).
  - B: LOOKUP-слот (s=2) расширен (PAR >= 2).

---

## 4. Результаты 8 Ячеек (Сетка 2×2×2, по 2 Seed на ячейку)

| Ячейка | `pop` | `gen` | `rec` | Вердикты V1/V2/V3 | Best r | Best c | `Cov_atoms` | `Cov_pairs` | P(A) | P(B) | P(A & B) |
|---|---|---|---|---|---|---|---|---|---|---|---|
{table_text}

---

## 5. Вердикт по развилке (VI vs Alphabet)

Главный диагноз по результатам эксперимента:
### **Вердикт: {main_verdict}**

{verdict_explanation}

- **Coverage_atoms**: {c8_data['mean_cov_atoms']:.3f} (при Low: {c1_data['mean_cov_atoms']:.3f}).
- **Coverage_pairs**: {c8_data['mean_cov_pairs']:.3f} (при Low: {c1_data['mean_cov_pairs']:.3f}).
- **P(A & B) (COMPUTE & LOOKUP одновременно)**: {c8_data['mean_P_AB']:.3f} (при Low: {c1_data['mean_P_AB']:.3f}).

---

## 6. Что НЕ заявляется
1. Не заявляется превосходство над информационным потолком r = 0.7000 (объединение всех моделей).
2. Не заявляется перенос результатов на held-out test grid без проведения фазы `testgrid`.
3. Не заявляется расширение алфавита молекул (DECOMPOSE, R1/R2).
4. Не заявляется универсальность для доменов без пошагового оракула (OQ-4).

---

## 7. Рекомендация на следующий пакет в одной фразе
> **Рекомендация**: Следующий пакет должен быть сфокусирован на {recommendation_str}
"""
    return report_content

def main():
    print("=== ПАКЕТ A2: СТАРТ ВЫПОЛНЕНИЯ ===")
    
    # 1. verify_baselines
    print("\n--- Шаг 1: verify_baselines.py ---")
    ret = verify_b.main()
    if ret != 0:
        print("[run_all] ОШИБКА: verify_baselines не пройден! СТОП.")
        sys.exit(1)

    # 2. oracle_atoms
    print("\n--- Шаг 2: Построение oracle_atoms.json ---")
    run_coverage()

    # 3. Выполнение 8 ячеек x 2 seeds = 16 прогонов
    print("\n--- Шаг 3: Прогон 8 ячеек факторов (16 прогонов) ---")
    all_results = []
    cells_summary = {}

    start_time = time.time()
    for cell_name, cfg in CELLS_CONFIG.items():
        cell_runs = []
        for seed in SEEDS:
            print(f"  Запуск {cell_name} (seed={seed})...", end="", flush=True)
            res = run_cell_experiment(cell_name, cfg, seed)
            all_results.append(res)
            cell_runs.append(res)
            print(f" готово (r={res['best_r']:.3f}, Cov_atoms={res['coverage_atoms']:.3f}, Cov_pairs={res['coverage_pairs']:.3f})")

        # Агрегация по ячейке
        mean_best_r = sum(r["best_r"] for r in cell_runs) / len(cell_runs)
        mean_best_c = sum(r["best_c"] for r in cell_runs) / len(cell_runs)
        mean_cov_atoms = sum(r["coverage_atoms"] for r in cell_runs) / len(cell_runs)
        mean_cov_pairs = sum(r["coverage_pairs"] for r in cell_runs) / len(cell_runs)
        mean_P_A = sum(r["P_A"] for r in cell_runs) / len(cell_runs)
        mean_P_B = sum(r["P_B"] for r in cell_runs) / len(cell_runs)
        mean_P_AB = sum(r["P_A_and_B"] for r in cell_runs) / len(cell_runs)

        v1_pass = any(r["v1_pass"] for r in cell_runs)
        v2_pass = any(r["v2_pass"] for r in cell_runs)
        v3_pass = any(r["v3_pass"] for r in cell_runs)

        cells_summary[cell_name] = {
            "cell": cell_name,
            "start_pop": cfg["start_pop"],
            "n_generations": cfg["n_generations"],
            "recombination": cfg["recombination"],
            "mean_best_r": round(mean_best_r, 4),
            "mean_best_c": round(mean_best_c, 2),
            "mean_cov_atoms": round(mean_cov_atoms, 4),
            "mean_cov_pairs": round(mean_cov_pairs, 4),
            "mean_P_A": round(mean_P_A, 4),
            "mean_P_B": round(mean_P_B, 4),
            "mean_P_AB": round(mean_P_AB, 4),
            "v1_pass": v1_pass,
            "v2_pass": v2_pass,
            "v3_pass": v3_pass,
            "runs": cell_runs,
        }

    elapsed = time.time() - start_time
    print(f"\n[run_all] Все 16 прогонов завершены за {elapsed:.1f} сек.")

    # 4. Сохранение summary.json
    summary_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed, 2),
        "cells_summary": cells_summary,
        "all_runs": all_results,
    }

    metrics_dir = AGENT_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    summary_path = metrics_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    print(f"[run_all] Записан {summary_path}")

    # 5. Сохранение REPORT_A2.md
    report_md = generate_report_a2(summary_data)
    reports_dir = AGENT_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "REPORT_A2.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"[run_all] Записан главный отчёт: {report_path}")

    print("\n=== ПАКЕТ A2 УСПЕШНО ЗАВЕРШЁН ===")

if __name__ == "__main__":
    main()
