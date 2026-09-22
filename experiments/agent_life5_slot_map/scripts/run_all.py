"""run_all.py — PROTOCOL.md: verify_seams -> smoke (default_channel only,
short) -> optional evolve (4 seeds, confirmed) -> map (24 mandatory + 4
optional) -> MAP_* baskets -> REPORT_LIFE5.md. One pass, no manual
"one more seed" steps afterward (PROTOCOL.md §7 anti-microtask rule).
"""

from __future__ import annotations

import importlib.util as _ilu
import json
import random
import statistics
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
AGENT_DIR = SCRIPTS.parent
SRC = AGENT_DIR / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import live_dataset as LD          # noqa: E402
import unfold_metrics as ML        # noqa: E402
import lean_seeds as LS            # noqa: E402
import default_channel as DC       # noqa: E402
import orchestrator as O           # noqa: E402
import slot_map as SM              # noqa: E402


def _load_module(name: str, path: Path):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


VS = _load_module("life5_verify_seams", SCRIPTS / "verify_seams.py")
RM = _load_module("life5_run_map", SCRIPTS / "run_map.py")
ROE = _load_module("life5_run_optional_evolve", SCRIPTS / "run_optional_evolve.py")

RUNS_DIR = AGENT_DIR / "runs"
REPORTS_DIR = AGENT_DIR / "reports"
METRICS_DIR = AGENT_DIR / "metrics"

STEP_KINDS = LD.HSTEP.STEP_KINDS
SMOKE_SEED = 999801
SMOKE_N_GEN = 6


def smoke_default_channel() -> dict:
    """Short (6-gen), throwaway-seed check that `default_channel`'s
    simplified dispatcher (transfer_slot | mutate, no third branch)
    actually produces admitted TRANSFER_SLOT births -- the underlying
    operator is already proven 24x across LIFE-3/4; this only exercises
    the NEW (smaller) dispatcher, proportionate to how small the change is."""
    ds = LD.default_dataset()
    panel_info = O.build_panel(ds)
    panel, screen = panel_info["panel"], panel_info["screen"]
    best_single = ML.best_single_per_slot(ds, panel)

    out_dir = RUNS_DIR / "smoke_throwaway"
    import shutil
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    births_path = out_dir / "births.jsonl"

    reg = LD.HSTEP.Registry()
    be = LD.HSTEP.HeterostepBackend(ds)
    seeds_pop = LS.build_seeds_pop20_lean(ds, panel, best_single)
    LS.assert_lean_population(seeds_pop, ds, panel, best_single, reg)
    print("[smoke] lean-20-distinct + |ImpSet|<2 assert: PASS", flush=True)

    imp_cache: dict = {}

    def imp_fn(genotype):
        cid = genotype.get("complex_id")
        if cid is not None and cid in imp_cache:
            return imp_cache[cid]
        val = ML.imp_set(ds, panel, best_single, genotype, reg)
        if cid is not None:
            imp_cache[cid] = val
        return val

    rng = random.Random(SMOKE_SEED)
    exp_id = f"life5_smoke_s{SMOKE_SEED}"
    ev = LD.E.Evolution(
        registry=reg, backend=be, dataset=ds, experiment_id=exp_id,
        runs_dir=out_dir / "traces", rng=rng, screen_ids=screen,
        control_slices=[panel], holdout_ids=[],
        assemble_n_steps=len(STEP_KINDS), budget_per_task=O.BUDGET_PER_TASK,
        g_stall=O.G_STALL,
    )
    ev.current_gen = 0
    ev.seed(seeds_pop, reference_names=LD.HSEED.REFERENCE_NAMES,
            baseline_name=LD.HSEED.BASELINE_NAME, gate_reference_name=LD.HSEED.GATE_REFERENCE_NAME)

    default_reproduce = DC.make_default_reproduce(births_path, imp_fn, p_slot_hgt=O.P_SLOT_HGT)

    for gen in range(SMOKE_N_GEN):
        ev.run_generation(gen)
        if ev.extinct:
            print(f"[smoke] extinct at gen={gen}", flush=True)
            break
        if gen < SMOKE_N_GEN - 1:
            res_data = LD.F.load_results(out_dir / "traces", exp_id)
            default_reproduce(ev, gen + 1, res_data, panel)

    births = []
    if births_path.exists():
        births = [json.loads(l) for l in births_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    n_transfer = sum(1 for b in births if b.get("operator") == "TRANSFER_SLOT" and b.get("admitted"))
    # `arch2.mutate.mutate`'s own report names one of M1-M6 (arch2/mutate.py:268-271,
    # `OPERATORS` dict) or None on total failure -- NOT a literal "MUTATE" string.
    # `arch2.mutate.random_genotype` (used for the fresh-individual slots, same as
    # LIFE-3/4's dispatcher) reports "random" (arch2/mutate.py:322). Dispatcher has
    # exactly 2 real branches (transfer_slot | mutate) plus the fresh-population
    # step; anything outside this set would mean a leaked third branch (e.g. a
    # stray BLOCK_INSERT).
    ALLOWED_OPS = {"TRANSFER_SLOT", "M1", "M2", "M3", "M4", "M5", "M6", "random", None}
    ops_seen = {b.get("operator") for b in births}
    dispatcher_ok = ops_seen <= ALLOWED_OPS
    has_block_fields = any(b.get("molecule_id") or b.get("block_insert_attempted") for b in births)

    ok = (n_transfer >= 1) and dispatcher_ok and not has_block_fields
    print(f"[smoke] admitted TRANSFER_SLOT={n_transfer}, ops_seen={sorted(str(o) for o in ops_seen)}, "
          f"dispatcher_ok={dispatcher_ok}, block-fields leaked={has_block_fields}", flush=True)
    return {"ok": ok, "n_transfer": n_transfer, "ops_seen": sorted(str(o) for o in ops_seen),
           "dispatcher_ok": dispatcher_ok, "has_block_fields": has_block_fields}


def classify_slot_basket(agg_gen1_slot: dict, s: int) -> tuple:
    """PROTOCOL.md §3.3 thresholds, computed on the MANDATORY 24-run pool
    only (never on the optional cross-check)."""
    n_imp = agg_gen1_slot["n_signatures_with_imp"]
    imp_sigs = [sig for sig in agg_gen1_slot["signatures"] if sig["imp"]]
    total_carriers = sum(sig["n_carriers"] for sig in imp_sigs)

    if n_imp <= 1 and total_carriers < 5:
        return "MAP_COVERAGE_GAP", {"n_signatures_with_imp": n_imp, "total_carriers": total_carriers}

    if n_imp >= 3:
        sorted_by_carriers = sorted(imp_sigs, key=lambda d: -d["n_carriers"])
        outlier = sorted_by_carriers[0] if sorted_by_carriers else None
        rest = sorted_by_carriers[1:]
        rest_all_thin = all(sig["n_carriers"] < 3 for sig in rest) if rest else True
        medians = [sig["median_L"] for sig in imp_sigs if sig["median_L"] is not None]
        any_low_median = any(m < 2 for m in medians) if medians else False
        if any_low_median or rest_all_thin:
            return "MAP_STABILITY_GAP", {"n_signatures_with_imp": n_imp,
                                         "median_Ls": medians, "rest_all_thin": rest_all_thin,
                                         "outlier": outlier["models"] if outlier else None}

    return "MAP_OTHER", {"n_signatures_with_imp": n_imp, "total_carriers": total_carriers,
                         "signatures": imp_sigs}


def classify_all(mandatory_map: dict) -> dict:
    per_slot_basket = {}
    for s_str, agg in mandatory_map["agg_gen1"].items():
        s = int(s_str) if isinstance(s_str, str) else s_str
        if STEP_KINDS[s] == "FORMAT":
            continue
        basket, detail = classify_slot_basket(agg, s)
        per_slot_basket[STEP_KINDS[s]] = {"basket": basket, "detail": detail}

    # cross-slot MAP_LOOKUP_ONLY check: LOOKUP reproducible across >=2
    # seeds/packages, READ/COMPUTE not.
    def reproducible_across_2(slot_name: str) -> bool:
        s = STEP_KINDS.index(slot_name)
        agg = mandatory_map["agg_gen1"][str(s)]
        for sig in agg["signatures"]:
            if sig["imp"] and sig["n_seed_appeared"] >= 2:
                return True
        return False

    lookup_repro = reproducible_across_2("LOOKUP")
    read_repro = reproducible_across_2("READ")
    compute_repro = reproducible_across_2("COMPUTE")

    lookup_only = lookup_repro and not read_repro and not compute_repro

    baskets_used = {v["basket"] for v in per_slot_basket.values()}
    if lookup_only:
        overall = "MAP_LOOKUP_ONLY"
    elif len(baskets_used) > 1:
        overall = "MAP_MIXED"
    elif baskets_used:
        overall = next(iter(baskets_used))
    else:
        overall = "MAP_BLOCKED"

    return {"overall": overall, "per_slot": per_slot_basket,
           "lookup_reproducible_2plus": lookup_repro,
           "read_reproducible_2plus": read_repro,
           "compute_reproducible_2plus": compute_repro}


def write_report(mandatory_map: dict, optional_map: dict, verdict: dict, smoke: dict) -> Path:
    lines = []
    A = lines.append
    A("# REPORT_LIFE5 — карта слотов + default без registry\n")
    A("Спека: `PROTOCOL.md`. Блок-этаж (LIFE-3/4, `UNIT_LIVE_FAIL`/`UNIT4_FAIL`, "
      "1/6 оба раза) отложен, не возобновляется. Default = ASSEMBLE + комплементарный "
      "slot-HGT (`p=0.35`), БЕЗ block registry / BLOCK_INSERT вообще.\n")

    A("## 1. Рамка\n")
    A("Не чиним registry снова. Вопрос: дефицит ПОКРЫТИЯ (мало полезных Imp-сигнатур "
      "на слоте) или дефицит УСТОЙЧИВОСТИ (сигнатуры есть, не копятся носители)? "
      "Карта строится по уже существующим архивам LIFE-3/4, не по новому эксперименту.\n")

    A("## 2. Источники данных\n")
    A(f"- Обязательный пул: {mandatory_map['n_runs']} прогонов "
      f"(`agent_life3_block_live`/`agent_life4_block_fix`, WITH+CTRL). "
      f"FORMAT sanity (Imp=0 везде): {'OK' if mandatory_map['format_sanity_ok'] else 'FAIL'}.")
    A("  - Из них 12 CTRL-прогонов УЖЕ являются точным экземпляром Phase-A дефолта "
      "(registry всегда был выключен там) — обязательный пул не чисто «старый код с "
      "мёртвым registry».")
    if optional_map:
        A(f"- Опциональный кросс-чек (подтверждено пользователем): {optional_map['n_runs']} "
          f"новых seed, чистый default-канал (0 registry-кода вообще). "
          f"Смоук (seed={SMOKE_SEED}, {SMOKE_N_GEN} поколений): admitted TRANSFER_SLOT="
          f"{smoke['n_transfer']}, операторы={smoke['ops_seen']}, "
          f"block-поля просочились={smoke['has_block_fields']}. Смоук: "
          f"{'PASS' if smoke['ok'] else 'FAIL'}.")
        A(f"  - FORMAT sanity (кросс-чек): "
          f"{'OK' if optional_map['format_sanity_ok'] else 'FAIL'}.")
    else:
        A("- Опциональный кросс-чек: не запускался.")

    A("\n## 3. Таблицы по слотам (обязательный пул, gen>=1)\n")
    for slot_name in STEP_KINDS:
        s = STEP_KINDS.index(slot_name)
        agg = mandatory_map["agg_gen1"][str(s)]
        A(f"### {slot_name} (slot {s})\n")
        A(f"N_distinct_signatures={agg['n_distinct_signatures']}, "
          f"N_signatures_with_Imp=1={agg['n_signatures_with_imp']}\n")
        top = [sig for sig in agg["signatures"] if sig["imp"]][:5]
        if top:
            A("| models | op | n_carriers | n_seed_appeared | median_L | max_L |")
            A("|---|---|---|---|---|---|")
            for sig in top:
                A(f"| {', '.join(sig['models'])} | {sig['op']} | {sig['n_carriers']} | "
                  f"{sig['n_seed_appeared']} | {sig['median_L']} | {sig['max_L']} |")
        else:
            A("Ни одной Imp=1 сигнатуры не найдено на этом слоте (gen>=1, обязательный пул).")
        dt = mandatory_map["delta_tally_pooled"][str(s)] if str(s) in mandatory_map["delta_tally_pooled"] \
            else mandatory_map["delta_tally_pooled"].get(s)
        if dt:
            A(f"\nΔImp на рождение (admitted, весь пул): +1={dt['gain']}, -1={dt['loss']}, "
              f"0={dt['same']}\n")

    A("## 4. Корзины MAP_*\n")
    for slot_name, info in verdict["per_slot"].items():
        A(f"- **{slot_name}**: {info['basket']} — {info['detail']}")
    A(f"\n**Сводная корзина: {verdict['overall']}**")
    A(f"- LOOKUP воспроизводима на >=2 seed/пакета: {verdict['lookup_reproducible_2plus']}")
    A(f"- READ воспроизводима на >=2 seed/пакета: {verdict['read_reproducible_2plus']}")
    A(f"- COMPUTE воспроизводима на >=2 seed/пакета: {verdict['compute_reproducible_2plus']}")

    A("\n## 5. Следствие\n")
    if verdict["overall"] == "MAP_LOOKUP_ONLY":
        A("Согласуется с LIFE-3/4: LOOKUP — единственный слот с воспроизводимым "
          "multi-seed аттрактором. READ/COMPUTE не показывают воспроизводимой "
          "multi-seed Imp-структуры на этом пуле — не значит «нет атомов», значит "
          "«регистрация там не воспроизводится за это окно/эти seed».")
    else:
        A(f"См. корзины по слотам выше (§4) — исход `{verdict['overall']}` не сводится "
          "к единственному LOOKUP-нарративу, разбор по слоту обязателен, не сводить "
          "к одной строке.")

    A("\n## 6. Non-claims\n")
    A("- Не заявляется достижение/сравнение с B3 в заголовке.")
    A("- Не заявляется, что READ/COMPUTE принципиально не могут иметь атомов — только "
      "что на данном пуле воспроизводимая Imp-структура там не наблюдалась.")
    A("- `arch2/`, `agent_a5_live_m/`, `agent_life1_mechanism/`, `agent_life2_complementary/`, "
      "`agent_life3_block_live/`, `agent_life4_block_fix/` не изменялись (только прочитаны).")
    A("- Ни A1-A5, ни block_registry, ни BLOCK_INSERT, ни R1-R4 здесь не пересматривались.")

    A("\n## 7. Дальше (ровно одна ветка)\n")
    if verdict["overall"] == "MAP_LOOKUP_ONLY":
        A("Не возобновлять block layer; default = complementary HGT без registry "
          "зафиксирован как продуктовый путь. Следующий крупный пакет — измерение r "
          "vs B2/B3 на этом канале, ИЛИ новый пул атомов с явной целью повышения "
          "ceiling (не A1-A5).")
    elif verdict["overall"] == "MAP_COVERAGE_GAP":
        A("Следующий пакет — атомы/V_a только для слота(ов) в MAP_COVERAGE_GAP "
          "(см. §4 для того, какой именно).")
    elif verdict["overall"] == "MAP_STABILITY_GAP":
        A("Следующий пакет — один рычаг survival/мутационной нагрузки на слоте(ах) "
          "в MAP_STABILITY_GAP (см. §4), не новый R1-тюнинг.")
    elif verdict["overall"] == "MAP_MIXED":
        A("Next только по худшему слоту (см. §4 по корзинам), не «всё сразу».")
    else:
        A("Инфраструктура/недостаточно данных — см. BLOCKERS.md.")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "REPORT_LIFE5.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


if __name__ == "__main__":
    print("=== [1/5] verify_seams ===", flush=True)
    ok_seams = VS.verify_seams()
    ok_grid = VS.verify_grid_copy()
    ok_fixture = VS.verify_fixture_unfold()
    if not (ok_seams and ok_grid and ok_fixture):
        print("[run_all] verify_seams FAILED -> BLOCKERS.md, стоп (MAP_BLOCKED)", flush=True)
        sys.exit(1)

    print("\n=== [2/5] smoke (default_channel dispatcher) ===", flush=True)
    smoke = smoke_default_channel()
    if not smoke["ok"]:
        print(f"[run_all] smoke FAILED: {smoke} -> BLOCKERS.md, стоп (MAP_BLOCKED)", flush=True)
        sys.exit(1)

    print("\n=== [3/5] optional evolve (4 seeds, user-confirmed) ===", flush=True)
    t0 = time.time()
    ROE.run_all_seeds()
    print(f"[run_all] optional evolve: {time.time() - t0:.0f}с", flush=True)

    print("\n=== [4/5] slot map (24 mandatory + 4 optional) ===", flush=True)
    map_result = RM.run_phase_b()

    print("\n=== [5/5] baskets + report ===", flush=True)
    verdict = classify_all(map_result["mandatory"])
    print(f"[run_all] verdict = {verdict['overall']}", flush=True)
    out_path = write_report(map_result["mandatory"], map_result["optional"], verdict, smoke)
    print(f"[run_all] отчёт записан: {out_path}", flush=True)
