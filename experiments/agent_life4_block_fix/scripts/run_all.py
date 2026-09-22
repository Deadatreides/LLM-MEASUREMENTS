"""run_all.py — PROTOCOL.md §5: smoke -> 12 клеток -> форензик -> REPORT_LIFE4.md,
одним заходом (не дробить на отдельные шаги позже). Провал смоука -> BLOCKERS.md,
стоп, без 12 клеток (PROTOCOL.md §5 item 2 / §0 non-goals).
"""

from __future__ import annotations

import importlib.util as _ilu
import json
import random
import shutil
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
import orchestrator as O           # noqa: E402
import metrics_lib as ML           # noqa: E402
import block_registry as BR        # noqa: E402
import lean_seeds as LS            # noqa: E402
import baselines as BASE           # noqa: E402


def _load_module(name: str, path: Path):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


VS = _load_module("life4_verify_seams", SCRIPTS / "verify_seams.py")

RUNS_DIR = AGENT_DIR / "runs"
REPORTS_DIR = AGENT_DIR / "reports"
METRICS_DIR = AGENT_DIR / "metrics"

SMOKE_SEED = 999901          # throwaway -- не одно из 12 реальных значений
SMOKE_N_GEN = 8


def _inject_dummy_block(ev, block_registry: dict, blocks_path: Path, gen: int) -> str:
    """PROTOCOL.md §5 item 2: R1-R4 могут не успеть органически за
    SMOKE_N_GEN поколений -- искусственно регистрирует ПЕРВЫЙ доступный
    слот с непустой models-сигнатурой из текущего архива (обходя R1-R4
    гейт, но используя РЕАЛЬНЫЙ `BR.register_block`, не спецпуть), чтобы
    подтвердить, что `BLOCK_INSERT`'s НАСТОЯЩИЙ (не fallback) код-путь
    действительно срабатывает, когда `block_registry` непуст."""
    reference_cids = set(ev.references)
    for cid, g in ev.archive.items():
        if cid in reference_cids:
            continue
        children = (g.get("root") or {}).get("children") or []
        for s, child in enumerate(children):
            models = ML.slot_models(child, ev.registry)
            if models:
                candidate = {"slot": s, "signature": (child.get("op"), tuple(sorted(models))),
                            "carriers": [cid], "n_use": 1, "median_L": 0}
                mid = BR.register_block(ev, candidate, gen, block_registry, blocks_path)
                print(f"[smoke] искусственно зарегистрирован dummy-блок {mid} "
                      f"(slot={s}, carrier={cid}, models={sorted(models)})", flush=True)
                return mid
    raise RuntimeError("smoke: не найден кандидат-слот с непустой models-сигнатурой для dummy-блока")


def run_smoke() -> dict:
    ds = LD.default_dataset()
    panel_info = O.build_panel(ds)
    panel, screen = panel_info["panel"], panel_info["screen"]
    best_single = ML.best_single_per_slot(ds, panel)

    out_dir = RUNS_DIR / "smoke_throwaway"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    births_path = out_dir / "births.jsonl"
    blocks_path = out_dir / "blocks.jsonl"

    reg = LD.HSTEP.Registry()
    be = LD.HSTEP.HeterostepBackend(ds)

    seeds_pop = LS.build_seeds_pop20_lean(ds, panel, best_single)
    LS.assert_lean_population(seeds_pop, ds, panel, best_single, reg)
    print("[smoke] lean-20-distinct + |ImpSet|<2 assert: PASS", flush=True)

    imp_cache: dict = {}

    def imp_fn(genotype: dict) -> frozenset:
        cid = genotype.get("complex_id")
        if cid is not None and cid in imp_cache:
            return imp_cache[cid]
        val = ML.imp_set(ds, panel, best_single, genotype, reg)
        if cid is not None:
            imp_cache[cid] = val
        return val

    rng = random.Random(SMOKE_SEED)
    exp_id = f"life4_smoke_s{SMOKE_SEED}"
    ev = LD.E.Evolution(
        registry=reg, backend=be, dataset=ds, experiment_id=exp_id,
        runs_dir=out_dir / "traces", rng=rng, screen_ids=screen,
        control_slices=[panel], holdout_ids=[],
        assemble_n_steps=len(LD.HSTEP.STEP_KINDS), budget_per_task=O.BUDGET_PER_TASK,
        g_stall=O.G_STALL,
    )
    ev.current_gen = 0
    ev.seed(seeds_pop, reference_names=LD.HSEED.REFERENCE_NAMES,
            baseline_name=LD.HSEED.BASELINE_NAME, gate_reference_name=LD.HSEED.GATE_REFERENCE_NAME)

    block_registry: dict = {}
    dummy_mid = _inject_dummy_block(ev, block_registry, blocks_path, gen=0)

    custom_reproduce = O.make_custom_reproduce(births_path, imp_fn, block_registry, enable_blocks=True)

    for gen in range(SMOKE_N_GEN):
        ev.run_generation(gen)
        if ev.extinct:
            print(f"[smoke] extinct at gen={gen}", flush=True)
            break
        if gen < SMOKE_N_GEN - 1:
            res_data = LD.F.load_results(out_dir / "traces", exp_id)
            custom_reproduce(ev, gen + 1, res_data, panel)

    births = []
    if births_path.exists():
        births = [json.loads(l) for l in births_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    n_transfer_complementary = sum(1 for b in births
                                   if b.get("operator") == "TRANSFER_SLOT" and b.get("complementary"))
    n_block_insert_real = sum(1 for b in births if b.get("operator") == "BLOCK_INSERT")

    print(f"[smoke] TRANSFER_SLOT complementary попыток: {n_transfer_complementary}", flush=True)
    print(f"[smoke] BLOCK_INSERT РЕАЛЬНЫХ (не fallback) попыток: {n_block_insert_real} "
          f"(dummy-блок={dummy_mid})", flush=True)

    fixture_ok = VS.verify_fixture_unfold()

    ok = (n_transfer_complementary >= 1) and (n_block_insert_real >= 1) and fixture_ok
    return {
        "ok": ok, "dummy_block_id": dummy_mid,
        "n_transfer_complementary": n_transfer_complementary,
        "n_block_insert_real": n_block_insert_real,
        "fixture_ok": fixture_ok,
        "n_generations_run": gen + 1,
    }


def _load_archive_ids(cell_dir: Path) -> set:
    with open(cell_dir / "archive.json", encoding="utf-8") as f:
        data = json.load(f)
    return {g["complex_id"] for g in data["genotypes"]}


def run_all_cells(ds, panel: list, screen: list) -> dict:
    best_single = ML.best_single_per_slot(ds, panel)
    results: dict = {}
    for mode in ("WITH", "CTRL"):
        for seed in O.SEEDS:
            out_dir = RUNS_DIR / f"life4_{mode.lower()}_s{seed}"
            out_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            summary = O.run_cell(mode, seed, ds, panel, screen, out_dir)
            dt = time.time() - t0
            print(f"[run_all] {mode} seed={seed} done in {dt:.0f}s "
                  f"n_generations={summary['n_generations']} extinct={summary['extinct']} "
                  f"n_blocks={summary.get('n_blocks_registered')}", flush=True)
            audit = ML.audit_cell(mode, seed, out_dir, ds, panel, best_single, verbose=True)
            results[(mode, seed)] = {"summary": summary, "audit": audit, "wall_s": dt}
    return results


def forensic_pairs(results: dict) -> list:
    out = []
    for seed in O.SEEDS:
        w_audit = results[("WITH", seed)]["audit"]
        c_audit = results[("CTRL", seed)]["audit"]
        n_blocks = (w_audit["block_a1_a5"] or {}).get("N_blocks_registered", 0)
        entry = {
            "seed": seed, "n_blocks": n_blocks,
            "n_ab_first_assembly_with": w_audit["N_AB_first_assembly"],
            "n_ab_first_assembly_ctrl": c_audit["N_AB_first_assembly"],
            "loss_gain_with": w_audit["loss_gain_overall"],
            "loss_gain_ctrl": c_audit["loss_gain_overall"],
            "unfolded_block_signatures": w_audit.get("unfolded_block_signatures") or [],
        }
        if n_blocks > 0:
            w_ids = _load_archive_ids(RUNS_DIR / f"life4_with_s{seed}")
            c_ids = _load_archive_ids(RUNS_DIR / f"life4_ctrl_s{seed}")
            entry["archives_identical"] = (w_ids == c_ids)
            entry["only_in_with"] = sorted(w_ids - c_ids)
            entry["only_in_ctrl"] = sorted(c_ids - w_ids)
            same_metrics = (entry["n_ab_first_assembly_with"] == entry["n_ab_first_assembly_ctrl"]
                           and entry["loss_gain_with"] == entry["loss_gain_ctrl"])
            entry["aggregates_equal_despite_blocks"] = same_metrics
            print(f"[forensic] seed={seed}: n_blocks={n_blocks}, archives_identical="
                  f"{entry['archives_identical']}", flush=True)
            if same_metrics:
                print(f"[forensic] !!! seed={seed}: n_blocks={n_blocks}>0 НО агрегаты WITH==CTRL "
                      f"-- registry сработал, агрегаты не сдвинуты", flush=True)
        out.append(entry)
    return out


def classify_basket(results: dict, grid_ok: bool, fixture_ok: bool) -> tuple:
    if not (grid_ok and fixture_ok):
        return "UNIT4_BLOCKED", {"reason": "grid copy or unfold fixture check failed"}

    a1a5_pass_seeds = [seed for seed in O.SEEDS
                      if (results[("WITH", seed)]["audit"]["block_a1_a5"] or {}).get("all_A1_A5_pass")]
    n_a1a5 = len(a1a5_pass_seeds)

    with_fa = [results[("WITH", s)]["audit"]["N_AB_first_assembly"] for s in O.SEEDS]
    ctrl_fa = [results[("CTRL", s)]["audit"]["N_AB_first_assembly"] for s in O.SEEDS]
    mean_with_fa = statistics.mean(with_fa)
    mean_ctrl_fa = statistics.mean(ctrl_fa)
    paired_sign_pos = sum(1 for w, c in zip(with_fa, ctrl_fa) if w > c)
    paired_sign_neg = sum(1 for w, c in zip(with_fa, ctrl_fa) if w < c)
    fa_grows = (mean_with_fa > mean_ctrl_fa) or (paired_sign_pos >= 4)
    fa_consistently_worse = paired_sign_neg >= 4

    with_lg = [results[("WITH", s)]["audit"]["loss_gain_overall"] for s in O.SEEDS]
    ctrl_lg = [results[("CTRL", s)]["audit"]["loss_gain_overall"] for s in O.SEEDS]
    mean_with_lg = statistics.mean(with_lg)
    mean_ctrl_lg = statistics.mean(ctrl_lg)
    lg_ok = mean_with_lg <= mean_ctrl_lg + 0.5

    detail = {
        "n_a1a5_pass": n_a1a5, "a1a5_pass_seeds": a1a5_pass_seeds,
        "mean_with_first_assembly": mean_with_fa, "mean_ctrl_first_assembly": mean_ctrl_fa,
        "paired_sign_positive": paired_sign_pos, "paired_sign_negative": paired_sign_neg,
        "fa_grows": fa_grows, "fa_consistently_worse": fa_consistently_worse,
        "mean_with_loss_gain": mean_with_lg, "mean_ctrl_loss_gain": mean_ctrl_lg,
        "loss_gain_ok": lg_ok,
    }

    if n_a1a5 >= 5 and fa_grows and lg_ok:
        return "UNIT4_WORKS", detail
    if n_a1a5 <= 2 or fa_consistently_worse:
        return "UNIT4_FAIL", detail
    return "UNIT4_PARTIAL", detail


def write_report(results: dict, forensic: list, basket: str, detail: dict,
                 grid_ok: bool, smoke: dict, b0: dict) -> Path:
    lines = []
    A = lines.append
    A("# REPORT_LIFE4 — измерение (unfold cx) + окно вымирания, LIVE (grid reuse)\n")
    A("Спека: `PROTOCOL.md` (ровно два изменения против `agent_life3_block_live/` "
      "-- unfold cx в `metrics_lib.slot_models` + G_STALL 12->20/max_gen 24->32; "
      "корзины `UNIT4_*` не двигались после чисел). Живая сетка ПЕРЕИСПОЛЬЗОВАНА "
      "byte-identical у `agent_life3_block_live` (md5 подтверждён), 0 новых "
      "generate() вызовов.\n")

    A("## 1. Рамка\n")
    A("LIFE-3-LIVE закрыл A1-A5 на 1/6 WITH-seed (`UNIT_LIVE_FAIL`) и раскрыл две "
      "конкретные причины: `slot_models` слеп к вложенным `cx.`-композитам, и окно "
      "вымирания (~gen 12) вероятно не даёт R1 накопить 3 носителя. Этот пакет "
      "чинит РОВНО эти два места и повторяет тот же WITH/CTRL/6-seed дизайн.\n")

    A("## 2. Грид + смоук\n")
    A(f"- Копия сетки: train/test/whole = 2400/2400/700 строк, md5 совпадает с "
      f"`agent_life3_block_live` (BLOCKERS.md). Grid check: {'OK' if grid_ok else 'FAIL'}.\n")
    A(f"- Фикстура разворота cx (`cx-068fae` слот 2 над `cx-b3b0bb`): "
      f"{'PASS -- 4 модели, не 1' if smoke['fixture_ok'] else 'FAIL'}.\n")
    A(f"- Смоук (seed={SMOKE_SEED}, {smoke['n_generations_run']} поколений): "
      f"TRANSFER_SLOT complementary попыток={smoke['n_transfer_complementary']}, "
      f"BLOCK_INSERT реальных попыток={smoke['n_block_insert_real']} "
      f"(dummy-блок {smoke['dummy_block_id']}). Смоук: {'PASS' if smoke['ok'] else 'FAIL'}.\n")
    if b0:
        A(f"- B0 (лучшая одиночная модель, переиспользованный `whole_grid.json`): "
          f"{b0['model']} (train_rate={b0['train_rate']:.3f}, test_rate={b0['test_rate']:.3f}).\n")

    def _fmt(v, spec=""):
        if v is None:
            return "н/д"
        if spec:
            return format(v, spec)
        return str(v)

    A("## 3. Таблица 6 WITH + 6 CTRL\n")
    A("| mode | seed | A1-A5 | N_blocks | N_AB_first_assembly | elite_AB_share | "
      "loss:gain overall | slot_match | extinct | n_gen |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for mode in ("WITH", "CTRL"):
        for seed in O.SEEDS:
            audit = results[(mode, seed)]["audit"]
            a1a5 = audit["block_a1_a5"]
            a1a5_disp = ("PASS" if a1a5["all_A1_A5_pass"] else a1a5["passes"]) if a1a5 else "н/д"
            n_blocks_disp = a1a5["N_blocks_registered"] if a1a5 else "-"
            A(f"| {mode} | {seed} | {a1a5_disp} | {n_blocks_disp} | "
              f"{audit['N_AB_first_assembly']} | {_fmt(audit['elite_AB_share'], '.3f')} | "
              f"{_fmt(audit['loss_gain_overall'], '.2f')} | {_fmt(audit['slot_match_rate'], '.3f')} | "
              f"{audit['extinct']} | {audit['n_generations']} |")

    A("\n## 4. Парное WITH-CTRL + форензик\n")
    A("| seed | N_AB_first_assembly WITH | CTRL | loss:gain WITH | CTRL | n_blocks(WITH) | "
      "archives_identical | agg_equal_despite_blocks |")
    A("|---|---|---|---|---|---|---|---|")
    for e in forensic:
        A(f"| {e['seed']} | {e['n_ab_first_assembly_with']} | {e['n_ab_first_assembly_ctrl']} | "
          f"{e['loss_gain_with']:.2f} | {e['loss_gain_ctrl']:.2f} | {e['n_blocks']} | "
          f"{e.get('archives_identical', '-')} | {e.get('aggregates_equal_despite_blocks', '-')} |")

    any_blocks = any(e["n_blocks"] > 0 for e in forensic)
    A("\n### 4.1 Сигнатуры блоков после разворота cx\n")
    if any_blocks:
        for e in forensic:
            for sig in e["unfolded_block_signatures"]:
                A(f"- seed={e['seed']}: `{sig['molecule_id']}` slot={sig['slot']} -- "
                  f"записано={sig['models_recorded']} -> после разворота={sig['models_after_unfold']} "
                  f"({'ИЗМЕНИЛОСЬ' if sig['unfold_changed'] else 'без изменений'})")
    else:
        A("Ни один блок не зарегистрирован органически ни в одной из 6 WITH-клеток.")

    A("\n## 5. Корзина\n")
    A(f"**{basket}**\n")
    A(f"- A1-A5 держится на {detail.get('n_a1a5_pass', '-')}/6 WITH-seed "
      f"(seeds: {detail.get('a1a5_pass_seeds', [])}).")
    A(f"- mean N_AB_first_assembly: WITH={detail.get('mean_with_first_assembly', float('nan')):.2f}, "
      f"CTRL={detail.get('mean_ctrl_first_assembly', float('nan')):.2f} "
      f"(парный знак +: {detail.get('paired_sign_positive', '-')}/6, "
      f"-: {detail.get('paired_sign_negative', '-')}/6).")
    A(f"- mean loss:gain(overall): WITH={detail.get('mean_with_loss_gain', float('nan')):.2f}, "
      f"CTRL={detail.get('mean_ctrl_loss_gain', float('nan')):.2f}.")

    A("\n## 6. Сравнение с LIFE-3\n")
    A(f"LIFE-3: `UNIT_LIVE_FAIL`, A1-A5 на 1/6, WITH==CTRL агрегаты на 5/6 (объяснено "
      f"RNG-lockstep-при-пустом-реестре). LIFE-4: **{basket}**, A1-A5 на "
      f"{detail.get('n_a1a5_pass', '-')}/6 после unfold+расширенного окна.")

    A("\n## 7. Non-claims\n")
    A("- Не заявляется превосходство над B3 без CI, исключающего равенство.")
    A("- Не заявляется перенос за пределы HETEROSTEP+6 моделей.")
    A("- `arch2/`, `agent_a5_live_m/`, `agent_life1_mechanism/`, "
      "`agent_life2_complementary/`, `agent_life3_block_live/` не изменялись "
      "(только прочитаны для копии сетки/фикстуры).")
    A("- R1-R4 не изменены вместе с окном (Вариант B1) -- только измерение под ними.")
    A("- `UNIT4_PARTIAL`/`UNIT4_FAIL`/`UNIT4_BLOCKED` -- явно допустимый исход, не провал задания.")

    A("\n## 8. Дальше (ровно одна строка)\n")
    if basket == "UNIT4_WORKS":
        A("Зафиксировать unfold+окно как рабочую конфигурацию; следующий пакет -- "
          "измерение r vs B2/B3 на этом канале ИЛИ новый пул атомов (не новый R1-тюнинг).")
    elif basket == "UNIT4_PARTIAL":
        A("Единственный остаточный рычаг уже назван в §5/§6 выше (темп BLOCK_INSERT "
          "vs качество блока, или недостаточный рост first_assembly) -- новый пакет "
          "только на него.")
    elif basket == "UNIT4_FAIL":
        A("Этаж «блок» отложить; default = complementary HGT без registry "
          "(LIFE-2's подтверждённый механизм); не плодить LIFE-5 с теми же R1.")
    else:
        A("Инфраструктура (grid copy / фикстура) -- см. BLOCKERS.md, вердикт не вынесен.")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "REPORT_LIFE4.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


if __name__ == "__main__":
    print("=== [1/4] verify_seams (seams + grid copy + unfold fixture) ===", flush=True)
    ok_seams = VS.verify_seams()
    ok_grid = VS.verify_grid_copy()
    ok_fixture = VS.verify_fixture_unfold()
    if not (ok_seams and ok_grid and ok_fixture):
        print("[run_all] verify_seams FAILED -> BLOCKERS.md, стоп (UNIT4_BLOCKED)", flush=True)
        sys.exit(1)

    print("\n=== [2/4] smoke ===", flush=True)
    smoke = run_smoke()
    if not smoke["ok"]:
        print(f"[run_all] smoke FAILED: {smoke} -> BLOCKERS.md, стоп (UNIT4_BLOCKED)", flush=True)
        sys.exit(1)

    print("\n=== [3/4] baselines B0 (whole_grid reuse, 0 new calls) + 12 cells ===", flush=True)
    ds = LD.default_dataset()
    panel_info = O.build_panel(ds)
    panel, screen = panel_info["panel"], panel_info["screen"]
    b0 = BASE.build_b0(ds)
    print(f"[run_all] B0 = {b0['model']} (train_rate={b0['train_rate']:.3f}, "
          f"test_rate={b0['test_rate']:.3f})", flush=True)
    (METRICS_DIR / "b0.json").write_text(json.dumps(b0, ensure_ascii=False, indent=2), encoding="utf-8")

    t0 = time.time()
    results = run_all_cells(ds, panel, screen)
    print(f"[run_all] 12 клеток за {time.time() - t0:.0f}с", flush=True)

    print("\n=== [4/4] forensic + report ===", flush=True)
    forensic = forensic_pairs(results)
    basket, detail = classify_basket(results, ok_grid, ok_fixture)
    print(f"[run_all] basket = {basket}: {detail}", flush=True)

    out_path = write_report(results, forensic, basket, detail, ok_grid, smoke, b0)
    print(f"[run_all] отчёт записан: {out_path}", flush=True)
