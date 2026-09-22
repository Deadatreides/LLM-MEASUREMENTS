"""evolve.py — Complex Layer, уровень 2: популяция, отбор, СМЕРТЬ, архив.

Реализует SPEC.md §3 и §7.3. Ключевое требование, ради которого слой вообще нужен:
отбор должен уметь УБИВАТЬ, и убивать по проверяемым правилам, а не по вкусу.

Три механизма смерти:
  D1 скрининговая  -- upper95(ΔR) < 0 И u = 0 (защита прекурсора: уникальное решение
                      делает комплекс неубиваемым по экономике на этом шаге);
  D2 субсидия      -- геометрическая, Σ ≤ S₀/(1−λ) аналитически;
  D3 неподтверждение -- элита переизбирается КАЖДОЕ поколение на свежем срезе контроля.

Отдельно: популяция может быть объявлена ВЫМЕРШЕЙ (§3.3). Это предрегистрированный
допустимый исход, а не сбой харнесса.

Фитнес считает `fitness.py` — отдельная программа, читающая трассы с диска.
Здесь он только вызывается.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitness as F
import genotype as G
import heredity as H
import library
import mutate as M
import runner as R

ARCH2 = Path(__file__).resolve().parent

# -- константы, НЕ выводимые из данных (SPEC.md §12) --------------------------------------
POP_SIZE = 24
ELITE_SIZE = 8
SCREEN_SIZE = 40
CONTROL_SIZE = 60
HOLDOUT_SIZE = 98
S_0 = 20_000
LAMBDA = 0.5
G_MAX = 3
G_STALL = 3
U_MIN = 5
N_MIN_REGISTER = 100
MAX_REGISTER_TOTAL = 8      # не более одной за поколение (H.MAX_REGISTER_PER_GEN) x H0_GENERATIONS
FRESH_FRACTION = 0.25
BUDGET_PER_TASK = 1200
FAMILY_QUOTA_DIVISOR = 2          # не более ceil(ELITE/2) с одним корневым семейством
N_BOOT = 2000
TIE_BREAK_ROUND = 4               # SPEC.md §24: округление shared_score для tie-break
D3_FAIL_STREAK_THRESHOLD = 2      # SPEC.md §25: подряд провальных срезов до изгнания


def _atomic_write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    tmp.replace(path)


@dataclass
class Account:
    born_gen: int
    subsidy: float = S_0
    spent_total: int = 0
    subsidy_granted_total: float = 0.0
    overspend_total: float = 0.0
    on_front_ever: bool = False
    novelty_gen: Optional[int] = None      # поколение, когда впервые был u > 0
    # -- наследственность (SPEC.md §16) --
    uses: int = 0                          # успехи самого комплекса ИЛИ его потомков
    last_use_gen: Optional[int] = None     # поколение последнего такого успеха
    w: float = H.W0                        # вес затухания
    shadow: bool = False                   # затух, но не умер; обратимо
    revivals: int = 0
    gate_fail_streak: int = 0              # SPEC.md §25: подряд провальных срезов планки


@dataclass
class Evolution:
    registry: object
    backend: object
    dataset: object
    experiment_id: str
    runs_dir: Path
    rng: random.Random
    screen_ids: list
    control_slices: list
    holdout_ids: list
    baseline_id: str = ""
    gate_reference_id: str = ""
    population: list = field(default_factory=list)
    references: list = field(default_factory=list)
    elite: list = field(default_factory=list)
    shadow: list = field(default_factory=list)
    archive: dict = field(default_factory=dict)
    edges: H.EdgeLog = field(default_factory=H.EdgeLog)
    registered: list = field(default_factory=list)     # [{molecule_id, complex_id, gen, ...}]
    merge_events: int = 0                              # обязан остаться 0 (SPEC.md §14.1)
    archive_sizes: list = field(default_factory=list)  # монотонность архива по поколениям
    accounts: dict = field(default_factory=dict)
    genotypes: dict = field(default_factory=dict)
    generations: list = field(default_factory=list)
    stall_count: int = 0
    extinct: bool = False
    assemble_n_steps: Optional[int] = None
    """Пятый цикл (SPEC.md §33): `None` -- поведение прежнее (MSARITH/код не знают
    ASSEMBLE). Число -- сколько слотов задаёт `random_genotype` при свежей случайной
    генерации для датасета, где генотип ОБЯЗАН быть ASSEMBLE-корневым, чтобы вообще
    видеть данные задачи (HETEROSTEP, `heterostep.py`). Прокидывается в `reproduce()`
    как `ctx["assemble_n_steps"]`."""
    g_stall: Optional[int] = None
    """Пятый цикл (SPEC.md §33): `None` -- прежний модульный `G_STALL=3`. HETEROSTEP
    стартует с 5 членами популяции против 13 именованных генотипов `library.py` --
    холоднее старт правдоподобно требует больше поколений на СОСТАВНОЕ улучшение
    (скрещивание/мутация уже открытых по отдельности отклонений в разных слотах),
    прежде чем судить о вымирании. Явный override, не правка общей константы."""
    budget_per_task: Optional[int] = None
    """Пятый цикл (SPEC.md §33): `None` -- используется старый модульный `BUDGET_PER_TASK`
    (1200, подобран под MSARITH), побайтово прежнее поведение. Измерено: ASSEMBLE-генотип
    B3 (полное жадное покрытие, `heterostep_seeds.py`) фактически тратит до 1552 токенов
    на 23 из 100 train-задач -- 1200 обрезал бы их в EXHAUSTED, занижая r ЭТАЛОНА, не
    только кандидатов. Значение здесь -- явный, видимый override, а не тихая правка
    общей константы."""

    # -- исполнение --------------------------------------------------------------------

    def _trace_path(self, complex_id: str) -> Path:
        return self.runs_dir / self.experiment_id / f"{complex_id}.jsonl"

    def evaluate(self, genotypes: list, task_ids: list, tag: str) -> int:
        """Прогоняет комплексы по задачам, дописывая трассы. Уже посчитанные пары
        (комплекс, задача) пропускаются — кампания резюмируема, как в эксп. 12/13."""
        spent = 0
        for g in genotypes:
            cid = g["complex_id"]
            path = self._trace_path(cid)
            done = set()
            if path.exists():
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        done.add(json.loads(line)["task_id"])
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                for tid in task_ids:
                    if tid in done:
                        continue
                    # `initial_state` -- пятый цикл (SPEC.md §33): интерфейс датасета
                    # обобщён под HETEROSTEP (`heterostep.Dataset`), который не имеет
                    # понятия "состояние коллизии". `replay.Dataset.initial_state`
                    # -- тонкий алиас на `collision_state`, сама она не переименована
                    # (используется по имени в 4 местах tests/), поведение MSARITH-
                    # кампаний E0/E2/E3/H0/H0b/S0/S1 не меняется ни на бит.
                    state = self.dataset.initial_state(tid)
                    budget = self.budget_per_task if self.budget_per_task is not None else BUDGET_PER_TASK
                    res = R.run(g, state, self.registry, self.backend, budget_tokens=budget)
                    spent += res.cost
                    f.write(json.dumps({
                        "experiment_id": self.experiment_id, "phase": tag,
                        "complex_id": cid, "task_id": tid, "outcome": res.outcome,
                        "cost": res.cost, "n_calls": res.n_calls,
                        "final_status": res.final_status, "stop_reason": res.stop_reason,
                        "exhausted": res.exhausted, "trace": res.trace,
                    }, ensure_ascii=False) + "\n")
        return spent

    # -- отбор --------------------------------------------------------------------------

    def _root_family(self, g: dict) -> str:
        """Корневое семейство = семейство первого генератора в порядке обхода.
        Квота по нему — прямое следствие F6/F8: гетерогенность семейств есть
        сильнейшая измеренная ось ортогональности ошибок."""
        for _, node in G.walk(g["root"]):
            if node.get("op") == "CALL":
                fam = self.registry.family_of(node["molecule"])
                return fam or "role"
        return "none"

    def select_elite(self, scored: dict, ids: list, results: dict, task_ids: list,
                     win_details: Optional[dict] = None) -> list:
        points = {cid: scored[cid]["fitness_vector"] for cid in ids if cid in scored}
        layers = F.pareto_layers(points)
        front = set(layers[0]) if layers else set()
        shared = F.shared_scores(results, ids, task_ids)

        def _tie_break_key(cid):
            # Первично -- плотность покрытия (защита от вырождения, F8); вторично --
            # парсимония (SPEC.md §24), включается ТОЛЬКО когда shared_score совпадает
            # с точностью TIE_BREAK_ROUND знаков (иначе плотность по-прежнему решает).
            return (-round(shared.get(cid, 0.0), TIE_BREAK_ROUND),
                    H.complexity_key(self.genotypes[cid], self.registry))

        elite, family_count = [], defaultdict(int)
        cap = -(-ELITE_SIZE // FAMILY_QUOTA_DIVISOR)
        deferred = []
        for layer in layers:
            for cid in sorted(layer, key=_tie_break_key):
                if len(elite) >= ELITE_SIZE:
                    break
                fam = self._root_family(self.genotypes[cid])
                if family_count[fam] >= cap:
                    deferred.append(cid)
                    continue
                elite.append(cid)
                family_count[fam] += 1
        # если квота недобрала элиту — добираем отложенными (квота ограничивает, но не
        # оставляет элиту пустой)
        for cid in deferred:
            if len(elite) >= ELITE_SIZE:
                break
            elite.append(cid)

        return self._reserve_niches(elite, ids, scored, win_details or {}, front)

    def _reserve_niches(self, elite: list, ids: list, scored: dict, win_details: dict,
                        front: set) -> list:
        """Резервирует минимум одного представителя на КВАЛИФИЦИРУЮЩУЮСЯ нишу
        (SPEC.md §23). Не третья ось Pareto: применяется ПОСЛЕ стандартного отбора и
        свопает только избыточных представителей ВНЕ фронта -- комплекс на фронте не
        трогается (тот же инвариант, что уже защищает D2, §3.2)."""
        elite = list(elite)
        elite_niches: dict = defaultdict(list)
        for cid in elite:
            elite_niches[H.niche_key(self.genotypes[cid])].append(cid)

        pop_niches: dict = defaultdict(list)
        for cid in ids:
            if cid in scored:
                pop_niches[H.niche_key(self.genotypes[cid])].append(cid)

        for niche, members in pop_niches.items():
            if niche in elite_niches:
                continue
            best = max(members, key=lambda c: scored[c]["r"])
            if not win_details.get(best, {}).get("beats_gate", False):
                continue
            donor_niche = next((n for n, mem in elite_niches.items()
                                if len(mem) >= 2 and any(m not in front for m in mem)), None)
            if donor_niche is None:
                continue
            swappable = [m for m in elite_niches[donor_niche] if m not in front]
            worst = min(swappable, key=lambda c: scored.get(c, {}).get("r", -1.0))
            elite.remove(worst)
            elite.append(best)
            elite_niches[donor_niche].remove(worst)
            elite_niches[niche] = [best]

        return elite

    # -- смерть -------------------------------------------------------------------------

    def _d1_screen_deaths(self, scored: dict, ids: list) -> list:
        dead = []
        for cid in ids:
            if cid in self.references:
                continue
            s = scored.get(cid)
            if not s:
                continue
            upper = s["delta_r"]["ci_95"][1]
            if upper < 0 and len(s["unique_solves"]) == 0:
                dead.append((cid, f"D1: upper95(dR)={upper:+.4f} < 0 and u=0"))
        return dead

    def _d2_subsidy_deaths(self, scored: dict, ids: list, front: set) -> list:
        """Смерть от исчерпания субсидии — с тремя защитами прекурсора:
        (а) новорождённый живёт минимум одно полное поколение;
        (б) на Парето-фронте (r, −c, u) не умирает никто — а u там ось, значит
            комплекс с уникальными решениями недоминируем и защищён;
        (в) у линии с новизной есть отдельный счёт, но конверсия обязана случиться
            за G_MAX поколений, иначе смерть (защита от бесконечной субсидии)."""
        dead = []
        for cid in ids:
            if cid in self.references:
                continue
            acc = self.accounts[cid]
            if acc.born_gen >= self.current_gen:
                continue
            if cid in front:
                continue
            has_front_descendant = any(
                cid in (self.genotypes[c].get("parent_ids") or []) for c in front
            )
            if has_front_descendant:
                continue
            if acc.novelty_gen is not None and (self.current_gen - acc.novelty_gen) > G_MAX:
                dead.append((cid, f"D2: novelty of gen {acc.novelty_gen} did not convert to "
                                  f"economy within G_MAX={G_MAX} generations"))
                continue
            if acc.subsidy <= 0:
                dead.append((cid, f"D2: subsidy exhausted (granted {acc.subsidy_granted_total:.0f}, "
                                  f"overspend {acc.overspend_total:.0f}), not on Pareto front"))
        return dead

    def _d3_elite_dropouts(self, results: dict, prev_elite: list, gate: str,
                           control_ids: list) -> list:
        """Dual-slice (SPEC.md §25, вариант (a)): изгнание из элиты только после ДВУХ
        ПОДРЯД провальных срезов относительно ПЛАНКИ (не baseline B0 -- задание всё
        формулирует через планку, и это единственное место, где D3 раньше молчаливо
        использовал B0-относительный delta_r из `scored`, а не гейт). `gate_fail_streak`
        сбрасывается при любом непровальном срезе -- один плохой срез больше не убивает."""
        gate_res = results.get(gate, {})
        dropped = []
        for cid in prev_elite:
            if cid in self.references:
                continue
            acc = self.accounts.get(cid)
            if acc is None or cid not in results:
                continue                      # умер по D1 раньше в этом же поколении
            d = F.paired_delta_r(results.get(cid, {}), gate_res, control_ids, n_boot=N_BOOT)
            if H.fail_gate(d):
                acc.gate_fail_streak += 1
            else:
                acc.gate_fail_streak = 0
            if acc.gate_fail_streak >= D3_FAIL_STREAK_THRESHOLD:
                dropped.append((cid, f"D3: {acc.gate_fail_streak} consecutive control-slice "
                                     f"failures vs gate (upper95(dR)={d['ci_95'][1]:+.4f} < 0)"))
        return dropped

    def _dual_slice_report(self, prev_elite: list) -> dict:
        """Информационная таблица «вошёл по одному срезу / данные есть за два»
        (SPEC.md §25) -- НЕ блокирует вход в элиту, только видимость в отчёте."""
        prev_gen_report = self.generations[-1] if self.generations else None
        out = {}
        for cid in self.elite:
            if cid in prev_elite or cid in self.references:
                continue
            prev_vs_gate = (prev_gen_report or {}).get("vs_gate", {}).get(cid) if prev_gen_report else None
            out[cid] = {
                "evaluated_on_previous_slice": prev_vs_gate is not None,
                "beat_gate_on_previous_slice": (prev_vs_gate or {}).get("beats_gate"),
            }
        return out

    def _charge_subsidy(self, ids: list, gen_spend: dict, layers: list, scored: dict) -> None:
        """L_econ = доля бюджета поколения по слою Парето. Перерасход сверх доли
        списывается с субсидии; субсидия выдаётся ТОЛЬКО линии новизны (SPEC.md §3.1)
        и убывает геометрически, поэтому полный расход по линии ≤ S₀/(1−λ).

        Комплекс на фронте свою трату ЗАРАБОТАЛ — с него не списывается ничего."""
        layer_of = {cid: i for i, layer in enumerate(layers) for cid in layer}
        weights = {cid: 0.5 ** layer_of.get(cid, len(layers)) for cid in ids}
        total_w = sum(weights.values()) or 1.0
        total_spend = sum(gen_spend.get(cid, 0) for cid in ids) or 1
        front = set(layers[0]) if layers else set()
        for cid in ids:
            acc = self.accounts[cid]
            spend = gen_spend.get(cid, 0)
            acc.spent_total += spend
            if cid in self.references or cid in front:
                continue
            has_novelty = acc.novelty_gen is not None or (scored.get(cid, {}).get("u", 0) > 0)
            cap = (S_0 * (LAMBDA ** max(0, self.current_gen - acc.born_gen))) if has_novelty else 0.0
            if acc.subsidy > cap:
                acc.subsidy = cap
            acc.subsidy_granted_total += cap
            l_econ = total_spend * weights[cid] / total_w
            over = max(0.0, spend - l_econ)
            acc.overspend_total += over
            acc.subsidy -= over

    # -- цикл ----------------------------------------------------------------------------

    def run_generation(self, gen: int) -> dict:
        self.current_gen = gen
        control_ids = self.control_slices[gen % len(self.control_slices)]
        prev_elite = list(self.elite)

        everyone = self.references + self.population
        spend_before = {cid: self.accounts[cid].spent_total for cid in everyone}

        self.evaluate([self.genotypes[c] for c in everyone], self.screen_ids, f"gen{gen}:screen")
        screen_results = F.load_results(self.runs_dir, self.experiment_id)
        # U(G) считается относительно элиты; пока элиты нет — относительно остальной
        # живой популяции, иначе в поколении 0 защита прекурсора не работала бы вовсе.
        peers = self.elite or [c for c in self.population]
        screen_scored = F.evaluate_population(screen_results, self.population, self.screen_ids,
                                              self.baseline_id, peers, n_boot=N_BOOT)

        d1 = self._d1_screen_deaths(screen_scored, self.population)
        d1 = [(c, why) for c, why in d1 if self.accounts[c].born_gen < gen]
        survivors = [c for c in self.population if c not in {c for c, _ in d1}]

        self.evaluate([self.genotypes[c] for c in survivors + self.references],
                      control_ids, f"gen{gen}:control")
        results = F.load_results(self.runs_dir, self.experiment_id)
        peers = self.elite or survivors
        scored = F.evaluate_population(results, survivors + self.references, control_ids,
                                       self.baseline_id, peers, n_boot=N_BOOT)

        points = {cid: scored[cid]["fitness_vector"] for cid in survivors if cid in scored}
        layers = F.pareto_layers(points)
        front = set(layers[0]) if layers else set()
        for cid in front:
            self.accounts[cid].on_front_ever = True
        for cid in survivors:
            if scored.get(cid, {}).get("unique_solves") and self.accounts[cid].novelty_gen is None:
                self.accounts[cid].novelty_gen = gen

        gen_spend = {}
        all_traces = F.load_results(self.runs_dir, self.experiment_id)
        for cid in everyone:
            total = sum(rec.get("cost", 0) for rec in all_traces.get(cid, {}).values())
            gen_spend[cid] = total - spend_before.get(cid, 0)
        self._charge_subsidy(survivors, gen_spend, layers, scored)

        # Планка — САМЫЙ СИЛЬНЫЙ эталон, и выбирается он ПО ФАКТУ на текущем control_ids,
        # а не назначается заранее. Причина конкретна: B3 («лучшая модель под каждое
        # семейство задач», выбранная задним числом на полном пуле) оказался СЛАБЕЕ B2 на
        # пуле состояний коллизии — подпопуляция другая. Фиксированная планка в такой
        # ситуации молча занижает требование, поэтому берётся максимум по всем эталонам.
        # Вычисляется ОДИН раз и используется везде ниже (D3, элита, регистрация,
        # вымирание) — единый control_ids, единая планка на всё поколение (SPEC.md §22).
        gate = max(self.references, key=lambda c: scored.get(c, {}).get("r", -1.0),
                   default=self.gate_reference_id or self.baseline_id)

        d3 = self._d3_elite_dropouts(results, prev_elite, gate, control_ids)
        d2 = self._d2_subsidy_deaths(scored, survivors, front)
        dead_now = {c for c, _ in d2}
        survivors = [c for c in survivors if c not in dead_now]

        # -- вымирание (§3.3) + предикат beats_gate, нужный нишам (§23) и uses (§16) --
        gate_res = results.get(gate, {})
        win_details = {}
        any_win = False
        for cid in survivors:
            if cid in self.references:
                continue
            d = F.paired_delta_r(results.get(cid, {}), gate_res, control_ids, n_boot=N_BOOT)
            eff = F.bootstrap_efficiency(results.get(cid, {}), gate_res, control_ids, n_boot=N_BOOT)
            lower_e = (eff.get("ci_95") or [0.0, 0.0])[0]
            beats = d["ci_95"][0] > 0 or lower_e > 1.0
            win_details[cid] = {"delta_r_vs_gate": d, "efficiency_vs_gate": eff, "beats_gate": beats}
            any_win = any_win or beats
        self.stall_count = 0 if any_win else self.stall_count + 1
        g_stall = self.g_stall if self.g_stall is not None else G_STALL
        if self.stall_count >= g_stall:
            self.extinct = True

        self.elite = self.select_elite(scored, survivors, results, control_ids, win_details)
        dual_slice_entrants = self._dual_slice_report(prev_elite)

        deaths = [{"complex_id": c, "reason": w} for c, w in d1 + d2]
        report = {
            "generation": gen, "control_slice_size": len(control_ids),
            "control_set_id": f"gen{gen}_slice{gen % len(self.control_slices)}",
            "population": list(self.population), "survivors": list(survivors),
            "elite": list(self.elite), "pareto_front": sorted(front),
            "deaths": deaths,
            "elite_dropouts_D3": [{"complex_id": c, "reason": w} for c, w in d3],
            "gate_reference_id": gate, "gate_r": scored.get(gate, {}).get("r"),
            "vs_gate": win_details, "dual_slice_entrants": dual_slice_entrants,
            "any_complex_beats_gate": any_win, "stall_count": self.stall_count,
            "extinct": self.extinct,
            "scores": {cid: {k: v for k, v in scored[cid].items() if k != "resolved"}
                       for cid in survivors if cid in scored},
            "references": {cid: {k: v for k, v in scored[cid].items() if k != "resolved"}
                           for cid in self.references if cid in scored},
            "baseline": scored.get("_baseline"),
            "accounts": {cid: vars(self.accounts[cid]) for cid in self.population},
            "generation_spend": gen_spend,
            "family_of": {cid: self._root_family(self.genotypes[cid]) for cid in survivors},
        }
        self.population = survivors
        report["parsimony_rejections"] = self._heredity_pass(
            gen, results, control_ids, scored, win_details, gate)

        report["edges"] = self.edges.counts()
        report["shadow"] = list(self.shadow)
        report["registered_molecules"] = list(self.registered)
        report["accounts"] = {cid: vars(self.accounts[cid])
                              for cid in self.population + self.shadow}
        report["merge_events"] = self.merge_events
        report["archive_size"] = len(self.archive)
        report["births_with_composite"] = sum(
            1 for cid in self.population if H.uses_composite(self.genotypes[cid]))
        report["skeletons"] = {cid: H.skeleton(self.genotypes[cid]) for cid in self.elite}
        report["niches"] = {H.niche_key(self.genotypes[cid]): cid for cid in self.elite}

        self.archive_sizes.append(len(self.archive))
        self.generations.append(report)
        return report

    # -- наследственность (SPEC.md §14-17) --------------------------------------------

    def _heredity_pass(self, gen: int, results: dict, control_ids: list,
                       scored: dict, win_details: dict, gate: str) -> list:
        """Хвост поколения: uses/w -> shadow/revival -> рёбра reinforces/echo ->
        попытка регистрации молекулы. Порядок важен: регистрация опирается на уже
        обновлённую элиту, а рёбра — на уже посчитанные фенотипы. -> parsimony_rejections."""
        self._update_uses_and_weight(gen, win_details)
        self._apply_shadow_and_revival(gen)
        self._emit_similarity_edges(gen, results, control_ids)
        return self._try_register_molecule(gen, results, control_ids, scored, gate)

    def _update_uses_and_weight(self, gen: int, win_details: dict) -> None:
        """uses растёт у комплекса И У ВСЕХ ЕГО ПРЕДКОВ: успех потомка — это и есть
        подтверждение полезности линии, ради которого наследственность вводилась."""
        succeeded = set(self.elite)
        for cid, d in (win_details or {}).items():
            if d.get("beats_gate"):
                succeeded.add(cid)

        credited = set()
        for cid in succeeded:
            if cid in self.references:
                continue
            credited.add(cid)
            credited |= self.edges.ancestors(cid)

        for cid in credited:
            acc = self.accounts.get(cid)
            if acc is None:
                continue
            acc.uses += 1
            acc.last_use_gen = gen

        for cid, acc in self.accounts.items():
            if cid in self.references:
                continue
            last = acc.last_use_gen if acc.last_use_gen is not None else acc.born_gen
            acc.w = H.weight(gen - last, acc.uses)

    def _apply_shadow_and_revival(self, gen: int) -> None:
        """Decay НЕ убивает: он переводит в shadow, и это обратимо (SPEC.md §16)."""
        for cid in list(self.population):
            acc = self.accounts[cid]
            if acc.born_gen >= gen:
                continue                      # новорождённому дают дожить поколение
            if H.is_shadow(acc.w):
                acc.shadow = True
                self.population.remove(cid)
                self.shadow.append(cid)

        for cid in list(self.shadow):
            acc = self.accounts[cid]
            if not H.is_shadow(acc.w):
                acc.shadow = False
                acc.subsidy = 0.0             # воскрешение всегда без субсидии (§3.4)
                acc.revivals += 1
                self.shadow.remove(cid)
                self.population.append(cid)

    def _emit_similarity_edges(self, gen: int, results: dict, control_ids: list) -> None:
        """reinforces/echo по ЯВНЫМ CASE-условиям (SPEC.md §15), не по argmax скаляра."""
        live = [c for c in self.population if c not in self.references]
        existing = {(e["src"], e["dst"], e["type"]) for e in self.edges.all()}
        for i, a in enumerate(live):
            for b in live[i + 1:]:
                sem = H.sem_similarity(results.get(a, {}), results.get(b, {}))
                dgen = abs(self.accounts[a].born_gen - self.accounts[b].born_gen)
                rel = H.classify_relation(sem, dgen, self.edges.same_lineage(a, b))
                if rel is None:
                    continue
                key = (a, b, rel)
                if key in existing:
                    continue
                existing.add(key)
                self.edges.add(a, b, rel, gen, {"sem": round(sem, 4), "dgen": dgen})

    def _try_register_molecule(self, gen: int, results: dict, control_ids: list,
                               scored: dict, gate: str) -> list:
        """Замыкание цикла: элитный комплекс становится молекулой (SPEC.md §17).

        Пятый пункт (SPEC.md §24), парсимония: не регистрировать комплекс, СТРОГО
        доминируемый уже зарегистрированной молекулой по (r,-c) при БОЛЬШЕЙ сложности
        по числу узлов -- дешёвая, но более слабая молекула этим не отклоняется
        (дешёвый край Парето обязан проходить, §17). -> список отклонений (V3).
        """
        rejections = []
        if len(self.registered) >= MAX_REGISTER_TOTAL:
            return rejections
        gate_metrics = F.metrics(results.get(gate, {}), control_ids)
        already = {r["complex_id"] for r in self.registered}

        candidates = []
        for cid in self.elite:
            if cid in self.references or cid in already:
                continue
            per_task = results.get(cid, {})
            metrics = F.metrics(per_task, control_ids)
            metrics["n_evaluated"] = len(per_task)          # ВСЕ оценки, не только срез
            n_nodes = G.n_nodes(self.genotypes[cid]["root"])
            verdict = H.registration_gate(cid, metrics, gate_metrics, self.genotypes[cid],
                                          self.registry, N_MIN_REGISTER)
            if not verdict["ok"]:
                continue
            dom = H.dominated_by_registered(metrics["r"], metrics["c"], n_nodes, self.registered)
            if dom is not None:
                rejections.append({
                    "complex_id": cid, "gen": gen, "n_nodes": n_nodes,
                    "dominated_by": dom["molecule_id"],
                    "reason": (f"dominated by {dom['molecule_id']} (r={dom['r']:.3f}, "
                              f"c={dom['c']:.1f}, n_nodes={dom['n_nodes']}) while being "
                              f"more complex ({n_nodes} nodes)"),
                })
                continue
            # при прочих равных — наименьшая избыточность относительно уже элиты
            dens = H.density_term(cid, results, self.elite, control_ids)
            candidates.append((dens, cid, metrics, per_task, n_nodes))

        if not candidates:
            return rejections
        candidates.sort(key=lambda x: -x[0])
        _, cid, metrics, per_task, n_nodes = candidates[0]

        profile = H.cost_profile_from_traces(per_task)
        mid = self.registry.register_composite(cid, self.genotypes[cid], profile)
        self.edges.add(cid, mid, "compressed_into", gen,
                       {"r": metrics["r"], "c": metrics["c"], "cost_profile": profile})
        self.registered.append({
            "molecule_id": mid, "complex_id": cid, "gen": gen,
            "r": metrics["r"], "c": metrics["c"], "n_evaluated": metrics["n_evaluated"],
            "n_nodes": n_nodes, "cost_profile": profile,
        })
        return rejections

    def _donor_score_fn(self, results: dict, task_ids: list):
        """Замыкание для M5: структурно совместимый, но фенотипически дополняющий донор
        (SPEC.md §15). Реализовано здесь, потому что нужны трассы, которых mutate не видит."""
        skels = {}

        def score(donor: dict) -> float:
            did = donor.get("complex_id")
            if did not in skels:
                skels[did] = H.skeleton(donor)
            best = 0.0
            for e in (self.elite or self.population):
                if e == did:
                    continue
                gg = self.genotypes.get(e)
                if gg is None:
                    continue
                if e not in skels:
                    skels[e] = H.skeleton(gg)
                sem = H.sem_similarity(results.get(did, {}), results.get(e, {}))
                best = max(best, H.donor_score(H.skel_similarity(skels[did], skels[e]), sem))
            return best or 0.5

        return score

    def reproduce(self, gen: int, results: Optional[dict] = None,
                  control_ids: Optional[list] = None) -> None:
        """Мутации + горизонтальный перенос + доля свежих случайных.

        Каждое рождение эмитит рёбра наследственности: `mutated_from` всегда,
        `crossed_from` при M5, `instantiated_from` при появлении CALL на композит.
        Без отчёта мутатора эти рёбра восстановить постфактум нельзя (SPEC.md §14).
        """
        results = results if results is not None else F.load_results(self.runs_dir, self.experiment_id)
        # Доноры для M5 — живые, архивные И shadow: затухший генотип остаётся
        # источником подграфов, это и есть «мёртвая линия влияет на будущее» (§9.2/§16).
        #
        # `observable_names` -- духом тот же приём, что `getattr(ctx.registry,
        # "is_composite", None)` в runner.py (пятый цикл, SPEC.md §33): реестр без
        # этого метода (`registry.Registry`, MSARITH/код) получает СТАРЫЙ хардкод
        # `list(G.OBSERVABLES)` побайтово; `heterostep.Registry` объявляет пустой
        # список -- у HETEROSTEP нет понятия коллизии/ретрая, чей контекст кодируют
        # эти наблюдаемые, а SWITCH по ним означал бы вызов `observe()`, которого
        # этот реестр честно не реализует (NotImplementedError, см. heterostep.py).
        obs_fn = getattr(self.registry, "observable_names", None)
        ctx = {
            "generators": [m for m in self.registry.generator_ids()],
            "observables": list(obs_fn()) if obs_fn else list(G.OBSERVABLES),
            "feasible_params": self.backend.feasible_params("arithmetic"),
            "donors": [self.archive[c] for c in self.archive],
            "composites": list(self.registry.composite_ids()),
            "donor_score_fn": self._donor_score_fn(results, control_ids or []),
        }
        if self.assemble_n_steps:
            ctx["assemble_n_steps"] = self.assemble_n_steps
        n_fresh = int(POP_SIZE * FRESH_FRACTION)
        n_children = max(0, POP_SIZE - len(self.population) - n_fresh)

        parents = self.elite or self.population
        parent_w = [max(1e-6, self.accounts[c].w) for c in parents] if parents else []
        added = 0
        for _ in range(n_children * 4):
            if added >= n_children:
                break
            if not parents:
                break
            # Родитель выбирается взвешенно по w(t): затухающая линия реже даёт потомство,
            # но не исключается — исключение было бы смертью, а decay не убивает (§16).
            parent_id = self.rng.choices(parents, weights=parent_w, k=1)[0]
            parent = self.genotypes[parent_id]
            child, report = M.mutate(parent, self.rng, ctx, self.registry, gen)
            if child and child["complex_id"] not in self.genotypes:
                self._admit(child, gen)
                self._emit_birth_edges(child, gen, report, parent_id)
                added += 1

        fresh = 0
        for _ in range(n_fresh * 6):
            if fresh >= n_fresh:
                break
            g, report = M.random_genotype(self.rng, ctx, self.registry, gen)
            if g and g["complex_id"] not in self.genotypes:
                self._admit(g, gen)
                self._emit_birth_edges(g, gen, report, None)
                fresh += 1

    def _emit_birth_edges(self, child: dict, gen: int, report: dict,
                          parent_id: Optional[str]) -> None:
        cid = child["complex_id"]
        op = (report or {}).get("operator")
        if parent_id:
            self.edges.add(cid, parent_id, "mutated_from", gen, {"operator": op})
        donor_id = (report or {}).get("donor_id")
        if donor_id and donor_id != parent_id:
            self.edges.add(cid, donor_id, "crossed_from", gen, {"operator": op})
        for call in H.composite_calls(child):
            self.edges.add(cid, call["molecule"], "instantiated_from", gen,
                           {"node_path": call["node_path"], "operator": op})

    def _admit(self, g: dict, gen: int) -> None:
        """NO_MERGE (SPEC.md §14.1): архив только растёт, ничего не сливается.

        Структурно одинаковый потомок — это ТОТ ЖЕ complex_id по каноническому хешу,
        и сюда он просто не доходит (проверка вызывающего кода). Это не слияние двух
        сущностей, поэтому merge_events остаётся нулём.
        """
        cid = g["complex_id"]
        self.genotypes[cid] = g
        self.archive[cid] = g
        self.accounts[cid] = Account(born_gen=gen)
        self.population.append(cid)

    def seed(self, seeds: dict, *, reference_names=None, baseline_name=None,
            gate_reference_name=None) -> None:
        """Эталоны попадают в `references`: их прогоняют каждое поколение, но они не
        умирают, не размножаются и не занимают мест в элите — это линейки, не участники.

        Три keyword-параметра — пятый цикл (SPEC.md §33): по умолчанию `None`, и
        тогда поведение побайтово прежнее (имена читаются из `library.*`, единственный
        существующий вызов — `evolve.build()`). Явные значения нужны датасету без
        SWITCH("task_family",...) (HETEROSTEP, `heterostep_seeds.py`), чьи эталоны и
        имена структурно не library.py — не ХАРДКОД под второй домен, а параметр.
        """
        ref_names = reference_names if reference_names is not None else library.REFERENCE_NAMES
        base_name = baseline_name if baseline_name is not None else library.BASELINE_NAME
        gate_name = gate_reference_name if gate_reference_name is not None else library.GATE_REFERENCE_NAME
        for name, g in seeds.items():
            cid = g["complex_id"]
            if cid in self.genotypes:
                continue
            self.genotypes[cid] = g
            self.archive[cid] = g
            self.accounts[cid] = Account(born_gen=0)
            if name in ref_names:
                self.references.append(cid)
            else:
                self.population.append(cid)
            if name == base_name:
                self.baseline_id = cid
            if name == gate_name:
                self.gate_reference_id = cid

    def run(self, n_generations: int) -> dict:
        for gen in range(n_generations):
            self.run_generation(gen)
            if self.extinct:
                break
            if gen < n_generations - 1:
                last = self.generations[-1]
                self.reproduce(gen + 1,
                               results=F.load_results(self.runs_dir, self.experiment_id),
                               control_ids=self.control_slices[gen % len(self.control_slices)])
        return self.summary()

    def summary(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "n_generations": len(self.generations),
            "extinct": self.extinct,
            "baseline_id": self.baseline_id,
            "final_population": list(self.population),
            "final_elite": list(self.elite),
            "final_shadow": list(self.shadow),
            "archive_size": len(self.archive),
            "archive_sizes_by_gen": list(self.archive_sizes),
            "merge_events": self.merge_events,
            "edge_counts": self.edges.counts(),
            "edges": self.edges.all(),
            "registered_molecules": list(self.registered),
            "constants": {
                "W_SEM": H.W_SEM, "W_TEMP": H.W_TEMP, "W_DENS": H.W_DENS, "W_SKEL": H.W_SKEL,
                "SEM_HIGH": H.SEM_HIGH, "ECHO_MIN_GEN": H.ECHO_MIN_GEN,
                "TEMP_HALF_LIFE_GENS": H.TEMP_HALF_LIFE_GENS,
                "W0": H.W0, "LAMBDA": H.LAMBDA, "RHO": H.RHO, "W_FLOOR": H.W_FLOOR,
                "MAX_REGISTER_PER_GEN": H.MAX_REGISTER_PER_GEN,
                "M6_WEIGHT_WITH_COMPOSITES": H.M6_WEIGHT_WITH_COMPOSITES,
                "P_COMPOSITE_LEAF": H.P_COMPOSITE_LEAF,
                "TIE_BREAK_ROUND": TIE_BREAK_ROUND,
                "D3_FAIL_STREAK_THRESHOLD": D3_FAIL_STREAK_THRESHOLD,
                "POP_SIZE": POP_SIZE, "ELITE_SIZE": ELITE_SIZE, "SCREEN_SIZE": SCREEN_SIZE,
                "CONTROL_SIZE": CONTROL_SIZE, "S_0": S_0, "LAMBDA": LAMBDA, "G_MAX": G_MAX,
                "G_STALL": G_STALL, "U_MIN": U_MIN, "FRESH_FRACTION": FRESH_FRACTION,
                "BUDGET_PER_TASK": BUDGET_PER_TASK,
                "subsidy_upper_bound_per_line": S_0 / (1 - LAMBDA),
            },
            "generations": self.generations,
        }


# -- сборка популяции --------------------------------------------------------------------


def split_pools(all_ids: list, rng: random.Random) -> dict:
    ids = list(all_ids)
    rng.shuffle(ids)
    screen = ids[:SCREEN_SIZE]
    rest = ids[SCREEN_SIZE:]
    holdout = rest[:HOLDOUT_SIZE]
    control_pool = rest[HOLDOUT_SIZE:]
    slices = [control_pool[i:i + CONTROL_SIZE] for i in range(0, len(control_pool), CONTROL_SIZE)]
    slices = [s for s in slices if len(s) >= CONTROL_SIZE // 2]
    return {"screen": screen, "holdout": holdout, "control_slices": slices}


def build(registry, backend, dataset, experiment_id: str, runs_dir: Path, seed: int) -> Evolution:
    rng = random.Random(seed)
    pools = split_pools(dataset.collision_ids(), random.Random(seed + 1))
    ev = Evolution(
        registry=registry, backend=backend, dataset=dataset, experiment_id=experiment_id,
        runs_dir=runs_dir, rng=rng, screen_ids=pools["screen"],
        control_slices=pools["control_slices"], holdout_ids=pools["holdout"],
    )
    ev.current_gen = 0
    ev.seed(library.seed_complexes(dataset.models))
    return ev
