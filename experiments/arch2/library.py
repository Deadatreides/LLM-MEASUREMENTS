"""library.py — стартовые генотипы («затравки» популяции) и эталон B₀.

Все затравки — это КОМПЛЕКСНЫЕ записи того, что уже измерялось поодиночке. Смысл
в том, чтобы отбор стартовал с точек, чей одиночный результат известен, и было видно,
что даёт именно КОМПОЗИЦИЯ, а не молекула.

Эталонов четыре (B0-B3), см. REFERENCE_NAMES. B₀ — один слепой повтор без контекста. Выбран не произвольно: это самая
дешёвая из измеренных вещей, которую за всю программу экспериментов 10-13 не побило
ни одно плечо (эксп. 11 §13: 657.3 ток/успех; эксп. 12 §5: 552.7 ток/успех). Он
всегда присутствует в популяции как измерительная линейка и в конкуренции не
участвует (SPEC.md §6).
"""

from __future__ import annotations

import genotype as G

SEAM_BY_FAMILY = {"arithmetic": "seam.multistep_arith", "code": "seam.enriched_code"}
MAX_TOKENS_BY_FAMILY = {"arithmetic": 150, "code": 320}
FAMILIES = ("arithmetic", "code")

# Подтипы коллизии с ИЗМЕРЕННО высоким барьером (barriers.py, E4.3):
# value 2.30 нат, other 2.54, dependency 2.06 — против syntax 0.52.
HIGH_BARRIER_SUBTYPES = ("value", "other", "dependency")


def attempt(prompt: str, molecule: str = "gen.same_operator", slot: int = 0,
            temperature: float = 0.5) -> dict:
    """Одна попытка «сгенерировать и проверить», построенная под семейство задачи.

    Шов и max_tokens зависят от семейства, поэтому попытка всегда обёрнута в SWITCH
    по `task_family`: язык генотипа не имеет полиморфизма, и это правильно — выбор
    проверки обязан быть виден в самом генотипе, а не прятаться в рантайме.
    """
    cases = {}
    for fam in FAMILIES:
        cases[fam] = G.SEQ(
            G.CALL(molecule, prompt=prompt, temperature=temperature, seed_slot=slot,
                   max_tokens=MAX_TOKENS_BY_FAMILY[fam]),
            G.CHECK(SEAM_BY_FAMILY[fam]),
        )
    return G.SWITCH("task_family", cases, G.STOP("unknown_family"))


def blind(slot: int = 0, molecule: str = "gen.same_operator") -> dict:
    return attempt("primary", molecule=molecule, slot=slot)


def composite_attempt(molecule_id: str) -> dict:
    """Попытка, где генератор — ЗАРЕГИСТРИРОВАННЫЙ комплекс-молекула.

    Отличается от `attempt` ровно одним: у композита пустая сетка параметров
    (SPEC.md §17.3), поэтому `params={}`. Шов по-прежнему выбирается по семейству
    задачи — проверка обязана быть видна в генотипе, а не прятаться в рантайме.
    """
    cases = {}
    for fam in FAMILIES:
        cases[fam] = G.SEQ({"op": "CALL", "molecule": molecule_id, "params": {}},
                           G.CHECK(SEAM_BY_FAMILY[fam]))
    return G.SWITCH("task_family", cases, G.STOP("unknown_family"))


# Четыре эталона, а не один. Сравнивать всё только с B0 было бы подтасовкой: B0 тратит
# один вызов, а комплекс — четыре, и «выиграл по r» означало бы всего лишь «потратил
# больше». Поэтому линеек четыре, и все они защищены от смерти и не участвуют в отборе.
REFERENCE_NAMES = ("B0_blind_resample", "B1_blind_x4_same_model", "B2_best_single_x4",
                   "B3_best_single_per_family_x4")

# Сильнейшая одиночная модель ПО ИТОГУ (oracle@4 = 0.691, barriers.py E4.2). Выбор
# сделан задним числом и в этом весь смысл: B2 — заведомо привилегированная линейка,
# которую комплекс обязан обойти, чтобы «гетерогенность» не оказалась просто
# «угадали сильную модель».
BEST_SINGLE_MODEL = "gemma-3-it-1b-q5_k_s"

# Ещё более привилегированная линейка: лучшая модель ОТДЕЛЬНО ПОД КАЖДОЕ семейство
# задач, тоже задним числом (barriers.py, разбивка по семействам на 55 задачах):
# арифметика — qwen2.5-coder (oracle@4 = 0.4667), код — smollm2 (oracle@4 = 1.0000).
# Нужна, чтобы отделить «гетерогенность помогает» от «просто угадали, какая модель
# сильна на этом типе задач»: B3 знает ответ на второй вопрос заранее.
BEST_SINGLE_BY_FAMILY = {"arithmetic": "qwen2.5-coder-1.5b-instruct-q4_0",
                         "code": "smollm2-1.7b-instruct-q4_k_m"}


def seed_complexes(models: list) -> dict:
    """{имя: генотип}. Имена — только для отчёта; идентичность даёт complex_id."""
    m0, m1, m2, m3 = models[0], models[1], models[2], models[3]
    out = {}

    # -- эталоны --
    out["B0_blind_resample"] = blind(0)
    out["B1_blind_x4_same_model"] = G.PAR(blind(0), blind(1), blind(2), blind(0))
    best = f"gen.{BEST_SINGLE_MODEL}"
    out["B2_best_single_x4"] = G.PAR(blind(0, best), blind(1, best), blind(2, best), blind(3, best))
    out["B3_best_single_per_family_x4"] = G.PAR(
        *[G.SWITCH("task_family",
                   {fam: G.SEQ(G.CALL(f"gen.{BEST_SINGLE_BY_FAMILY[fam]}", prompt="primary",
                                      temperature=0.5, seed_slot=slot,
                                      max_tokens=MAX_TOKENS_BY_FAMILY[fam]),
                               G.CHECK(SEAM_BY_FAMILY[fam]))
                    for fam in FAMILIES},
                   G.STOP("unknown_family"))
          for slot in (0, 1, 2, 3)])

    # -- одиночные плечи (то, что мерили эксперименты 11-13) --
    out["A_retry_K0"] = attempt("K0")
    out["A_retry_promptB"] = attempt("promptB")
    out["A_retry_K2"] = attempt("K2")
    out["A_retry_other_operator"] = attempt("K0", molecule="gen.other_operator")

    # -- чистое размещение бюджета: 4 попытки, разное распределение --
    out["P_cross_model_x4"] = G.PAR(blind(0, f"gen.{m0}"), blind(0, f"gen.{m1}"),
                                    blind(0, f"gen.{m2}"), blind(0, f"gen.{m3}"))
    out["P_mixed_2x2"] = G.PAR(blind(0, f"gen.{m0}"), blind(1, f"gen.{m0}"),
                               blind(0, f"gen.{m1}"), blind(1, f"gen.{m1}"))

    # -- маршрутизация по барьеру: дёшево там, где барьер низкий; STOP там, где высокий --
    out["R_stop_on_high_barrier"] = G.SWITCH(
        "collision_subtype",
        {**{s: G.STOP("barrier_too_high") for s in HIGH_BARRIER_SUBTYPES},
         "syntax": attempt("promptB")},
        blind(0),
    )
    out["R_route_syntax_promptB"] = G.SWITCH(
        "collision_subtype", {"syntax": attempt("promptB")}, blind(0),
    )

    # -- маршрутизация + размещение под общим потолком --
    out["RP_stop_or_cross_model"] = G.BUDGET(700, G.SWITCH(
        "collision_subtype",
        {**{s: G.STOP("barrier_too_high") for s in HIGH_BARRIER_SUBTYPES},
         "syntax": attempt("promptB")},
        G.PAR(blind(0, f"gen.{m0}"), blind(0, f"gen.{m1}")),
    ))

    # -- последовательная эскалация: дёшево, потом дороже, с ранним выходом --
    out["S_escalate_blind_then_retry"] = G.PAR(blind(0), attempt("promptB"))

    return {name: G.genotype(root, gen=0, origin=f"seed:{name}") for name, root in out.items()}


BASELINE_NAME = "B0_blind_resample"          # линейка стоимости (ΔR и E считаются к ней)
GATE_REFERENCE_NAME = "B3_best_single_per_family_x4"   # планка: самый сильный эталон


def baseline_id(models: list) -> str:
    return seed_complexes(models)[BASELINE_NAME]["complex_id"]
