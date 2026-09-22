"""
tests/test_mechanisms.py — Ш0.4. Интеграционные тесты механизмов.

Чем отличается от tests/test_feedback_loops.py и соседей: те переопределяют
формулы локально и ищут подстроки в orchestrator.py

    src = open('orchestrator.py').read()
    check("R_mem used", "R_mem = getattr(self, '_r_mem'" in src)

то есть проверяют, что в файле есть текст, а не что код работает. Поэтому при
mu, совпадающих у всех девяти агентов ровно, при Phi == 0 и K_ACT == 6 они
показывали «все связи проверены».

Здесь импортируются НАСТОЯЩИЕ SwarmBandit, Hypercycle, ChaosController,
EmergenceMetric, SemanticBlackboard.

Каждый механизм проверяется в ДВУХ состояниях: с выключенным флагом правки
(утверждается сломанное поведение — это фиксация дефекта) и с включённым
(утверждается исправленное). Поэтому набор зелёный и до, и после Ш1/Ш2, а
регрессия видна сразу.

Запуск:
    python -m pytest tests/test_mechanisms.py -v
"""

import copy
import os
import sys

import numpy as np
import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.bandit import SwarmBandit                      # noqa: E402
from core.chaos import ChaosController                   # noqa: E402
from core.hypercycle import Hypercycle                   # noqa: E402
from core.scorer import EmergenceMetric                  # noqa: E402


def base_cfg():
    with open(os.path.join(ROOT, 'config', 'settings.yaml'), encoding='utf-8') as f:
        return yaml.safe_load(f)


def cfg_with(**flags):
    c = copy.deepcopy(base_cfg())
    c.setdefault('fix', {}).update(flags)
    return c


def two_models():
    def mk(i, fam):
        return {'id': f'agent_{i}', 'provider': f'local_{fam}', 'family': fam,
                'model_name': fam, 'endpoint': 'http://127.0.0.1:8077/v1',
                'max_rpm': 999, 'roles': ['G', 'C', 'S'], 'mu': 0.5,
                'sigma': 0.5, 'cost_per_1k': 0.0}
    return [mk(0, 'strong'), mk(1, 'weak')]


# ── R5: гиперцикл двигает квоты? ───────────────────────────────────────────

K_GAP_REALISTIC = 0.084      # измерено в прогоне: max(k)-min(k), условие M
K_GAP_AFTER_FIX = 0.54       # проекция при сигнале q_env (диапазон в 6.4x шире)


def _drive_hypercycle(cfg, gap, steps=20000):
    hc = Hypercycle(cfg)
    mean_e = {'G': 0.5 + gap, 'C': 0.5, 'S': 0.5 + gap / 2}
    for _ in range(steps):
        hc.update(mean_e)
    dev = max(abs(v - 1 / 3) for v in hc.x.values())
    allocs = {tuple(sorted(hc.allocate(6).items())) for _ in range(200)}
    return dev, allocs


def test_hypercycle_inert_without_fix():
    """Дефект R5: при выключенных флагах квоты не двигаются даже за 20k шагов
    и даже при разрыве 0.54, много большем наблюдаемого 0.084."""
    cfg = cfg_with(hypercycle_mutation=False, hypercycle_multinomial=False)
    thresh = 1 / (2 * 6)
    dev_obs, alloc_obs = _drive_hypercycle(cfg, K_GAP_REALISTIC)
    dev_big, alloc_big = _drive_hypercycle(cfg, K_GAP_AFTER_FIX)

    assert dev_obs < thresh, (
        f"неожиданно: при наблюдаемом разрыве квоты сдвинулись "
        f"({dev_obs:.5f} >= {thresh:.4f}) — перепроверьте константы")
    assert dev_big < thresh, (
        f"при разрыве {K_GAP_AFTER_FIX} |dx|={dev_big:.5f} < порога "
        f"{thresh:.4f}: мутация 0.03/шаг сильнее каталитического "
        f"дифференциала (~0.0014/шаг)")
    assert len(alloc_big) == 1, f"alloc не должен меняться, а меняется: {alloc_big}"


def test_hypercycle_moves_with_fix():
    """Ш2.3: снижение мутации выводит квоты за порог — но ТОЛЬКО при разрыве
    уровня 0.54, который даёт per_agent_reward. При наблюдаемом 0.084 не
    хватает и с правкой: Ш2.3 бесполезен до Ш1.3."""
    cfg = cfg_with(hypercycle_mutation=True, hypercycle_mutation_rate=0.005,
                   hypercycle_multinomial=False)
    thresh = 1 / (2 * 6)
    dev_obs, _ = _drive_hypercycle(cfg, K_GAP_REALISTIC)
    dev_big, alloc_big = _drive_hypercycle(cfg, K_GAP_AFTER_FIX)

    assert dev_obs < thresh, (
        "при наблюдаемом разрыве правка не должна помогать — иначе вывод "
        "'Ш2.3 downstream от Ш1.3' неверен")
    assert dev_big >= thresh, (
        f"правка не сработала: |dx|={dev_big:.5f} < {thresh:.4f}")
    assert len(alloc_big) > 1 or dev_big >= thresh


def test_phi_is_inert_in_hypercycle():
    """Найдено при подготовке Ш2.3: Phi (и phi_min) не влияют ни на что.

    x_new[c] = x[c]*(1 + a*k[c]*x_prev)/(1 + a*Phi) — знаменатель ОБЩИЙ для
    всех каст, а следующая строка делит на sum(x_new), и общий множитель
    сокращается точно. Значит защита max(Phi, phi_min), введённая аудитом
    v8.5 против «деления на нулевое Phi», охраняет деление, которое не может
    ни на что повлиять.
    """
    cfg_a = cfg_with()
    cfg_b = cfg_with()
    cfg_b['hypercycle'] = dict(cfg_b['hypercycle'])
    cfg_b['hypercycle']['phi_min'] = 0.9        # в 18 раз больше 0.05

    dev_a, _ = _drive_hypercycle(cfg_a, K_GAP_AFTER_FIX, steps=3000)
    dev_b, _ = _drive_hypercycle(cfg_b, K_GAP_AFTER_FIX, steps=3000)
    assert abs(dev_a - dev_b) < 1e-12, (
        f"phi_min всё-таки влияет: {dev_a:.12f} vs {dev_b:.12f} — "
        f"тогда вывод об инертности Phi неверен")


# ── R6: PID управляет K_ACT? ───────────────────────────────────────────────

def _drive_chaos(cfg, steps=40, growth=1.82):
    """Ряд с ГЕОМЕТРИЧЕСКИ растущими приращениями — устойчивый режим «хаос».

    Белый шум для этого не годится: lambda_L в текущей форме безразмерен и
    инвариантен к масштабу, поэтому смена амплитуды шума с 0.01 на 1.0 его не
    двигает. Нужен режим с реальной экспоненциальной расходимостью:
    ln(growth) = ln(1.82) = 0.599 — устойчивое положительное lambda_L.
    """
    ch = ChaosController(cfg)
    ks, lls = [], []
    e, d = 0.0, 1e-6
    for _ in range(steps):
        e += d
        d *= growth
        st = ch.update(e)
        ks.append(st['k_act'])
        lls.append(st['lambda_L'])
    return ks, lls


def test_lyapunov_telescopes():
    """Дефект оценки lambda_L (Ш2.2b), найденный при отладке PID-теста.

    Сумма ln(d_i/d_{i-1}) телескопируется в ln(d_last/d_first), поэтому
    величина зависит только от ДВУХ граничных дельт окна. Путь между ними
    не влияет. Проверяется прямо: одни и те же дельты в другом порядке дают
    другое значение.
    """
    from core.chaos import LyapunovEstimator
    ds = [1e-6 * (1.82 ** i) for i in range(40)]

    def lam(deltas, regression=False):
        est = LyapunovEstimator(window=20, regression=regression)
        e, out = 0.0, []
        for d in deltas:
            e += d
            out.append(est.update(e))
        return out[-1]

    ordered = lam(ds)
    # Детерминированная перестановка. Встроенный hash() для этого не годится:
    # он рандомизирован по процессам (PYTHONHASHSEED), и тест «проходил через
    # раз». Та же ошибка описана в PTG_EMBEDDER.md как найденный там дефект.
    rng = np.random.default_rng(20260809)
    shuffled_ds = list(np.array(ds)[rng.permutation(len(ds))])
    shuffled = lam(shuffled_ds)

    assert abs(ordered - np.log(1.82)) < 1e-6, (
        f"ожидалось ln(1.82)={np.log(1.82):.4f}, получено {ordered:.4f}")
    assert abs(ordered - shuffled) > 0.3, (
        f"перестановка не изменила оценку ({ordered:.4f} vs {shuffled:.4f}) — "
        f"тогда телескопирования нет и вывод неверен")

    # Прямая проверка самого свойства: lambda_L == ln(d_last/d_first)/N,
    # то есть определяется только границами окна.
    from core.chaos import LyapunovEstimator as LE
    est = LE(window=20)
    e = 0.0
    for d in ds:
        e += d
        val = est.update(e)
    hist = list(est.e_history)
    deltas = [abs(hist[i] - hist[i - 1]) for i in range(1, len(hist))]
    telescoped = np.log((deltas[-1] + 1e-6) / (deltas[0] + 1e-6)) / (len(deltas) - 1)
    assert abs(val - telescoped) < 1e-9, (
        f"телескопирование не подтвердилось: {val:.6f} vs {telescoped:.6f}")


def test_lyapunov_regression_is_path_dependent():
    """Ш2.2b: наклон МНК использует все точки окна, поэтому на монотонном
    росте даёт ту же величину, а на перемешанном — устойчиво иную, но не
    швыряется от одной граничной дельты."""
    from core.chaos import LyapunovEstimator
    ds = [1e-6 * (1.82 ** i) for i in range(40)]
    est = LyapunovEstimator(window=20, regression=True)
    e, out = 0.0, []
    for d in ds:
        e += d
        out.append(est.update(e))
    assert abs(out[-1] - np.log(1.82)) < 0.05, (
        f"на чистой экспоненте наклон должен совпасть с ln(1.82)="
        f"{np.log(1.82):.4f}, получено {out[-1]:.4f}")


def test_pid_frozen_without_fix():
    """Дефект R6: PID вызывается раз в 50 шагов и берёт МГНОВЕННОЕ lambda_L.
    На прогоне в 30-40 шагов он срабатывает ровно один раз, на нулевом, когда
    lambda_L ещё 0 — и ставит k_act_center. Ровно это дало K_ACT == 6 во всех
    198 шагах телеметрии."""
    cfg = cfg_with(pid_integrate=False)
    ks, lls = _drive_chaos(cfg, steps=40)
    assert max(lls) > 0.4, (
        f"сигнал не создал режима хаоса (max lambda_L={max(lls):.3f}) — "
        f"тест ни о чём")
    assert len(set(ks)) == 1, (
        f"K_ACT неожиданно изменился: {sorted(set(ks))}. Проверьте "
        f"chaos.pid_interval")


def test_pid_moves_with_fix():
    """Ш2.2: усреднение lambda_L за интервал + интервал 10 → K_ACT реагирует
    на устойчивый хаос (lambda_L ~ +0.6) и снижает число активных агентов.

    Отдельная находка при подгонке этого теста: даже с исправленным
    интервалом PID реагирует МЕДЛЕННО. При kp=0.5, ki=0.1 и затухании
    интеграла 0.97 нужно порядка пяти-шести срабатываний, чтобы |u| дошло до
    0.5 и round(6+u) сдвинулся хотя бы на единицу:

        срабатывание 2:  u = -0.31   round(5.69) = 6
        срабатывание 3:  u = -0.41   round(5.59) = 6
        срабатывание 4:  u = -0.46   round(5.54) = 6
        срабатывание 6:  u = -0.58   round(5.42) = 5   <- первый сдвиг

    При интервале 10 это 50 шагов, при исходном интервале 50 — 250 шагов.
    То есть на прогонах в 30 задач PID не может подействовать в принципе, и
    интервал тут не единственная причина: сами коэффициенты слабы.
    Коэффициенты здесь НЕ трогаю — это отдельная правка после замера, чтобы
    не менять две вещи разом.
    """
    cfg = cfg_with(pid_integrate=True, pid_interval=10)
    ks, lls = _drive_chaos(cfg, steps=60)
    assert len(set(ks)) > 1, (
        f"K_ACT так и не сдвинулся: {sorted(set(ks))} при lambda_L в "
        f"[{min(lls):.3f}, {max(lls):.3f}]")
    assert min(ks) < 6, (
        f"при lambda_L > 0 (хаос) PID обязан СНИЖАТЬ K_ACT, а получилось "
        f"{sorted(set(ks))}")


# ── R3: cost/budget несёт информацию? ──────────────────────────────────────

OBSERVED_TOKENS = [221, 506, 894, 1200, 2488]     # из телеметрии прогона


def test_cost_term_saturated_without_fix():
    """Дефект R3: budget = tokens_per_hour/3600 = 222 токена — это токенов в
    СЕКУНДУ, подставлено как бюджет ШАГА. Все наблюдавшиеся расходы дают
    отношение >= 1, член вырождается в константу."""
    cfg = cfg_with(budget_per_step=False)
    em = EmergenceMetric(cfg)
    budget = cfg['budget']['tokens_per_hour'] / 3600
    vals = [em.compute_base(e_score=0.0, d_sem=0.0, q_syn=0.5, k_act=6,
                            cost=t, budget=budget, lambda_L=0.0)
            for t in OBSERVED_TOKENS]
    # 221 токен даёт 221/222.2 = 0.9946, поэтому «практически константа»,
    # а не буквально. Разброс на три порядка меньше самого члена.
    assert max(vals) - min(vals) < 1e-3, (
        f"член перестал быть практически константой: {vals}")
    assert abs(min(vals) - (-0.182 * 6 / 9)) < 1e-9


def test_cost_term_informative_with_fix():
    """Ш1.1: бюджет шага 18000 → отношение перестаёт быть на клипе."""
    cfg = cfg_with(budget_per_step=True, budget_tokens_per_step=18000)
    em = EmergenceMetric(cfg)
    budget = cfg['fix']['budget_tokens_per_step']
    vals = [em.compute_base(e_score=0.0, d_sem=0.0, q_syn=0.5, k_act=6,
                            cost=t, budget=budget, lambda_L=0.0)
            for t in OBSERVED_TOKENS]
    assert max(vals) - min(vals) > 0.01, (
        f"член всё ещё почти константа: {vals}")


# ── R1: Thompson способен разделить модели, если награда различима? ─────────

def test_bandit_separates_when_rewards_differ():
    """Контроль: сам SwarmBandit не сломан. При РАЗНЫХ наградах mu расходятся.
    Значит проблема R1 не в бандите, а в том, что ему подают одно число."""
    cfg = cfg_with()
    b = SwarmBandit(cfg, two_models())
    vec = np.ones(1408, dtype=np.float32) / np.sqrt(1408)
    for _ in range(200):
        b.update(['agent_0', 'agent_1'],
                 {'agent_0': 0.9, 'agent_1': 0.1}, vec)
    mus = [b.agents['agent_0'].mu, b.agents['agent_1'].mu]
    assert mus[0] - mus[1] > 0.3, f"mu не разошлись: {mus}"


def test_bandit_cannot_separate_on_uniform_reward():
    """Дефект R1 в чистом виде: одинаковая награда → mu тождественно равны.
    Ровно это наблюдалось в прогоне: spread(mu) = 0.000000 по всем агентам."""
    cfg = cfg_with()
    b = SwarmBandit(cfg, two_models())
    vec = np.ones(1408, dtype=np.float32) / np.sqrt(1408)
    for _ in range(200):
        r = 0.5
        b.update(['agent_0', 'agent_1'], {'agent_0': r, 'agent_1': r}, vec)
    mus = [b.agents['agent_0'].mu, b.agents['agent_1'].mu]
    assert abs(mus[0] - mus[1]) < 1e-12, (
        f"mu неожиданно разошлись при одинаковой награде: {mus}")


# ── R1b: LinUCB обновляет сыгранную руку? ──────────────────────────────────

def test_linucb_updates_played_arm_with_fix():
    """Ш1.4: температура, реально использованная в шаге, должна попадать в
    update. Без правки bandit.update пере-выбирает лучшую руку и обновляет
    её — контрфактика не собирается, лидер не может проиграть."""
    cfg = cfg_with(linucb_played_arm=True)
    b = SwarmBandit(cfg, two_models())
    vec = np.ones(1408, dtype=np.float32) / np.sqrt(1408)
    arms = sorted(b.temp_bandit.arms)
    played = arms[0]
    other = arms[-1]
    for _ in range(60):
        b.update(['agent_0'], {'agent_0': 1.0}, vec, temperature=played)
    n_played = int(b.temp_bandit.arms[played].b.any())
    n_other = int(b.temp_bandit.arms[other].b.any())
    assert n_played == 1, "сыгранная рука не обновилась"
    assert n_other == 0, (
        f"обновилась несыгранная рука {other} — значит update по-прежнему "
        f"пере-выбирает лучшую")


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v', '--tb=short']))


# ── Регрессия: обрезка reasoning-тегов не должна съедать ответ ─────────────

def test_strip_think_keeps_answer_when_tag_unclosed():
    """Реальный случай, который уже обнулил целую модель в замере.

    Qwen3-1.7B-Q4_0 при '/no_think' открывает <think>, ничего в него не
    пишет и сразу выдаёт код. Закрывающего тега нет никогда. Наивное
    r'<think>.*$' съедало ответ целиком: 29 из 30 задач вернули пустой текст,
    q_env упала до 0.42, и вывод был бы «модель не умеет в код».
    """
    from lab.llm_server import _strip_think, _truncate_after_fence
    raw = ('<think>\n\n```python\ndef find_shared_elements(a, b):\n'
           '    return list(set(a) & set(b))\n```')
    s = _strip_think(raw)
    assert 'def find_shared_elements' in s, f"ответ потерян: {s!r}"
    assert '<think>' not in s
    t, fenced = _truncate_after_fence(s)
    assert fenced and 'def find_shared_elements' in t


def test_strip_think_removes_closed_block():
    from lab.llm_server import _strip_think
    raw = '<think>\nразмышление с ```кодом``` внутри\n</think>\n\n```python\ndef f(): pass\n```'
    s = _strip_think(raw)
    assert 'размышление' not in s
    assert 'def f()' in s


def test_strip_think_drops_truncated_reasoning():
    """Оборванное на лимите рассуждение без признаков ответа — выбрасываем."""
    from lab.llm_server import _strip_think
    raw = '<think>\nOkay, I need to write a function. First let me think about'
    assert _strip_think(raw) == ''


# ── Ш1.3: вырожденность сигнала награды ────────────────────────────────────

_FOUR_TESTS = (
    "import sys, os\n"
    "sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\n"
    "from solution import *\n"
    "def test_0(): assert f(1) == 1\n"
    "def test_1(): assert f(2) == 4\n"
    "def test_2(): assert f(3) == 9\n"
    "def test_3(): assert f(4) == 16\n"
)
_CANDS = [
    ("4/4", "def f(n): return n*n"),
    ("3/4", "def f(n): return n*n if n != 4 else 0"),
    ("2/4", "def f(n): return n*n if n < 3 else 0"),
    ("1/4", "def f(n): return 1 if n == 1 else 0"),
    ("0/4", "def f(n): return -1"),
]


def _sandbox_available():
    import platform
    return (platform.system() != 'Windows'
            or os.getenv('MYCELIUM_ALLOW_WINDOWS_CODE_EXEC') == '1')


@pytest.mark.skipif(not _sandbox_available(),
                    reason='нужен MYCELIUM_ALLOW_WINDOWS_CODE_EXEC=1')
def test_q_env_collapses_partial_results():
    """Дефект, из-за которого пер-агентная награда была вырождена.

    _q_code_safe: return 1.0 if ratio == 1.0 else max(0.4, ratio * 0.8)
    Всё с ratio <= 0.5 схлопывается ровно в 0.4. Замерено на прогоне: 12 из
    20 черновиков попадали в 0.4, и пять из семи «одинаковых наград» были
    именно этим, а не честной ничьей.
    """
    from core.scorer import QualityScorer
    sc = QualityScorer(base_cfg())
    vals = [sc.q_env(code, 'code', _FOUR_TESTS) for _, code in _CANDS]
    assert len(set(vals)) < len(_CANDS), (
        f"q_env неожиданно различает все случаи: {vals}")
    collapsed = [v for name, v in zip([n for n, _ in _CANDS], vals)
                 if name in ('2/4', '1/4', '0/4')]
    assert len(set(collapsed)) == 1 and abs(collapsed[0] - 0.4) < 1e-9, (
        f"ожидалось схлопывание 2/4, 1/4, 0/4 в 0.4, получено {collapsed}")


@pytest.mark.skipif(not _sandbox_available(),
                    reason='нужен MYCELIUM_ALLOW_WINDOWS_CODE_EXEC=1')
def test_q_env_fine_resolves_all_levels():
    """Ш1.3: непрерывный сигнал различает то, что q_env схлопывает, и
    сохраняет порядок смыслов (синтаксическая смерть хуже 0 из N тестов)."""
    from core.scorer import QualityScorer
    sc = QualityScorer(base_cfg())
    fine = [sc.q_env_fine(code, 'code', _FOUR_TESTS) for _, code in _CANDS]
    assert all(v is not None for v in fine)
    assert len(set(fine)) == len(_CANDS), f"не все различимы: {fine}"
    assert fine == sorted(fine, reverse=True), f"порядок нарушен: {fine}"

    broken = sc.q_env_fine("def f(n) return n*n", 'code', _FOUR_TESTS)
    assert broken < min(fine), (
        f"синтаксическая ошибка ({broken}) должна быть хуже "
        f"худшего рабочего ({min(fine)})")


def test_q_env_fine_returns_none_without_tests():
    """Отсутствие внешней истины должно быть ЯВНЫМ, а не средним значением.

    Прежнее поведение (0.7 за «модуль импортировался» и 0.0 при отсутствии
    pytest) дважды молча обнуляло замеры: метрика выглядела рабочей.
    """
    from core.scorer import QualityScorer
    sc = QualityScorer(base_cfg())
    assert sc.q_env_fine("def f(n): return n", 'code', tests=None) is None
    assert sc.q_env_fine("любой текст", 'general') is None


# ── Ш1.3c: mu по паре (агент, роль) ────────────────────────────────────────

def _drive_roles(cfg, steps=300):
    """Воспроизводит НАСТОЯЩИЙ механизм перестановки, а не «усреднение вредно».

    agent_0 — сильный генератор (0.9), но роль G достаётся ему редко;
    agent_1 — слабый генератор (0.3), зато его синтез часто удачен (0.8).

    Скалярная mu смешивает шкалы и ранжирует по тому, КАКИЕ РОЛИ агент чаще
    играл, а не по тому, насколько он хорош в роли, под которую его сейчас
    отбирают. Ровно это дало на прогоне sh1_30 mu(qwen3_17)=0.616 выше
    mu(coder15_a)=0.599 при одиночном baseline 0.569 против 0.684.
    """
    b = SwarmBandit(cfg, two_models())
    vec = np.ones(1408, dtype=np.float32) / np.sqrt(1408)
    for i in range(steps):
        if i % 5 == 0:                     # редкий шаг: оба генерируют
            roles = {'agent_0': 'G', 'agent_1': 'G'}
            rew = {'agent_0': 0.9, 'agent_1': 0.3}
        else:                              # чаще: оба синтезируют
            roles = {'agent_0': 'S', 'agent_1': 'S'}
            rew = {'agent_0': 0.45, 'agent_1': 0.80}
        b.update(['agent_0', 'agent_1'], rew, vec, roles_played=roles)
    return b


def test_mixed_role_rewards_invert_ranking():
    """Дефект: скалярная mu ставит СЛАБОГО генератора выше сильного, потому
    что он чаще получал щедрые синтезаторские награды."""
    b = _drive_roles(cfg_with(per_role_mu=False))
    a0, a1 = b.agents['agent_0'].mu, b.agents['agent_1'].mu
    assert a1 > a0, (
        f"перестановка не воспроизвелась: agent_0={a0:.3f} agent_1={a1:.3f}")


def test_per_role_mu_restores_ranking():
    """Ш1.3c: в роли G порядок восстанавливается — 0.9 против 0.3."""
    b = _drive_roles(cfg_with(per_role_mu=True))
    g0 = b.agents['agent_0'].role_mu('G')
    g1 = b.agents['agent_1'].role_mu('G')
    assert g0 > g1, f"в роли G порядок не восстановлен: {g0:.3f} vs {g1:.3f}"
    assert g0 - g1 > 0.2, f"разрыв слишком мал: {g0:.3f} vs {g1:.3f}"
    # а в роли S правда обратная, и она тоже должна быть видна
    s0 = b.agents['agent_0'].role_mu('S')
    s1 = b.agents['agent_1'].role_mu('S')
    assert s1 > s0, f"в роли S порядок должен быть обратным: {s0:.3f} vs {s1:.3f}"


def test_per_role_mu_survives_save_load(tmp_path):
    """Без сохранения пер-ролевая статистика терялась бы при каждом
    перезапуске, а оркестратор создаётся заново на каждое условие."""
    b = _drive_roles(cfg_with(per_role_mu=True), steps=40)
    path = str(tmp_path / 'state.json')
    b.save(path)
    b2 = SwarmBandit(cfg_with(per_role_mu=True), two_models())
    b2.load(path)
    assert b2.agents['agent_0'].role_mu('G') == b.agents['agent_0'].role_mu('G')
    assert b2.agents['agent_1'].role_mu('S') == b.agents['agent_1'].role_mu('S')


# ── Ш2.1: масштаб энтропии ─────────────────────────────────────────────────

_CODE_BLOCK = '''```python
def solve(items):
    out = []
    for x in items:
        if x > 0:
            out.append(x * 2)
        else:
            out.append(0)
    return out

class Helper:
    def run(self, n):
        return solve(range(n))
```'''


def _block_entropy(cfg):
    from core.blackboard import SemanticBlackboard
    bb = SemanticBlackboard(cfg)
    b = bb.upsert('blk', _CODE_BLOCK, task='transform a list', task_type='code')
    return b.entropy, b.graph_nodes


def test_entropy_in_nats_breaks_threshold():
    """Дефект: H хранится в натах, а depth.entropy_threshold = 0.08 писался
    под нормированную величину. Гейт «энтропия блока мала → не углубляться»
    не существовал: депт-луп запускался 202/202 раза."""
    cfg = cfg_with(entropy_norm=False)
    h, n = _block_entropy(cfg)
    thresh = cfg['depth']['entropy_threshold']
    assert n > 1, 'граф блока вырожден, тест ни о чём'
    assert h > thresh * 5, (
        f"H={h:.3f} должна быть много больше порога {thresh} — иначе дефект "
        f"не воспроизводится")


def test_entropy_norm_makes_threshold_meaningful():
    """Ш2.1: H_norm ∈ [0,1], и порог 0.08 снова что-то отсекает."""
    cfg = cfg_with(entropy_norm=True)
    h, n = _block_entropy(cfg)
    assert 0.0 <= h <= 1.0, f"H_norm вне [0,1]: {h}"
    assert n > 1
    # на связном коде распределение PPR не равномерно, значит и не 1.0
    assert h < 1.0


def test_entropy_norm_fallback_same_scale():
    """Аварийный путь (n<=1) должен жить в той же шкале, иначе при включённом
    флаге гейты снова поедут: часть блоков в [0,1], часть в натах."""
    from core.blackboard import _text_entropy_heuristic
    txt = 'alpha beta gamma delta alpha beta gamma epsilon zeta'
    nats = _text_entropy_heuristic(txt, normalize=False)
    norm = _text_entropy_heuristic(txt, normalize=True)
    assert nats > 1.0, f"ожидались наты, получено {nats}"
    assert 0.0 <= norm <= 1.0, f"нормированная вне [0,1]: {norm}"


# ── Ш1.3d: частичный пул холодных пер-ролевых ячеек ───────────────────────

def test_cold_role_cell_falls_back_to_scalar():
    """При нуле наблюдений в роли оценка должна равняться скалярной mu.

    Иначе повторяется дефект role2_30: 4-7 наблюдений на ячейку при
    ema_alpha=0.047 дают шум вокруг априорного 0.5, и rho падает с +0.714
    до +0.086 — то есть пер-ролевая оценка оказывается ХУЖЕ скалярной.
    """
    b = SwarmBandit(cfg_with(per_role_mu=True, per_role_pool=True), two_models())
    a = b.agents['agent_0']
    a.mu = 0.80
    a.mu_role['G'] = 0.20            # ячейка есть, но данных нет
    a.calls_role['G'] = 0
    assert abs(a.role_mu('G', pooled=True) - 0.80) < 1e-9, (
        f"при n=0 должна быть скалярная mu, получено "
        f"{a.role_mu('G', pooled=True):.4f}")


def test_warm_role_cell_trusts_itself():
    """При большом числе наблюдений — почти чистая пер-ролевая оценка."""
    b = SwarmBandit(cfg_with(per_role_mu=True, per_role_pool=True), two_models())
    a = b.agents['agent_0']
    a.mu = 0.80
    a.mu_role['G'] = 0.20
    a.calls_role['G'] = 200
    v = a.role_mu('G', pooled=True)
    assert abs(v - 0.20) < 0.05, f"ожидалась ~0.20, получено {v:.4f}"


def test_pooling_is_monotone_in_observations():
    """Вес пер-ролевой оценки должен монотонно расти с числом наблюдений —
    иначе «частичный пул» не частичный, а произвольный."""
    b = SwarmBandit(cfg_with(per_role_mu=True, per_role_pool=True), two_models())
    a = b.agents['agent_0']
    a.mu = 1.0
    a.mu_role['G'] = 0.0
    vals = []
    for n in (0, 2, 5, 10, 30, 100):
        a.calls_role['G'] = n
        vals.append(a.role_mu('G', pooled=True))
    assert vals == sorted(vals, reverse=True), f"немонотонно: {vals}"
    assert vals[0] > 0.99 and vals[-1] < 0.1


def test_role_cell_warm_starts_from_scalar():
    """Новая ячейка стартует от скалярной mu агента, а не от 0.5: иначе
    первые наблюдения тратятся на то, чтобы уползти от априора."""
    cfg = cfg_with(per_role_mu=True, per_role_pool=True)
    b = SwarmBandit(cfg, two_models())
    vec = np.ones(1408, dtype=np.float32) / np.sqrt(1408)
    a = b.agents['agent_0']
    a.mu = 0.9                                    # агент уже «известен»
    b.update(['agent_0'], {'agent_0': 0.9}, vec, roles_played={'agent_0': 'G'})
    assert a.mu_role['G'] > 0.8, (
        f"ячейка стартовала не от скалярной mu: {a.mu_role['G']:.4f}")
