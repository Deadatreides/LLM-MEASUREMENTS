"""
core/scorer.py — Метрика эмерджентности E_total и Scorer
Σ_v8.5 «Мицелий»

Ключевые исправления vs всех предыдущих версий:
1. E_score = Q(Y_syn) - max_i Q(Y_i)  — не r_ens - r_base!
2. D_sem по ОТВЕТАМ, не по распределению роутера
3. Θ(Q-0.6): разнообразие только при достаточном качестве
4. Все веса нормированы: сумма ≈ 1.0
5. Q_llm только при Q_ext ∈ (0.35, 0.65) — экономия GPU
"""

import numpy as np
import ast
import re
from typing import List, Optional, Dict


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def semantic_diversity(embeddings: List[np.ndarray]) -> float:
    """
    D_sem = (2 / K(K-1)) * Σ_{i<j} (1 - cos(e_i, e_j))
    Диапазон [0, 1]. Вычисляется по ответам, не по роутеру!
    """
    k = len(embeddings)
    if k < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(k):
        for j in range(i+1, k):
            total += 1.0 - cosine_similarity(embeddings[i], embeddings[j])
            count += 1
    return total / count if count > 0 else 0.0


class QualityScorer:
    """
    Q = 0.35·Q_ext + 0.28·Q_env + 0.18·consistency + ...
    Q_ext: автоматический (pytest, sympy, heuristic)
    Q_env: sandbox OpenCode
    Q_llm: локальная модель (только в зоне неопределённости)
    """
    def __init__(self, cfg: dict):
        s = cfg['scorer']
        self.w_ext = s['w_ext']
        self.w_env = s['w_env']
        self.w_cons = s['w_consistency']
        self.w_lat = s['w_latency']
        self.lat_norm = s['latency_norm']
        self.w_cost = s['w_cost']
        self.w_info = s['w_info']
        self.len_opt = s['length_optimal']
        self.len_penalty = s['length_penalty']
        # Порог для вызова Q_llm
        self.llm_lo = 0.35
        self.llm_hi = 0.65

    def q_ext(self, text: str, task_type: str = 'general') -> float:
        """Быстрая внешняя оценка без LLM"""
        if task_type == 'code':
            return self._q_code(text)
        elif task_type == 'math':
            return self._q_math(text)
        else:
            return self._q_heuristic(text)

    def _extract_python_code(self, text: str) -> str:
        blocks = re.findall(r'```(?:python|py)?\s*(.*?)```', text,
                            flags=re.IGNORECASE | re.DOTALL)
        return '\n\n'.join(blocks).strip() if blocks else text.strip()

    def _q_code(self, code: str) -> float:
        """Static code scoring only; generated code is never executed on host."""
        code_text = self._extract_python_code(code)
        score = max(self._q_heuristic(code), 0.35)
        try:
            tree = ast.parse(code_text)
        except SyntaxError:
            return min(score, 0.2)

        node_types = {type(node) for node in ast.walk(tree)}
        if ast.FunctionDef in node_types or ast.AsyncFunctionDef in node_types:
            score += 0.12
        if ast.ClassDef in node_types:
            score += 0.08
        if 'def test_' in code_text or 'assert ' in code_text:
            score += 0.10

        dangerous = [
            'os.system', 'subprocess.', 'eval(', 'exec(', '__import__',
            'socket.', 'requests.', 'shutil.rmtree', 'Path.unlink',
        ]
        if any(marker in code_text for marker in dangerous):
            score -= 0.25
        return float(np.clip(score, 0.0, 1.0))

    def _q_math(self, text: str) -> float:
        """sympy проверка если есть уравнения"""
        try:
            import sympy
            # Простая эвристика: ищем числовые результаты
            numbers = re.findall(r'=\s*([-+]?\d+\.?\d*)', text)
            if numbers:
                return min(0.8, 0.4 + 0.1 * len(numbers))
            return self._q_heuristic(text)
        except Exception:
            return self._q_heuristic(text)

    def _q_heuristic(self, text: str) -> float:
        """Быстрые эвристики без внешних вызовов"""
        if not text or len(text) < 20:
            return 0.0
        score = 0.5
        # Структурированность
        if any(x in text for x in ['```', '##', '**', '1.', '- ']):
            score += 0.05
        # Длина (оптимум ~800 токенов ≈ 600 слов)
        words = len(text.split())
        ratio = abs(words / 600 - 1)
        score -= min(self.len_penalty, ratio * 0.15)
        # Признаки ошибки
        error_markers = ['i cannot', "i can't", 'error:', 'exception:',
                         'undefined', 'traceback']
        if any(m in text.lower() for m in error_markers):
            score -= 0.2
        return float(np.clip(score, 0.0, 1.0))


    # ── Детальный результат исполнения ────────────────────────────────────
    # [Ш1.3] Отдельно от q_env, потому что у них разные задачи.
    #
    # q_env — ОЦЕНОЧНАЯ метрика, шестиуровневая, и её менять нельзя: по ней
    # считаются все сравнения с прошлыми прогонами.
    #
    # Но как ИСТОЧНИК НАГРАДЫ она вырождена. Формула
    #     return 1.0 if ratio == 1.0 else max(0.4, ratio * 0.8)
    # схлопывает всё с ratio <= 0.5 ровно в 0.4: черновик, прошедший 0 тестов
    # из 4, и черновик, прошедший 2 из 4, неразличимы. Замерено на прогоне
    # sh1: 0.4 — это 12 из 20 черновиков, и из 7 шагов с одинаковой наградой
    # пять были «обе 0.4». То есть пер-агентная награда вырождалась не из-за
    # числа генераторов, а потому что метрика выбрасывала долю пройденных
    # тестов именно там, где черновики различаются.
    #
    # code_detail() возвращает сырой результат один раз; q_env строит из него
    # прежнюю шкалу, q_env_fine — непрерывную для награды.

    def code_detail(self, code: str, tests: str = None) -> dict:
        """Один запуск песочницы → сырой результат.

        {'status': ok|partial|import_error|syntax_error|timeout|no_sandbox|error,
         'passed': int, 'total': int, 'ratio': float|None}
        """
        import subprocess, tempfile, os, sys

        code_text = self._extract_python_code(code)
        if sys.platform == 'win32' and os.getenv('MYCELIUM_ALLOW_WINDOWS_CODE_EXEC') != '1':
            return {'status': 'no_sandbox', 'passed': 0, 'total': 0, 'ratio': None}
        try:
            ast.parse(code_text)
        except SyntaxError:
            return {'status': 'syntax_error', 'passed': 0, 'total': 0, 'ratio': 0.0}

        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = os.path.join(tmpdir, 'solution.py')
            with open(code_path, 'w', encoding='utf-8') as f:
                f.write(code_text)
            if tests:
                with open(os.path.join(tmpdir, 'test_solution.py'), 'w',
                          encoding='utf-8') as f:
                    f.write(tests)
            try:
                r = subprocess.run(
                    [sys.executable, '-c',
                     'import sys; sys.path.insert(0,"."); import solution'],
                    capture_output=True, timeout=5, cwd=tmpdir)
                if r.returncode != 0:
                    return {'status': 'import_error', 'passed': 0, 'total': 0,
                            'ratio': 0.0}
                if not tests:
                    return {'status': 'ok', 'passed': 0, 'total': 0, 'ratio': None}
                tr = subprocess.run(
                    [sys.executable, '-m', 'pytest', 'test_solution.py',
                     '-q', '--tb=no', '--no-header'],
                    capture_output=True, timeout=10, cwd=tmpdir)
                out = tr.stdout.decode('utf-8', errors='ignore')
                pm = re.search(r'(\d+) passed', out)
                fm = re.search(r'(\d+) failed', out)
                passed = int(pm.group(1)) if pm else 0
                failed = int(fm.group(1)) if fm else 0
                total = passed + failed
                if total == 0:
                    return {'status': 'no_tests_ran', 'passed': 0, 'total': 0,
                            'ratio': None}
                ratio = passed / total
                return {'status': 'ok' if ratio == 1.0 else 'partial',
                        'passed': passed, 'total': total, 'ratio': ratio}
            except subprocess.TimeoutExpired:
                return {'status': 'timeout', 'passed': 0, 'total': 0, 'ratio': 0.0}
            except FileNotFoundError:
                # pytest отсутствует. Раньше здесь возвращался 0.0, и внешняя
                # оценка молча становилась нулём для ВСЕХ решений, включая
                # правильные, продолжая выглядеть рабочей метрикой. Дважды
                # обнуляло замеры (отчёт §9.2). Теперь это явный статус.
                return {'status': 'no_pytest', 'passed': 0, 'total': 0, 'ratio': None}
            except Exception:
                return {'status': 'error', 'passed': 0, 'total': 0, 'ratio': None}

    def q_env_fine(self, answer: str, task_type: str = 'general',
                   tests: str = None, detail: dict = None) -> Optional[float]:
        """Непрерывный внешний сигнал для НАГРАДЫ, [0,1]. None = сигнала нет.

        Порядок сохраняет смысл q_env (синтаксическая смерть хуже частичного
        прохождения), но внутри «частично» различает долю тестов.
        """
        if task_type != 'code':
            return None
        d = detail if detail is not None else self.code_detail(answer, tests)
        s = d.get('status')
        if s == 'syntax_error':
            return 0.0
        if s == 'timeout':
            return 0.05
        if s == 'import_error':
            return 0.10
        if d.get('ratio') is None:
            return None                      # тестов нет — сигнала нет
        # 0.15..1.0: даже 0 из N тестов лучше, чем несобирающийся код
        return float(0.15 + 0.85 * d['ratio'])

    def _q_code_safe(self, code: str, tests: str = None) -> float:
        """
        Безопасное выполнение кода в изолированной среде.
        Возвращает градуированную оценку:
          1.0 — код прошёл тесты
          0.7 — код импортируется без ошибок, тестов нет
          0.4 — тесты запустились, часть упала
          0.2 — SyntaxError или ImportError
          0.1 — timeout (бесконечный цикл)
          0.0 — критическая ошибка sandbox

        ВАЖНО: выполняется в tmpdir, изолировано от хоста.
        resource.setrlimit недоступен на Windows — используем timeout.

        Шкала НЕ меняется: по ней считаются все сравнения с прошлыми
        прогонами. Для награды есть q_env_fine().
        """
        import subprocess, tempfile, os, sys

        code_text = self._extract_python_code(code)
        if sys.platform == 'win32' and os.getenv('MYCELIUM_ALLOW_WINDOWS_CODE_EXEC') != '1':
            return min(self._q_code(code_text), 0.3)

        # Предварительная AST-проверка (быстрая)
        try:
            ast.parse(code_text)
        except SyntaxError:
            return 0.2

        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = os.path.join(tmpdir, 'solution.py')
            with open(code_path, 'w', encoding='utf-8') as f:
                f.write(code_text)

            if tests:
                test_path = os.path.join(tmpdir, 'test_solution.py')
                with open(test_path, 'w', encoding='utf-8') as f:
                    f.write(tests)

            # Ограничения ресурсов (только Linux)
            preexec = None
            if sys.platform != 'win32':
                try:
                    import resource
                    def _limit():
                        try:
                            resource.setrlimit(resource.RLIMIT_AS,
                                               (256 * 1024 * 1024, 256 * 1024 * 1024))
                            resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
                        except Exception:
                            pass
                    preexec = _limit
                except ImportError:
                    pass

            try:
                # Шаг 1: импорт (проверяем выполнимость)
                result = subprocess.run(
                    [sys.executable, '-c',
                     f'import sys; sys.path.insert(0,"{{tmpdir}}"); import solution'],
                    capture_output=True, timeout=5,
                    cwd=tmpdir, preexec_fn=preexec
                )
                if result.returncode != 0:
                    return 0.2   # SyntaxError/ImportError в рантайме

                # Шаг 2: тесты
                if tests:
                    test_result = subprocess.run(
                        [sys.executable, '-m', 'pytest', test_path,
                         '-v', '--tb=no', '-q', '--no-header'],
                        capture_output=True, timeout=10,
                        cwd=tmpdir, preexec_fn=preexec
                    )
                    out = test_result.stdout.decode('utf-8', errors='ignore')
                    import re
                    passed_m = re.search(r'(\d+) passed', out)
                    failed_m = re.search(r'(\d+) failed', out)
                    passed = int(passed_m.group(1)) if passed_m else 0
                    failed = int(failed_m.group(1)) if failed_m else 0
                    total = passed + failed
                    if total == 0:
                        return 0.5
                    ratio = passed / total
                    return 1.0 if ratio == 1.0 else max(0.4, ratio * 0.8)

                return 0.7  # Импортировался, тестов нет

            except subprocess.TimeoutExpired:
                return 0.1   # Бесконечный цикл
            except FileNotFoundError:
                return 0.0   # pytest не установлен
            except Exception:
                return 0.0

    def q_env(self, answer: str, task_type: str = 'general',
             tests: str = None) -> float:
        """
        Оценка среды выполнения.
        Для code: вызывает _q_code_safe() с sandbox.
        Для general/math: нейтральная оценка 0.5.
        """
        if task_type != 'code':
            return 0.5
        return self._q_code_safe(answer, tests)

    def consistency(self, answers: List[str], embeddings: List[np.ndarray]) -> float:
        """Согласованность ответов: высокая → хорошо"""
        if len(answers) < 2 or not embeddings:
            return 0.5
        # Среднее попарное сходство (не то же, что D_sem — там несходство)
        pairs = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                pairs.append(cosine_similarity(embeddings[i], embeddings[j]))
        return float(np.mean(pairs)) if pairs else 0.5

    def score(self, answer: str, task_type: str, latency: float,
              cost: float, budget: float, delta_I: float,
              q_llm: Optional[float] = None) -> float:
        """Итоговый Q"""
        qe = self.q_ext(answer, task_type)
        qv = self.q_env(answer, task_type)

        # Q_llm только при неопределённости
        if self.llm_lo < qe < self.llm_hi:
            # Вызывающий код должен передать q_llm
            ql = q_llm if q_llm is not None else qe
        else:
            ql = qe  # прокси без GPU-свопа

        lat_penalty = self.w_lat * min(latency / self.lat_norm, 1.0)
        cost_penalty = self.w_cost * min(cost / max(budget, 1), 1.0)
        q = (self.w_ext * qe + self.w_env * qv + self.w_cons * 0.5
             - lat_penalty - cost_penalty + self.w_info * delta_I)
        return float(np.clip(q, 0.0, 1.0))


class EmergenceMetric:
    """
    E_total финальный:
    E = 0.381·tanh(E_score/0.089)
      + 0.277·tanh(D_sem)·Θ(Q-0.6)
      - 0.182·(K/9)·(cost/budget)
      - 0.108·tanh(1.92·|λ_L|)
      + 0.051·ΔI

    Проверка диапазонов:
    - tanh(E/0.089) ∈ (-1, 1) → 0.381·(-1..1) = (-0.381, 0.381)
    - tanh(D_sem) ∈ (0, 1) → 0.277·(0..1) = (0, 0.277)  [умножено на Θ]
    - -(K/9)·(c/b) ∈ (-0.182, 0)
    - -tanh(1.92|λ|) ∈ (-0.108, 0)
    - ΔI ∈ (-0.051, 0.051) практически
    Итого: ≈ (-0.75, 0.71) — нормировано, стабильно.
    """
    def __init__(self, cfg: dict):
        e = cfg['emergence']
        self.w1 = e['w_escore']     # 0.381
        self.s1 = e['escore_scale'] # 0.089
        self.w2 = e['w_dsem']       # 0.277
        self.q_th = e['q_threshold']# 0.60
        self.w3 = e['w_cost']       # 0.182
        self.w4 = e['w_chaos']      # 0.108
        self.lL_s = e['lyapunov_scale'] # 1.92
        self.info_weight = 1.0 - e['leaky_decay']
        self.evo_th = e['evolution_threshold']  # 0.80

    def compute_base(self, e_score: float, d_sem: float, q_syn: float,
                     k_act: int, cost: float, budget: float,
                     lambda_L: float) -> float:
        """
        Параметры:
        e_score: Q(Y_syn) - max_i Q(Y_i)  [может быть < 0]
        d_sem:   семантическое разнообразие ответов [0..1]
        q_syn:   качество итогового синтеза [0..1]
        k_act:   количество моделей в шаге
        cost:    токены потраченные в этом шаге
        budget:  токенов в час
        lambda_L: показатель Ляпунова
        """
        term1 = self.w1 * np.tanh(e_score / self.s1)
        theta = 1.0 if q_syn >= self.q_th else 0.0
        term2 = self.w2 * np.tanh(d_sem) * theta
        term3 = -self.w3 * (k_act / 9.0) * min(cost / max(budget, 1), 1.0)
        term4 = -self.w4 * np.tanh(self.lL_s * abs(lambda_L))

        return float(term1 + term2 + term3 + term4)

    def apply_delta_i(self, e_base: float, delta_I: float) -> float:
        return float(e_base + self.info_weight * delta_I)

    def compute(self, e_score: float, d_sem: float, q_syn: float,
                k_act: int, cost: float, budget: float,
                lambda_L: float, delta_I: float = 0.0) -> float:
        e_base = self.compute_base(e_score, d_sem, q_syn, k_act, cost, budget, lambda_L)
        return self.apply_delta_i(e_base, delta_I)

    def is_evolution_worthy(self, e_total: float) -> bool:
        return e_total >= self.evo_th
