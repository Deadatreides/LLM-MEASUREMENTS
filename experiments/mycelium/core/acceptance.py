"""core/acceptance.py — приёмка по исполнению. ОДИН механизм на обе ветки.

TASK_ACCEPTANCE §B.4: «Требование: одинаковый механизм приёмки в M и в BoN-G,
из одной функции. Не два похожих кода, а один вызов из общего модуля. Это
проверяемо: разница между ветками должна остаться ровно в том, как порождаются
кандидаты.»

Поэтому здесь нет ничего про оркестрацию, касты и BoN. На вход — список
кандидатов, на выход — какой взять и почему. Обе ветки зовут `select()`.

Трёхступенчатый гейт (§B.3):
  1. импортируется ли — компиляция и импорт модуля, без тестов. Утечки ноль,
     стоимость ноль. Отсекает syntax/import_error (8 из 32 потерь прогона).
  2. приёмочный ассерт — вынесенный из оценки. Отсекает основную массу
     ratio<=0.5.
  3. выбор лучшего среди прошедших: результат ступени 2 бинарен, поэтому при
     равенстве решает q_ext как тай-брейкер.

Про утечку (§B.2). Приёмка обязана идти по ассерту, НЕ входящему в оценку,
иначе отбор идёт по той же величине, по которой считается результат. Отсюда
`split_tests`: test_0 → приёмка, остальные → оценка.

Офлайн-проверка на текстах sh2_200 (lab/acceptance_offline.py):
    f2 0.340 → 0.041 [0.016, 0.101],  solved 32.0 % → 46.5 %,
    спасено 30 задач, сломана 1.
"""
from __future__ import annotations

import ast
import logging
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Callable, List, Optional

log = logging.getLogger(__name__)

CODE_FENCE = re.compile(r'```(?:python|py)?\s*(.*?)```',
                        re.IGNORECASE | re.DOTALL)
TEST_DEF = re.compile(r'^def (test_\w+)\s*\(', re.MULTILINE)
PYTEST_LINE = re.compile(r'::(test_\w+)\s+(PASSED|FAILED|ERROR)')


def extract_code(text: str) -> str:
    """Идентично core/scorer.py::_extract_python_code."""
    blocks = CODE_FENCE.findall(text or '')
    return '\n\n'.join(blocks).strip() if blocks else (text or '').strip()


def split_tests(tests_src: str, accept_index: int = 0):
    """Разрезать файл тестов на приёмочную и оценочную части.

    Возвращает (accept_src, eval_src, accept_names, eval_names).
    Если тест один — приёмка пустая: расщеплять нечего, и гейт вырождается
    в ступень 1, а не начинает подглядывать в оценку.
    """
    names = TEST_DEF.findall(tests_src or '')
    if len(names) < 2:
        return '', tests_src, [], names

    starts = [m.start() for m in TEST_DEF.finditer(tests_src)]
    header = tests_src[:starts[0]]
    bodies = [tests_src[s:(starts[i + 1] if i + 1 < len(starts) else len(tests_src))]
              for i, s in enumerate(starts)]

    idx = max(0, min(accept_index, len(bodies) - 1))
    accept_src = header + bodies[idx]
    eval_src = header + ''.join(b for i, b in enumerate(bodies) if i != idx)
    return (accept_src, eval_src, [names[idx]],
            [n for i, n in enumerate(names) if i != idx])


@dataclass
class CandidateProbe:
    """Что гейт узнал про одного кандидата."""
    importable: bool = False
    accepted: bool = False
    status: str = 'unknown'
    q_ext: float = 0.0


@dataclass
class GateResult:
    chosen: int                       # индекс выбранного кандидата
    stage: str                        # на какой ступени решилось
    probes: List[CandidateProbe] = field(default_factory=list)
    n_stage1: int = 0
    n_stage2: int = 0
    # что выбрали бы прежние правила — для сравнения ВНУТРИ прогона (§D)
    would_first: int = 0              # s_ok[0]
    would_qext: int = 0               # argmax q_ext


class SandboxProbe:
    """Ступени 1 и 2 в песочнице. Приёмочные тесты приходят снаружи.

    В продакшене эталонных тестов нет, и сюда подставляется другой
    проверяющий; порог для него — доля ложных приёмов ниже 0.10
    (DECOMPOSITION §2). Здесь — лабораторный вариант.
    """

    def __init__(self, accept_tests: str = '', timeout_import: int = 5,
                 timeout_tests: int = 10):
        self.accept_tests = accept_tests or ''
        self.timeout_import = timeout_import
        self.timeout_tests = timeout_tests
        self.allowed = (sys.platform != 'win32'
                        or os.getenv('MYCELIUM_ALLOW_WINDOWS_CODE_EXEC') == '1')

    def __call__(self, text: str) -> CandidateProbe:
        code = extract_code(text)
        if not code:
            return CandidateProbe(status='empty')
        if not self.allowed:
            return CandidateProbe(importable=True, accepted=False,
                                  status='no_sandbox')
        try:
            ast.parse(code)
        except SyntaxError:
            return CandidateProbe(status='syntax_error')

        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, 'solution.py'), 'w',
                      encoding='utf-8') as f:
                f.write(code)
            try:
                r = subprocess.run(
                    [sys.executable, '-c',
                     'import sys; sys.path.insert(0,"."); import solution'],
                    capture_output=True, timeout=self.timeout_import, cwd=d)
            except subprocess.TimeoutExpired:
                return CandidateProbe(status='timeout')
            except Exception as exc:
                return CandidateProbe(status=f'error:{type(exc).__name__}')
            if r.returncode != 0:
                return CandidateProbe(status='import_error')

            # ── ступень 1 пройдена ────────────────────────────────────────
            if not self.accept_tests.strip():
                return CandidateProbe(importable=True, accepted=False,
                                      status='no_accept_test')
            with open(os.path.join(d, 'test_accept.py'), 'w',
                      encoding='utf-8') as f:
                f.write(self.accept_tests)
            try:
                tr = subprocess.run(
                    [sys.executable, '-m', 'pytest', 'test_accept.py',
                     '-v', '--tb=no', '--no-header', '-p', 'no:cacheprovider'],
                    capture_output=True, timeout=self.timeout_tests, cwd=d)
            except subprocess.TimeoutExpired:
                return CandidateProbe(importable=True, status='timeout_tests')
            except Exception as exc:
                return CandidateProbe(importable=True,
                                      status=f'error:{type(exc).__name__}')
            out = tr.stdout.decode('utf-8', errors='ignore')
            verdicts = PYTEST_LINE.findall(out)
            if not verdicts:
                return CandidateProbe(importable=True, status='no_tests_ran')
            ok = all(v == 'PASSED' for _n, v in verdicts)
            return CandidateProbe(importable=True, accepted=ok,
                                  status='accepted' if ok else 'rejected')


def select(candidates: List[str],
           q_ext: Callable[[str], float],
           probe: Optional[Callable[[str], CandidateProbe]] = None
           ) -> GateResult:
    """ЕДИНСТВЕННАЯ точка выбора ответа. Зовут и M, и BoN-G.

    candidates — тексты в порядке порождения; candidates[0] это то, что
    прежнее правило `s_ok[0]` вернуло бы в M и первый кандидат в BoN.
    """
    if not candidates:
        return GateResult(chosen=-1, stage='empty')

    qs = []
    for c in candidates:
        try:
            qs.append(float(q_ext(c)))
        except Exception:
            qs.append(0.0)
    would_qext = max(range(len(candidates)), key=lambda i: qs[i])

    if probe is None:
        # Без проверяющего гейт вырождается в прежнее правило BoN.
        return GateResult(chosen=would_qext, stage='qext_only',
                          probes=[CandidateProbe(q_ext=q) for q in qs],
                          would_first=0, would_qext=would_qext)

    probes = []
    for c, q in zip(candidates, qs):
        p = probe(c)
        p.q_ext = q
        probes.append(p)

    s1 = [i for i, p in enumerate(probes) if p.importable]
    s2 = [i for i in s1 if probes[i].accepted]

    if s2:
        pool, stage = s2, 'stage2'
    elif s1:
        pool, stage = s1, 'stage1'
    else:
        pool, stage = list(range(len(candidates))), 'fallback_all'

    chosen = max(pool, key=lambda i: qs[i])
    return GateResult(chosen=chosen, stage=stage, probes=probes,
                      n_stage1=len(s1), n_stage2=len(s2),
                      would_first=0, would_qext=would_qext)


def trace_of(res: GateResult) -> dict:
    """Компактная запись для лога шага (§D: сравнивать старую и новую
    приёмку ВНУТРИ прогона, а не между прогонами)."""
    return {
        'chosen': res.chosen, 'stage': res.stage,
        'n_candidates': len(res.probes),
        'n_stage1': res.n_stage1, 'n_stage2': res.n_stage2,
        'would_first': res.would_first, 'would_qext': res.would_qext,
        'changed_vs_first': res.chosen != res.would_first,
        'changed_vs_qext': res.chosen != res.would_qext,
        'statuses': [p.status for p in res.probes],
        'q_ext': [round(p.q_ext, 4) for p in res.probes],
    }
