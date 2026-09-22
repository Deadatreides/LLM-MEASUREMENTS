"""run_checks.py — все контрольные случаи слоя. Та же роль, что arch1/run_checks.py.

Модуль без пройденных контрольных случаев считается неготовым (норма проекта).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    loader = unittest.TestLoader()
    suite = loader.discover(str(ROOT / "tests"), pattern="test_*.py", top_level_dir=str(ROOT))
    result = unittest.TextTestRunner(verbosity=2).run(suite)

    # Гейт доверия швов — отдельной строкой, потому что это I-15, а не обычный тест.
    import registry

    reg = registry.default_registry()
    print("\nI-15 gate (seam self-tests):")
    all_ok = True
    for sid, rep in reg.self_test_report().items():
        mark = "OK  " if rep["passed"] else "FAIL"
        print(f"  [{mark}] {sid}: {rep['n_cases']} control cases")
        for f in rep["failures"]:
            print(f"          - {f}")
        all_ok = all_ok and rep["passed"]

    total = result.testsRun
    failed = len(result.failures) + len(result.errors)
    print(f"\n{total - failed}/{total} control cases passed; seam gate: {'OK' if all_ok else 'FAILED'}")
    return 0 if (failed == 0 and all_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
