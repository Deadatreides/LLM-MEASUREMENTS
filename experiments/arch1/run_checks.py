"""run_checks.py — единая точка «прогнать все контрольные случаи».

Обнаруживает контрольные случаи в tests/ (unittest.TestCase) и
запускает их все. На этапе 0 модулей ещё нет, поэтому 0 проверок,
0 провалов — это ожидаемый, а не пустой результат.

Использование:
    python run_checks.py
"""

import sys
import unittest


def main() -> int:
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir="tests", pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)

    total = result.testsRun
    failed = len(result.failures) + len(result.errors)
    print(f"{total} проверок, {failed} провалов")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
