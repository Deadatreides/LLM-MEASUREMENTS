"""Контрольные случаи для каждого шва из src/seams/.

Для КАЖДОГО шва: заведомо верный вход, заведомо неверный вход,
неприменимый вход (WORK_PLAN.md, Этап 3). Шов, не прошедший эти
случаи, помечается is_trusted = false и не может давать HARD-evidence.

Реализовано на этапе 3: по файлу на шов (test_exec_seam.py,
test_format_seam.py, test_numeric_seam.py) + межмодульные гарантии
SeamEngine в test_seam_engine.py.
"""
