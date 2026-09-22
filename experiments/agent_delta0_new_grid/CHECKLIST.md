# CHECKLIST DELTA-0 (исполнять сверху вниз, не прыгать)

- [x] C0 PROTOCOL.md написан (формулы Δ, B, пороги, без цифр прогона)
- [x] C1 Выбран и описан класс задач T (критерии §2), 0 кода моделей
- [x] C2 Реализован generator: ровно N_TRAIN+N_TEST задач, seed зафиксирован
- [x] C3 У каждой задачи: final_oracle + ≥2 intermediate checkpoints с локальным v
- [x] C4 self_test generator: 5 фикстур, все PASS
- [x] C5 Бюджет B зафиксирован в PROTOCOL (токены in+out на задачу для single)
- [x] C6 Замерен r_m*(B) whole для КАЖДОЙ модели registry (таблица)
- [x] C7 m* выбран = argmax r_m на TEST при cost≤B
- [x] C8 Определён набор атомов A (≥3 типов) с локальным v и парсером шва
- [x] C9 Замерен lower-bound r_∪: oracle-routed атомы / перебор планов ограниченный
- [x] C10 Δ = r_∪ - r_m* на TEST посчитан
- [x] C11 Корзина §7 записана механически
- [x] C12 REPORT_DELTA0.md + Next из §7
- [x] C13 CHECKLIST все [x], step_log.jsonl полон

Правило: нельзя начинать C{k+1}, пока C{k} не [x] и артефакт на диске.
