# CHECKLIST FORK-1
- [x] K0 PROTOCOL.md (три пути, формулы, пороги, без цифр прогона)
- [x] K1 Переиспользовать delta0 F2 TEST (40) + артефакты read-only
- [x] K2 Путь A: det FILTER + det AGGREGATE + det/LLM DERIVE на том же TEST
- [x] K3 Путь A: r_A, cost_A, Δ_A = r_A - r_m*_delta0 записаны
- [x] K4 Путь B: новый T_hard, N_TEST≥30, final НЕ boolean-порог
- [x] K5 Путь B: r_m*_B whole по всем registry-моделям
- [x] K6 Путь B: r_∪_B (золотой план; FILTER/логика — LLM или det по PROTOCOL)
- [x] K7 Путь B: Δ_B = r_∪_B - r_m*_B
- [x] K8 Путь C: спека продукта whole±tools из фактов A/B (не новый прогон)
- [x] K9 Сводная таблица A/B/C + механическая корзина §8
- [x] K10 REPORT_FORK1.md + Next + CHECKLIST все [x]

Нельзя K{n+1} пока K{n} не [x] и файл-артефакт на диске.
