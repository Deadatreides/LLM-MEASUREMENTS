### B (`none`) vs B2 (`nf4`) — Qwen/Qwen3-1.7B

Band size s = 48 (128 bands of 6144).

| metric | B | B2 | tolerance (§8) | |
|---|---|---|---|---|
| q (EPOCH, steady), worst over all M | worst at M=50% | 0.06 pp | < 3.0 pp | PASS |
| L_persist | None | None | equal | PASS |
| conditional entropy, bits per layer | 6.011 | 6.023  (0.2 %) | < 15 % | PASS |
| N_eff, literal §8 form | 2^162.30 | 2^162.63 | < 2x (got 1.3x) — not a usable criterion at this magnitude | — |
| unique units | 3584 | 3584  (0.0 %) | < 10 % | PASS |

**Direct check on the units themselves** (same token sequence replayed, so the active sets line up one to one): fraction of B's selected units that B2 also selects — mean 0.9545, worst layer 0.9376 (layer 27), best 0.9715.

| layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| agreement | 0.971 | 0.969 | 0.965 | 0.963 | 0.961 | 0.961 | 0.959 | 0.958 | 0.956 | 0.956 | 0.956 | 0.955 | 0.954 | 0.951 | 0.950 | 0.952 | 0.955 | 0.953 | 0.952 | 0.955 | 0.953 | 0.950 | 0.946 | 0.943 | 0.945 | 0.949 | 0.950 | 0.938 |

This is the sharp form of the §8 question. The aggregate rows above can agree for the wrong reason when the working set is close to the whole model; this row cannot.

**Все метрики в допуске — определение единицы устойчиво к квантованию.**

### G (`none`) vs G2 (`simnf4`) — ibm-granite/granite-3.1-1b-a400m-base

| metric | G | G2 | tolerance (§8) | |
|---|---|---|---|---|
| q (EPOCH, steady), worst over all M | worst at M=30% | 0.19 pp | < 3.0 pp | PASS |
| L_persist | 1 | 1  (0.0 %) | < 15 % | PASS |
| conditional entropy, bits per layer | 2.823 | 2.808  (0.5 %) | < 15 % | PASS |
| N_eff, literal §8 form | 2^64.94 | 2^64.59 | < 2x (got 1.3x) — not a usable criterion at this magnitude | — |
| unique units | 768 | 768  (0.0 %) | < 10 % | PASS |

**Direct check on the units themselves** (same token sequence replayed, so the active sets line up one to one): fraction of G's selected units that G2 also selects — mean 0.9415, worst layer 0.9265 (layer 21), best 0.9632.

| layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| agreement | 0.957 | 0.963 | 0.947 | 0.949 | 0.952 | 0.930 | 0.934 | 0.941 | 0.928 | 0.928 | 0.936 | 0.945 | 0.945 | 0.950 | 0.941 | 0.949 | 0.949 | 0.938 | 0.935 | 0.942 | 0.929 | 0.926 | 0.930 | 0.951 |

This is the sharp form of the §8 question. The aggregate rows above can agree for the wrong reason when the working set is close to the whole model; this row cannot.

**Все метрики в допуске — определение единицы устойчиво к квантованию.**
