# Path C — product spec (K8, 0 new generate() calls)

Из уже посчитанных чисел: Δ_A=0.475 (TOOL_PIPELINE), Δ_B_LLM=0.025, Δ_B_DET=1.000.

## Продукт

- **primary**: single whole `m*` на классе short structured QA (F2's m*=0.525-tier reference; T_hard's m*_B=llama-3.2-1b-instruct-q4_0, r_m*_B=0.000).

- **optional**: детерминированный tool pipeline, когда нужен точный set/sum (Path A/B-DET показали потолок ~1.0 при 0 LLM-вызовах в критическом атоме).

- **explicit non-goal**: multi-LLM HGT на 1-2B без показанного Δ>0 в критическом LLM-атоме (LLM_SWARM_HAS_SENSE=False на этом прогоне).

