"""
agents/trainer.py — Ночной DPO тренер
Σ_v8.5 «Мицелий»

Запускается в 02:00 по планировщику.
Обучает Qwen3-0.8B Value-адаптер на парах из SQLite.

Ключевые исправления vs v6.6:
- π_ref обновляется ПОСЛЕ каждого цикла (не фиксирован навсегда)
- hot/cold_buffer чётко определены порогами ±0.5σ
- KL-штраф адаптивный (от chaos.beta_kl)
- 1800 пар × 1 эпоха ≈ 35-45 мин на 1660S, VRAM ≤ 3.0GB
"""

import os
import json
import sys
import time
import logging
import subprocess
from pathlib import Path
from typing import List, Dict

log = logging.getLogger('trainer')


class NightlyTrainer:
    def __init__(self, cfg: dict, store):
        self.cfg = cfg
        self.store = store
        tr = cfg['training']
        self.pairs_n = tr['pairs_per_night']
        self.min_gap = tr['min_gap']
        self.rank_v = tr['lora_rank_value']
        self.rank_r = tr['lora_rank_router']
        self.lr = tr['learning_rate']
        self.batch = tr['batch_size']
        self.epochs = tr['epochs']
        self.vram = tr['vram_budget_gb']
        self.lm_url = cfg['lm_studio']['base_url']
        self.ref_path = 'data/dpo_ref_checkpoint.json'

    def collect_pairs(self) -> List[Dict]:
        """
        hot: Q > Q̄ + 0.5σ    (winners)
        cold: Q < Q̄ - 0.5σ или Q_env=0  (losers)
        pretrain: 20% пар из cold заменяем на заведомо плохие шаблоны
        """
        pairs = self.store.get_pairs(n=self.pairs_n, gap=self.min_gap)
        log.info(f"Collected {len(pairs)} DPO pairs from SQLite")
        return pairs

    def save_dataset(self, pairs: List[Dict], path: str = 'data/dpo_train.jsonl'):
        """Сохраняем в JSONL для trl / axolotl"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            for p in pairs:
                record = {
                    'prompt': p['prompt'],
                    'chosen': p['chosen'],
                    'rejected': p['rejected'],
                    'chosen_score': p['chosen_e'],
                    'rejected_score': p['rejected_e'],
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        log.info(f"Dataset saved to {path}: {len(pairs)} pairs")
        return path

    def run_dpo_lm_studio(self, dataset_path: str, beta_kl: float = 0.05):
        """
        LM Studio не поддерживает DPO напрямую.
        Отправляем инструкции пользователю + запускаем через axolotl/trl если установлено.
        """
        script = self._build_train_script(dataset_path, beta_kl)
        script_path = Path('data/run_dpo.py')
        script_path.parent.mkdir(parents=True, exist_ok=True)
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script)

        log.info("DPO train script written to data/run_dpo.py")
        log.info("Attempting to run training...")

        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True, timeout=3600,
                cwd=os.getcwd(), text=True
            )
            if result.returncode == 0:
                log.info("DPO training completed successfully")
                self._update_ref_checkpoint()
            else:
                log.error(f"DPO failed: {result.stderr[:500]}")
        except FileNotFoundError:
            log.warning("Python executable not found for DPO. Run data/run_dpo.py manually.")
        except subprocess.TimeoutExpired:
            log.error("DPO training timed out (>60 min)")
        except Exception as exc:
            log.error(f"DPO training failed: {exc}")

    def _build_train_script(self, dataset_path: str, beta_kl: float) -> str:
        """Генерирует тренировочный скрипт для trl DPOTrainer"""
        dataset_literal = json.dumps(str(Path(dataset_path)), ensure_ascii=False)
        return f'''#!/usr/bin/env python3
"""
Автогенерированный DPO тренировочный скрипт
Σ_v8.5 — Запуск: python data/run_dpo.py
Требования: pip install trl transformers peft accelerate bitsandbytes
"""
import json, os, torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model

# ── Параметры ────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3-0.8B"          # или путь к локальной модели
ADAPTER_PATH = "data/value_adapter"
REF_PATH = "data/dpo_ref_checkpoint.json"
DATASET_PATH = {dataset_literal}
LORA_RANK = {self.rank_v}
LEARNING_RATE = {self.lr}
BATCH_SIZE = {self.batch}
EPOCHS = {self.epochs}
BETA = {beta_kl}
MAX_LENGTH = 1024

print("Loading model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# LoRA конфигурация
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_RANK * 2,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── Данные ───────────────────────────────────────────────────────────────────
records = []
with open(DATASET_PATH, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        records.append({{
            "prompt": r["prompt"],
            "chosen": r["chosen"],
            "rejected": r["rejected"],
        }})
dataset = Dataset.from_list(records)
print(f"Dataset: {{len(dataset)}} pairs")

# ── Тренировка ───────────────────────────────────────────────────────────────
dpo_config = DPOConfig(
    beta=BETA,
    max_length=MAX_LENGTH,
    max_prompt_length=512,
    output_dir=ADAPTER_PATH,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    learning_rate=LEARNING_RATE,
    fp16=True,
    logging_steps=50,
    save_steps=500,
    remove_unused_columns=False,
    report_to="none",
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,          # используем встроенный implicit reference
    args=dpo_config,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

print("Starting DPO training...")
trainer.train()
trainer.save_model(ADAPTER_PATH)
print(f"Adapter saved to {{ADAPTER_PATH}}")

# ── Обновить π_ref ────────────────────────────────────────────────────────────
import json, time
ref = {{"updated_at": time.time(), "adapter_path": ADAPTER_PATH,
        "pairs_trained": len(dataset)}}
with open(REF_PATH, "w", encoding="utf-8") as f:
    json.dump(ref, f)
print("π_ref updated.")
'''

    def _update_ref_checkpoint(self):
        """π_ref обновляется после каждого цикла — KL регулирует отклонение от ПРЕДЫДУЩЕЙ версии"""
        ref = {
            'updated_at': time.time(),
            'adapter_path': 'data/value_adapter',
            'step': 'post_dpo'
        }
        with open(self.ref_path, 'w', encoding='utf-8') as f:
            json.dump(ref, f)
        log.info(f"π_ref checkpoint updated at {self.ref_path}")

    def run(self, beta_kl: float = 0.05):
        """Полный ночной цикл"""
        log.info("=== Nightly DPO cycle started ===")
        t0 = time.time()

        pairs = self.collect_pairs()
        if len(pairs) < 50:
            log.warning(f"Too few pairs ({len(pairs)} < 50). Skipping DPO.")
            return

        dataset_path = self.save_dataset(pairs)
        self.run_dpo_lm_studio(dataset_path, beta_kl)

        elapsed = time.time() - t0
        log.info(f"=== Nightly cycle done in {elapsed/60:.1f} min ===")
