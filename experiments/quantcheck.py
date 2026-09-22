"""Which MoE architectures bitsandbytes can actually 4-bit quantise.

bnb replaces nn.Linear modules.  transformers 5.x refactored several MoE
implementations to store experts as fused 3D Parameters, which bnb leaves
untouched -- so `load_in_4bit=True` silently keeps the bulk of the model in
fp32/fp16 and the memory budget is unchanged.  This decides the size of model
that can fit in 6 GB, so it is measured rather than assumed.

Tiny models are built from config, saved, and reloaded quantised.  No download.
"""

import shutil
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

COMMON = dict(vocab_size=256, hidden_size=64, num_hidden_layers=2,
              num_attention_heads=4, num_key_value_heads=2,
              max_position_embeddings=128)


def build(name):
    import transformers as tf
    if name == "olmoe":
        return tf.OlmoeForCausalLM(tf.OlmoeConfig(
            intermediate_size=128, num_experts=8, num_experts_per_tok=2, **COMMON))
    if name == "qwen3_moe":
        return tf.Qwen3MoeForCausalLM(tf.Qwen3MoeConfig(
            intermediate_size=128, moe_intermediate_size=64, num_experts=8,
            num_experts_per_tok=2, head_dim=16, **COMMON))
    if name == "qwen2_moe":
        return tf.Qwen2MoeForCausalLM(tf.Qwen2MoeConfig(
            intermediate_size=128, moe_intermediate_size=64,
            shared_expert_intermediate_size=64, num_experts=8,
            num_experts_per_tok=2, **COMMON))
    if name == "granitemoe":
        return tf.GraniteMoeForCausalLM(tf.GraniteMoeConfig(
            intermediate_size=128, num_local_experts=8, num_experts_per_tok=2,
            **COMMON))
    if name == "mixtral":
        return tf.MixtralForCausalLM(tf.MixtralConfig(
            intermediate_size=128, num_local_experts=8, num_experts_per_tok=2,
            **COMMON))
    if name == "phimoe":
        return tf.PhimoeForCausalLM(tf.PhimoeConfig(
            intermediate_size=128, num_local_experts=8, num_experts_per_tok=2,
            **COMMON))
    if name == "gpt_oss":
        return tf.GptOssForCausalLM(tf.GptOssConfig(
            intermediate_size=128, num_local_experts=8, num_experts_per_tok=2,
            head_dim=16, **COMMON))
    if name == "lfm2_moe":
        return tf.Lfm2MoeForCausalLM(tf.Lfm2MoeConfig(
            intermediate_size=128, num_experts=8, num_experts_per_tok=2, **COMMON))
    raise KeyError(name)


def expert_params(model):
    """Parameters that live in the MoE experts, by name."""
    out = []
    for n, p in model.named_parameters():
        if "expert" in n.lower() and "gate" not in n.split(".")[-1]:
            out.append((n, p))
    return out


def check(name):
    try:
        m = build(name)
    except Exception as e:
        return f"{name:14s} build failed: {type(e).__name__}: {str(e)[:60]}"
    d = Path(tempfile.mkdtemp())
    try:
        m.save_pretrained(d)
        del m
        qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.float16,
                                bnb_4bit_use_double_quant=True)
        q = AutoModelForCausalLM.from_pretrained(d, quantization_config=qc,
                                                 device_map={"": 0})
        eps = expert_params(q)
        n_lin = sum(1 for _, mod in q.named_modules()
                    if type(mod).__name__ == "Linear4bit")
        if not eps:
            # fused experts may be modules rather than named parameters
            info = "no expert parameters found by name"
        else:
            dtypes = {str(p.dtype) for _, p in eps}
            n_el = sum(p.numel() for _, p in eps)
            quantised = all(str(p.dtype) == "torch.uint8" for _, p in eps)
            info = (f"expert params {n_el:>8d} el, dtype {'/'.join(sorted(dtypes))}"
                    f" -> {'QUANTISED' if quantised else 'NOT quantised'}")
        ids = torch.randint(0, 200, (1, 6)).cuda()
        q(input_ids=ids)
        return f"{name:14s} Linear4bit modules {n_lin:3d};  {info}"
    except Exception as e:
        return f"{name:14s} FAILED: {type(e).__name__}: {str(e)[:80]}"
    finally:
        shutil.rmtree(d, ignore_errors=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    for name in ("olmoe", "qwen3_moe", "qwen2_moe", "granitemoe", "mixtral",
                 "phimoe", "gpt_oss", "lfm2_moe"):
        print(check(name), flush=True)
