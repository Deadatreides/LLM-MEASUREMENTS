"""
Manual chat-template rendering for GGUF models via llama-cpp-python.

We do NOT use Llama.create_chat_completion()'s built-in auto-detected chat
handler, for two reasons discovered during smoke testing on this machine:
  1. this llama-cpp-python version's create_chat_completion() has no way to
     pass extra Jinja template variables (e.g. Qwen3's enable_thinking), and
  2. one local GGUF conversion (SmolLM2-360M "noimpl") has no embedded
     tokenizer.chat_template at all.

Instead we render the prompt string ourselves with jinja2, using each
model's own embedded template (extracted from GGUF metadata) or an explicit
fallback template, then submit it through the low-level completion API
(Llama.create_completion) with a family-appropriate stop string. This keeps
prompting behavior uniform, inspectable, and reproducible across models.
"""
import jinja2

FALLBACK_TEMPLATES = {
    # SmolLM2's own ChatML template (taken verbatim from the bartowski
    # conversion of the same nominal model), used for the "noimpl"
    # conversion that ships without any embedded chat template.
    "smollm2-360m-instruct-noimpl-q5_k_m": (
        "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}"
        "{{ '<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n' }}"
        "{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
        "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    ),
}

STOP_SEQUENCES = {
    "llama-3.2-1b-instruct-q4_0": ["<|eot_id|>"],
    "qwen2.5-0.5b-instruct-q5_0": ["<|im_end|>"],
    "smollm2-1.7b-instruct-q4_k_m": ["<|im_end|>"],
    "smollm2-360m-instruct-noimpl-q5_k_m": ["<|im_end|>"],
    "smollm2-360m-instruct-bartowski-q5_k_m": ["<|im_end|>"],
    "gemma-3-1b-it-q5_k_s": ["<end_of_turn>"],
    "qwen2.5-coder-1.5b-instruct-q4_0": ["<|im_end|>"],
    "qwen3-1.7b-q4_0-unsloth": ["<|im_end|>"],
}


def _raise_exception(msg):
    raise ValueError(msg)


def _token_text(llm, token_id):
    if token_id is None:
        return ""
    try:
        return llm.detokenize([int(token_id)]).decode("utf-8", "ignore")
    except Exception:
        return ""


def render_prompt(llm, model_id, messages, add_generation_prompt=True, enable_thinking=None):
    template_str = llm.metadata.get("tokenizer.chat_template") or FALLBACK_TEMPLATES.get(model_id)
    if template_str is None:
        raise ValueError(f"No chat template available for {model_id}")

    bos = _token_text(llm, llm.metadata.get("tokenizer.ggml.bos_token_id"))
    eos = _token_text(llm, llm.metadata.get("tokenizer.ggml.eos_token_id"))

    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = _raise_exception
    tmpl = env.from_string(template_str)

    kwargs = dict(
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        bos_token=bos,
        eos_token=eos,
    )
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    return tmpl.render(**kwargs)
