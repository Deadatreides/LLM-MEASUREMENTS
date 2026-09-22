"""lab/bench_speed.py — фактическая скорость генерации ДО прогона (§8 задания)."""
import os, sys, time, json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
M = os.path.join(ROOT, 'models')

CASES = [
    ('Qwen3-1.7B', os.path.join(M, 'Qwen3-1.7B'), torch.float16, 'cuda'),
    ('granite-1b', os.path.join(M, 'granite-3.1-1b-a400m-base'), torch.float16, 'cuda'),
]

PROMPT_SHORT = "Write a Python function that returns the n-th Fibonacci number.\n"
PROMPT_LONG = (PROMPT_SHORT + "Context:\n" + ("lorem ipsum dolor sit amet consectetur " * 120))


def run(name, path, dtype, dev, prompt, new_tokens=128, n=3):
    tok = AutoTokenizer.from_pretrained(path)
    t0 = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(dev).eval()
    load_s = time.monotonic() - t0
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    has_tpl = getattr(tok, 'chat_template', None)
    if has_tpl:
        try:
            text = tok.apply_chat_template([{'role': 'user', 'content': prompt}],
                                           tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template([{'role': 'user', 'content': prompt}],
                                           tokenize=False, add_generation_prompt=True)
    else:
        text = prompt
    enc = tok(text, return_tensors='pt').to(dev)
    n_in = enc['input_ids'].shape[1]
    times = []
    outs = []
    for i in range(n):
        t0 = time.monotonic()
        with torch.inference_mode():
            o = model.generate(**enc, max_new_tokens=new_tokens, do_sample=True,
                               temperature=0.7, top_p=0.95,
                               pad_token_id=tok.pad_token_id)
        times.append(time.monotonic() - t0)
        outs.append(int(o.shape[1] - n_in))
    # swap timing
    t0 = time.monotonic(); model.to('cpu'); torch.cuda.empty_cache(); off = time.monotonic() - t0
    t0 = time.monotonic(); model.to('cuda'); on = time.monotonic() - t0
    t0 = time.monotonic(); model.to('cpu'); torch.cuda.empty_cache(); time.monotonic() - t0
    del model
    torch.cuda.empty_cache()
    tps = sum(outs) / sum(times)
    return dict(name=name, n_in=int(n_in), load_s=round(load_s, 1),
                gen_s=[round(t, 2) for t in times], n_out=outs,
                tok_per_s=round(tps, 2), swap_off_s=round(off, 2), swap_on_s=round(on, 2))


if __name__ == '__main__':
    res = []
    for name, path, dtype, dev in CASES:
        if not os.path.isdir(path):
            print(f"skip {name}: not found"); continue
        for tag, p in (('short', PROMPT_SHORT), ('long', PROMPT_LONG)):
            r = run(name, path, dtype, dev, p)
            r['prompt'] = tag
            print(json.dumps(r), flush=True)
            res.append(r)
    print('---')
    print(json.dumps(res, indent=2))
