"""Where does part C spend its time? Measured, not guessed."""

import sys
import time

import torch

sys.argv = ["prof"]
import needle          # noqa: E402
import probe           # noqa: E402

torch.set_grad_enabled(False)

t0 = time.time()
tok, model, cfg = probe.load("B")
print(f"load model            {time.time()-t0:6.2f} s", flush=True)

paras = needle.make_haystack()
nd = needle.make_needles()[0]
blocks, at = needle.insert(paras, nd["sentence"], 50)
full = "\n\n".join(blocks)

t0 = time.time()
js = [needle.negentropy(b, nd["question"])[0] for b in blocks]
print(f"negentropy {len(blocks):3d} blocks {time.time()-t0:6.2f} s", flush=True)


def ntok(s):
    return len(tok(s, add_special_tokens=False)["input_ids"])


t0 = time.time()
n_full = ntok(full)
print(f"tokenize full ({n_full} tok) {time.time()-t0:6.2f} s", flush=True)

t0 = time.time()
picked, used = needle.select(blocks, js, 2000, ntok)
print(f"select ({len(picked)} blocks, {used} tok) {time.time()-t0:6.2f} s", flush=True)

for name, ctx in (("A full ", full),
                  ("B 2000 ", "\n\n".join(blocks[i] for i in picked))):
    text = needle.PROMPT.format(ctx=ctx, q=nd["question"])
    enc = tok(text, return_tensors="pt").to("cuda")
    n = enc["input_ids"].shape[1]
    torch.cuda.synchronize()
    t0 = time.time()
    out = model.generate(**enc, max_new_tokens=24, do_sample=False,
                         pad_token_id=tok.pad_token_id or tok.eos_token_id)
    torch.cuda.synchronize()
    dt = time.time() - t0
    ans = tok.decode(out[0, n:], skip_special_tokens=True).strip()[:60]
    print(f"{name} ctx={n:5d} generate {dt:6.2f} s   answer: {ans!r}", flush=True)

    # split prefill from decode
    torch.cuda.synchronize()
    t0 = time.time()
    model(input_ids=enc["input_ids"])
    torch.cuda.synchronize()
    print(f"          prefill only  {time.time()-t0:6.2f} s", flush=True)

print("VRAM free/total GB", [round(v / 1e9, 2) for v in torch.cuda.mem_get_info()])
print("attn impl:", getattr(model.config, "_attn_implementation", "?"))
