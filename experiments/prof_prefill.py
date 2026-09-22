"""Prefill throughput against context length.

94 s to prefill 4194 tokens on a 1.7B model is ~30x off what the card should
do.  Linear scaling would point at the matmuls, quadratic at attention.
"""

import time

import torch

import probe

torch.set_grad_enabled(False)
tok, model, cfg = probe.load("B")
print("attn:", getattr(model.config, "_attn_implementation", "?"),
      " dtype:", next(model.parameters()).dtype, flush=True)

for n in (256, 512, 1024, 2048, 4096):
    ids = torch.randint(0, 1000, (1, n), device="cuda")
    model(input_ids=ids)                       # warm up kernels
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(3):
        model(input_ids=ids)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / 3
    print(f"  n={n:5d}  {dt*1000:8.1f} ms   {n/dt:8.0f} tok/s", flush=True)

print("\nsame, batched 8 x 512 (same token count as 1 x 4096):", flush=True)
ids = torch.randint(0, 1000, (8, 512), device="cuda")
model(input_ids=ids)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(3):
    model(input_ids=ids)
torch.cuda.synchronize()
dt = (time.time() - t0) / 3
print(f"  {dt*1000:8.1f} ms   {8*512/dt:8.0f} tok/s", flush=True)

print("\nraw matmul reference (fp16, 2048x2048 @ 2048xN):", flush=True)
a = torch.randn(4096, 2048, device="cuda", dtype=torch.float16)
b = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(20):
    a @ b
torch.cuda.synchronize()
dt = (time.time() - t0) / 20
print(f"  fp16 {dt*1000:6.2f} ms -> {2*4096*2048*2048/dt/1e12:5.2f} TFLOP/s",
      flush=True)
a32, b32 = a.float(), b.float()
torch.cuda.synchronize()
t0 = time.time()
for _ in range(20):
    a32 @ b32
torch.cuda.synchronize()
dt = (time.time() - t0) / 20
print(f"  fp32 {dt*1000:6.2f} ms -> {2*4096*2048*2048/dt/1e12:5.2f} TFLOP/s",
      flush=True)
