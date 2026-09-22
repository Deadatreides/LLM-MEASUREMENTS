"""Can the fp16 GEMM penalty on TU116 be worked around?

fp32 storage does not fit the card for a 1.7B model, so the question is
whether fp16 *storage* with fp32 *compute* recovers the throughput.
"""

import time

import torch

N = 4096


def bench(fn, n=20, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n


a16 = torch.randn(N, N, device="cuda", dtype=torch.float16)
b16 = torch.randn(N, N, device="cuda", dtype=torch.float16)
a32, b32 = a16.float(), b16.float()
flop = 2 * N ** 3

cases = {
    "fp16 x fp16 -> fp16": lambda: a16 @ b16,
    "fp32 x fp32 -> fp32": lambda: a32 @ b32,
    "cast fp16->fp32, gemm, cast back": lambda: (a16.float() @ b16.float()).half(),
    "fp16 act x fp32 weight (cast w only)": lambda: a16.float() @ b32,
}
for name, fn in cases.items():
    dt = bench(fn)
    print(f"{name:40s} {dt*1000:8.2f} ms  {flop/dt/1e12:5.2f} TFLOP/s", flush=True)

print()
try:
    torch.backends.cuda.preferred_blas_library("cublaslt")
    dt = bench(lambda: a16 @ b16)
    print(f"{'fp16 with cublaslt':40s} {dt*1000:8.2f} ms  "
          f"{flop/dt/1e12:5.2f} TFLOP/s", flush=True)
except Exception as e:
    print("cublaslt preference unavailable:", type(e).__name__, e)

try:
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    dt = bench(lambda: a16 @ b16)
    print(f"{'fp16, no reduced-precision reduce':40s} {dt*1000:8.2f} ms  "
          f"{flop/dt/1e12:5.2f} TFLOP/s", flush=True)
except Exception as e:
    print("flag unavailable:", type(e).__name__, e)

# memory cost of keeping a 1.7B model in each dtype
for name, nbytes in (("fp16", 2), ("fp32", 4)):
    print(f"Qwen3-1.7B weights in {name}: {2.03e9*nbytes/1e9:.2f} GB "
          f"(card has {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB)")
