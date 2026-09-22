"""Work around the fp16 GEMM penalty of the GTX 1660 Super (TU116).

Measured on this card, 4096^3 GEMM:

    fp16 x fp16                    244.9 ms    0.56 TFLOP/s
    fp32 x fp32                     28.1 ms    4.89 TFLOP/s
    cast fp16->fp32, gemm, cast     29.5 ms    4.66 TFLOP/s

TU116 has no tensor cores and cuBLAS takes a pathological path for half
precision, so fp16 is 8.6x SLOWER than fp32 rather than faster.  fp32 storage
does not fit (8.1 GB for a 1.7B model against 6.4 GB of VRAM), but fp32
*compute* over fp16 *storage* recovers essentially all of it.

Only worth it when the matmul is compute-bound.  For single-token decode the
GEMV is memory-bound and casting the weight would cost more than the operation
itself, so the patch applies only above a token threshold.

This changes numerics slightly -- fp32 accumulation is more accurate than fp16 --
so it must not be mixed into a comparison against traces collected without it.
"""

import torch
import torch.nn.functional as F

_orig_linear = F.linear
_installed = False


def _fast_linear(input, weight, bias=None):
    if (input.dtype == torch.float16 and weight.dtype == torch.float16
            and input.ndim >= 2
            and input.numel() // input.shape[-1] >= _fast_linear.threshold):
        out = _orig_linear(input.float(), weight.float(),
                           None if bias is None else bias.float())
        return out.to(input.dtype)
    return _orig_linear(input, weight, bias)


_fast_linear.threshold = 32


def install(threshold=32):
    """Route large fp16 linear layers through fp32 compute."""
    global _installed
    _fast_linear.threshold = threshold
    if not _installed:
        F.linear = _fast_linear
        _installed = True
    return _installed


def uninstall():
    global _installed
    if _installed:
        F.linear = _orig_linear
        _installed = False


if __name__ == "__main__":
    import time

    import probe

    torch.set_grad_enabled(False)
    tok, model, cfg = probe.load("B")
    for use in (False, True):
        uninstall()
        if use:
            install()
        for n in (1024, 4096):
            ids = torch.randint(0, 1000, (1, n), device="cuda")
            model(input_ids=ids)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(2):
                model(input_ids=ids)
            torch.cuda.synchronize()
            dt = (time.time() - t0) / 2
            print(f"patch={use!s:5s} n={n:5d}  {dt*1000:8.1f} ms  "
                  f"{n/dt:8.0f} tok/s", flush=True)
    uninstall()
