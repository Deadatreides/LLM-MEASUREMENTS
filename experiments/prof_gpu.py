"""Why does this card deliver 0.5 TFLOP/s when it is rated ~5?"""

import subprocess
import threading
import time

import torch

print("torch", torch.__version__)
print("arch list:", torch.cuda.get_arch_list())
print("device:", torch.cuda.get_device_name(0),
      "capability", torch.cuda.get_device_capability(0))
p = torch.cuda.get_device_properties(0)
print(f"SMs {p.multi_processor_count}, "
      f"total {p.total_memory/1e9:.2f} GB")

stop = False


def sample():
    out = []
    while not stop:
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,power.draw,"
                 "temperature.gpu,utilization.gpu,pstate,"
                 "clocks_throttle_reasons.active",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5)
            out.append(r.stdout.strip())
        except Exception:
            pass
        time.sleep(0.4)
    print("\nclocks sampled during the load "
          "(sm, mem, power, temp, util, pstate, throttle):")
    for line in out[-8:]:
        print("   ", line)


th = threading.Thread(target=sample)
th.start()

for dtype in (torch.float16, torch.float32):
    a = torch.randn(4096, 4096, device="cuda", dtype=dtype)
    b = torch.randn(4096, 4096, device="cuda", dtype=dtype)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(30):
        a @ b
    torch.cuda.synchronize()
    dt = (time.time() - t0) / 30
    print(f"{str(dtype):20s} 4096^3 GEMM {dt*1000:7.2f} ms -> "
          f"{2*4096**3/dt/1e12:5.2f} TFLOP/s", flush=True)
    del a, b
    torch.cuda.empty_cache()

stop = True
th.join()
