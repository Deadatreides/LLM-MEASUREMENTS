"""Direct check of the MoE families worth considering, by name.

Keyword search on the hub is unreliable for this (it is dominated by test
stubs and re-quantised mirrors), so the candidates that matter are checked
explicitly.  Metadata and config.json only -- no weights.

Reported per model:
  GB fp16          what has to fit in VRAM, since bnb cannot quantise experts
  active/layer     top_k / n_experts; the GPT-OSS-120B target is 3.1 %
  M_iso            budget reproducing the target's cache/activation ratio,
                   = 1.60 * top_k / n_experts -- 5 % for the target itself
"""

import json
import time

from huggingface_hub import HfApi, hf_hub_download

api = HfApi()

CANDIDATES = [
    "allenai/OLMoE-1B-7B-0924",
    "allenai/OLMoE-1B-7B-0125",
    "allenai/OLMoE-1B-7B-0924-Instruct",
    "ibm-granite/granite-3.0-1b-a400m-base",
    "ibm-granite/granite-3.1-1b-a400m-base",
    "ibm-granite/granite-3.0-3b-a800m-base",
    "ibm-granite/granite-3.1-3b-a800m-base",
    "ibm-granite/granite-4.0-h-tiny",
    "ibm-granite/granite-4.0-h-micro",
    "Qwen/Qwen1.5-MoE-A2.7B",
    "Qwen/Qwen3-30B-A3B",
    "LiquidAI/LFM2-8B-A1B",
    "LiquidAI/LFM2.5-8B-A1B",
    "deepseek-ai/deepseek-moe-16b-base",
    "microsoft/Phi-3.5-MoE-instruct",
    "openai/gpt-oss-20b",
    "NousResearch/moe-10b-a1b-8k-wsd-lr3e4-1t",
    "jetmoe/jetmoe-8b",
    "TroyDoesAI/Qwen3-15B-A2B-Base",
]

USABLE_GB = 4.9          # 6 GB card minus desktop and activation headroom
RAM_GB = 13.5            # free system RAM measured on this machine


def look(mid, attempts=6):
    """Anonymous hub access is capped at 500 requests per 300 s; back off."""
    for a in range(attempts):
        try:
            info = api.model_info(mid, files_metadata=True)
            gb = sum(f.size or 0 for f in info.siblings
                     if f.rfilename.endswith(".safetensors")) / 1e9
            cfg = json.loads(open(hf_hub_download(mid, "config.json"),
                                  encoding="utf-8").read())
            break
        except Exception as e:
            msg = str(e)
            if "429" in msg and a < attempts - 1:
                wait = 60
                for tok in msg.split():
                    if tok.isdigit() and 5 < int(tok) < 400:
                        wait = int(tok) + 5
                        break
                print(f"  rate limited on {mid}, waiting {wait}s", flush=True)
                time.sleep(wait)
                continue
            return dict(id=mid, err=f"{type(e).__name__}: {msg[:60]}")
    ne = cfg.get("num_experts") or cfg.get("num_local_experts") \
        or cfg.get("n_routed_experts")
    tk = cfg.get("num_experts_per_tok")
    if not ne or not tk:
        return dict(id=mid, err="not a routed MoE config")
    return dict(id=mid, mt=cfg.get("model_type"), gb=gb,
                layers=cfg.get("num_hidden_layers"),
                hidden=cfg.get("hidden_size"), experts=ne, top_k=tk,
                sparsity=tk / ne, iso_M=100 * 1.60 * tk / ne,
                units=cfg.get("num_hidden_layers", 0) * ne,
                active=cfg.get("num_hidden_layers", 0) * tk)


def verdict(r):
    if r["gb"] <= USABLE_GB:
        return "GPU fp16"
    if r["gb"] <= RAM_GB:
        return "CPU fp16 / GPU+offload"
    return "does not fit"


if __name__ == "__main__":
    rows = [look(m) for m in CANDIDATES]      # sequential, to stay under the cap
    ok = [r for r in rows if "err" not in r]
    bad = [r for r in rows if "err" in r]
    ok.sort(key=lambda r: (r["sparsity"], r["gb"]))

    hdr = (f"{'model':40s} {'arch':12s} {'GB':>6s} {'L':>3s} {'exp':>4s} "
           f"{'top':>3s} {'act/l':>6s} {'M_iso':>6s} {'units':>6s} "
           f"{'act/tok':>7s}  fit")
    print(hdr)
    print("-" * len(hdr))
    for r in ok:
        print(f"{r['id'][:40]:40s} {str(r['mt'])[:12]:12s} {r['gb']:6.2f} "
              f"{r['layers']:3d} {r['experts']:4d} {r['top_k']:3d} "
              f"{100*r['sparsity']:5.1f}% {r['iso_M']:5.1f}% {r['units']:6d} "
              f"{r['active']:7d}  {verdict(r)}")
    print(f"\n{'GPT-OSS-120B (target)':40s} {'gpt_oss':12s} "
          f"{'~65':>6s} {36:3d} {128:4d} {4:3d} {3.1:5.1f}% {5.0:5.1f}% "
          f"{4608:6d} {144:7d}")
    for r in bad:
        print(f"\n{r['id']}: {r['err']}")
