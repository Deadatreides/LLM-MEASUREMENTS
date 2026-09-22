"""Find MoE models that fit 6 GB of VRAM in fp16.

bitsandbytes cannot quantise fused MoE experts (see quantcheck.py), so the
whole model has to fit unquantised.  Only repository metadata and config.json
are fetched -- no weights.

Ranking is by how close the routing geometry is to the GPT-OSS-120B target:
36 layers x 128 experts, top-4, i.e. 3.1 % of experts active per layer.  The
budget that reproduces the target's cache/activation ratio is
    M_iso = 1.60 * top_k / n_experts
which depends only on sparsity, so a smaller top_k/n_experts is a better proxy.
"""

import json
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import HfApi

api = HfApi()

SUPPORTED = {
    "olmoe", "qwen2_moe", "qwen3_moe", "qwen3_5_moe", "granitemoe",
    "granitemoeshared", "granitemoehybrid", "mixtral", "phimoe", "jetmoe",
    "gpt_oss", "glm4_moe", "glm4_moe_lite", "lfm2_moe", "hunyuan_v1_moe",
    "ernie4_5_moe", "exaone_moe", "afmoe", "cohere2_moe", "seed_oss",
    "deepseek_v2", "deepseek_v3",
}
VRAM_GB = 6.0
USABLE_GB = 4.9          # 6.0 minus desktop and activation headroom
TARGET_SPARSITY = 4 / 128

QUERIES = ["moe", "mixture of experts", "a400m", "a800m", "a1b", "a2b", "a3b",
           "olmoe", "granitemoe", "tiny moe", "mini moe"]


def candidates():
    """Repos whose config says MoE and whose parameter count could fit."""
    seen = {}
    for q in QUERIES:
        try:
            for m in api.list_models(search=q, limit=200, sort="downloads",
                                     fetch_config=True,
                                     num_parameters={"max": 4_000_000_000}):
                seen[m.id] = m
        except Exception as e:
            print(f"  search '{q}' failed: {type(e).__name__}: {str(e)[:70]}")
    return list(seen.values())


def from_listing(m):
    cfg = getattr(m, "config", None) or {}
    mt = cfg.get("model_type", "")
    if mt not in SUPPORTED:
        return None
    ne = cfg.get("num_experts") or cfg.get("num_local_experts") \
        or cfg.get("n_routed_experts")
    tk = cfg.get("num_experts_per_tok")
    if not ne or not tk:
        return None
    return dict(id=m.id, model_type=mt, layers=cfg.get("num_hidden_layers"),
                hidden=cfg.get("hidden_size"), experts=ne, top_k=tk,
                sparsity=tk / ne, iso_M=100 * 1.60 * tk / ne,
                downloads=m.downloads or 0, gb=None)


# repos that are not usable weights for this probe
JUNK = ("tiny", "random", "-test", "test-", "testing", "debug", "dummy",
        "pocket", "minimind")
NOT_LOADABLE = ("mlx", "gguf", "awq", "gptq", "nvfp4", "mxfp4", "-4bit",
                "-8bit", "-3bit", "-5bit", "-6bit", "int4", "int8", "w8a8",
                "w4a8", "-ov", "eagle3", "quark", "abliterated")


def usable(mid):
    low = mid.lower()
    return not any(j in low for j in JUNK) and \
        not any(q in low for q in NOT_LOADABLE)


def weight_size(row):
    """Real weight size plus the config fields the listing does not carry."""
    try:
        info = api.model_info(row["id"], files_metadata=True)
        b = sum(f.size or 0 for f in info.siblings
                if f.rfilename.endswith(".safetensors"))
        row["gb"] = b / 1e9 if b else None
        row["quant_in_repo"] = bool(getattr(info, "config", {}) and
                                    "quantization_config" in (info.config or {}))
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(row["id"], "config.json")
        cfg = json.loads(open(p, encoding="utf-8").read())
        row["layers"] = cfg.get("num_hidden_layers")
        row["hidden"] = cfg.get("hidden_size")
        row["quant_in_repo"] = "quantization_config" in cfg
    except Exception:
        row["gb"] = None
    return row


if __name__ == "__main__":
    cands = candidates()
    print(f"{len(cands)} repositories matched; filtering by config...")
    rows = [r for r in (from_listing(m) for m in cands) if r]
    print(f"{len(rows)} are MoE with a supported architecture")
    rows = [r for r in rows if usable(r["id"])]
    print(f"{len(rows)} are real weights in a loadable format; sizing...")
    with ThreadPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(weight_size, rows))
    rows = [r for r in rows
            if r["gb"] and not r.get("quant_in_repo")
            and isinstance(r.get("layers"), int) and r["layers"] >= 8]

    fits = [r for r in rows if r["gb"] <= USABLE_GB]
    fits.sort(key=lambda r: (r["sparsity"], -r["downloads"]))
    print(f"\n{len(rows)} MoE models with a supported architecture, "
          f"{len(fits)} of them fit {USABLE_GB} GB in fp16\n")
    hdr = (f"{'model':52s} {'arch':16s} {'GB':>5s} {'L':>3s} {'exp':>4s} "
           f"{'topk':>4s} {'active/layer':>12s} {'M_iso%':>7s} {'downloads':>9s}")
    print(hdr)
    print("-" * len(hdr))
    for r in fits:
        L = r["layers"] if isinstance(r["layers"], int) else 0
        print(f"{r['id'][:52]:52s} {r['model_type']:16s} {r['gb']:5.2f} "
              f"{L:3d} {r['experts']:4d} {r['top_k']:4d} "
              f"{100*r['sparsity']:11.1f}% {r['iso_M']:7.1f} {r['downloads']:9d}")
    print(f"\ntarget GPT-OSS-120B: 36 layers, 128 experts, top-4, "
          f"{100*TARGET_SPARSITY:.1f} % active per layer, M_iso = 5.0 %")

    too_big = sorted([r for r in rows if r["gb"] > USABLE_GB],
                     key=lambda r: r["sparsity"])[:12]
    if too_big:
        print(f"\nbest geometry but too large for {VRAM_GB} GB:")
        for r in too_big:
            print(f"  {r['id'][:52]:52s} {r['gb']:6.2f} GB  "
                  f"{100*r['sparsity']:5.1f} % active/layer")
