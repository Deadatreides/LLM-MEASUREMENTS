"""Trace collection, §4 + step 1 of §10.

Two phases per request:
  1. generate() with hooks OFF  -> obtain the full token sequence
  2. ONE forward pass over that sequence with hooks ON -> per-position routing

Phase 2 is exact: attention is causal, so the hidden state (and therefore the
router logits / FFN activations) at position t in a full-sequence forward is the
same as in incremental decoding.  It is also far faster, and it lets a second
model replay the *identical* token sequence, which is what makes the
quantisation control of §8 a clean comparison rather than a comparison of two
different texts.

Modes:
  --introspect   step 1 of §10: locate modules, verify hooks, 10 tokens, no run
  --run          full corpus
  --replay FILE  run over token sequences recorded by an earlier model
"""

import argparse
import gc
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

ROOT = Path(__file__).parent
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

MODELS = {
    # G is the only real MoE that fits 6 GB of VRAM unquantised; bitsandbytes
    # cannot quantise fused MoE experts, so 4-bit is not an option (quantcheck.py)
    "G":  dict(repo="ibm-granite/granite-3.1-1b-a400m-base", quant="none",
               device="cuda", kind="moe", dtype="float16"),
    "G2": dict(repo="ibm-granite/granite-3.1-1b-a400m-base", quant="simnf4",
               device="cuda", kind="moe", dtype="float16"),
    # random-weight rehearsal of the MoE path, see moesmoke.py
    "GSMOKE":  dict(repo="granite-moe-smoke", quant="none", device="cuda",
                    kind="moe", dtype="float16"),
    "GSMOKE2": dict(repo="granite-moe-smoke", quant="simnf4", device="cuda",
                    kind="moe", dtype="float16"),
    "A":  dict(repo="allenai/OLMoE-1B-7B-0924", quant="nf4",  device="cuda", kind="moe"),
    # OLMoE on CPU in fp16, as task 3 stage 4 prescribes.  bf16 is emulated on
    # Zen 1 just as fp16 is, and fp32 would need 27.7 GB.
    "O":  dict(repo="allenai/OLMoE-1B-7B-0924", quant="none", device="cpu",
               kind="moe", dtype="float16"),
    "A2": dict(repo="allenai/OLMoE-1B-7B-0924", quant="none", device="cpu",  kind="moe",
               dtype="bfloat16"),
    "B":  dict(repo="Qwen/Qwen3-1.7B",          quant="none", device="cuda", kind="dense",
               dtype="float16"),
    "B2": dict(repo="Qwen/Qwen3-1.7B",          quant="nf4",  device="cuda", kind="dense"),
    "C":  dict(repo="Qwen/Qwen3-4B",            quant="int8", device="cuda", kind="dense"),
    "D":  dict(repo="Qwen/Qwen3-30B-A3B",       quant="nf4",  device="cpu",  kind="moe"),
}

MASS_FRACTION = 0.90        # §3.2
DENSE_TARGET_BANDS = 128    # §3.1  s = d_ffn / 128
MOE_TARGET_BANDS = 45       # §3.1  s = d_expert / 45   (level 2)


# --------------------------------------------------------------- loading ----

def resolve(repo):
    """Prefer a plain local folder over the hub.

    Downloads here restart rather than resume, so shards are fetched by hand
    into models/<name>/ and loaded from there; no network access at all.
    """
    p = ROOT / "models" / repo.split("/")[-1]
    if (p / "config.json").exists():
        return str(p)
    return repo


def load(tag, band_override=None):
    cfg = MODELS[tag]
    kw = {}
    if cfg["quant"] == "nf4":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif cfg["quant"] == "int8":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        kw["dtype"] = getattr(torch, cfg.get("dtype", "float32"))

    if cfg["device"] == "cuda":
        kw["device_map"] = {"": 0}
    else:
        kw["device_map"] = {"": "cpu"}

    src = resolve(cfg["repo"])
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForCausalLM.from_pretrained(src, **kw)
    model.eval()
    if cfg["quant"] == "simnf4":
        n, el = simulate_nf4_(model)
        print(f"[{tag}] simulated NF4 on {n} tensors, {el/1e6:.1f} M parameters",
              flush=True)
    return tok, model, cfg


@torch.no_grad()
def simulate_nf4_(model, skip=("embed", "lm_head", "norm")):
    """Quantise to NF4 and immediately dequantise, in place.

    bitsandbytes cannot quantise fused MoE experts, so a real 4-bit MoE cannot
    be loaded at all (see quantcheck.py).  This injects exactly the numerical
    error of NF4 while keeping the fp16 layout: bnb's own kernels also compute
    in fp16 on dequantised weights, so the arithmetic a hook observes is the
    same.  What it does NOT reproduce is the memory saving -- irrelevant here,
    since the question §8 asks is whether quantisation changes the branching.
    """
    import bitsandbytes.functional as F
    n = el = 0
    for name, p in model.named_parameters():
        if p.dim() < 2 or any(s in name for s in skip):
            continue
        flat = p.detach().reshape(-1).to(torch.float16).cuda()
        q, state = F.quantize_4bit(flat, blocksize=64, quant_type="nf4")
        p.copy_(F.dequantize_4bit(q, state).reshape(p.shape).to(p.dtype))
        n += 1
        el += p.numel()
    return n, el


# --------------------------------------------------------- introspection ----

def find_layers(model):
    """Locate the list of decoder layers without assuming an attribute path.

    Identified structurally: the ModuleList whose entries contain an attention
    submodule.  Naming differs per architecture (`mlp`, `feed_forward`,
    `block_sparse_moe`), so keying on the FFN name breaks on new models; the
    expert ModuleLists never contain attention, so this discriminates cleanly.
    """
    cands = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.ModuleList) or len(mod) < 2:
            continue
        if any(any("attention" in type(s).__name__.lower()
                   for s in c.modules()) for c in mod):
            cands.append((name, mod))
    if not cands:
        raise RuntimeError("decoder layer list not found")
    # nested stacks would both match; the outermost (shortest name) is the stack
    cands.sort(key=lambda x: (x[0].count("."), -len(x[1])))
    return cands[0]


def find_router(layer, n_experts):
    """Find the router inside a MoE layer.

    Two shapes exist in the wild and both must be handled:
      * an nn.Linear (or a bitsandbytes replacement) with out_features == n_experts
      * transformers>=5 fused routers (e.g. OlmoeTopKRouter) which are plain
        nn.Module holding a `weight` Parameter of shape (n_experts, hidden)
    Identified by shape, never by name, per §10 step 1.
    """
    hits = []
    for name, mod in layer.named_modules():
        if not name:
            continue
        if getattr(mod, "out_features", None) == n_experts:
            hits.append((name, mod))
            continue
        w = getattr(mod, "weight", None)
        if isinstance(w, torch.Tensor) and w.dim() == 2 and w.shape[0] == n_experts:
            hits.append((name, mod))
    if len(hits) != 1:
        raise RuntimeError(f"router ambiguous or missing: {[h[0] for h in hits]}")
    return hits[0]


def find_down_proj(layer):
    """Find the FFN output projection; its *input* is silu(gate)*up (§4.3)."""
    hits = [(n, m) for n, m in layer.named_modules()
            if n.endswith("down_proj") and hasattr(m, "in_features")]
    if len(hits) != 1:
        raise RuntimeError(f"down_proj ambiguous or missing: {[h[0] for h in hits]}")
    return hits[0]


# ---------------------------------------------------------------- hooks -----

def truth_from_router_output(out, n_tokens, n_experts):
    """Experts the model actually ran, taken from the router's own output.

    Two shapes are handled:
      * transformers>=5 fused routers return (..., top_k_index) -- the last
        element is [tokens, k] of expert ids;
      * GraniteMoeTopKGating returns
        (index_sorted_experts, batch_index, batch_gates, expert_size, logits),
        where tokens are grouped by expert; the assignment is rebuilt from
        expert_size and batch_index.
    Returns a boolean [tokens, n_experts] mask, or None if unrecognised.
    """
    if not isinstance(out, (tuple, list)):
        return None
    if len(out) == 5 and isinstance(out[3], (list, tuple)):
        _, batch_index, _, expert_size, _ = out
        mask = np.zeros((n_tokens, n_experts), dtype=bool)
        bi = batch_index.detach().cpu().numpy()
        off = 0
        for e, sz in enumerate(expert_size):
            if sz:
                mask[bi[off:off + sz], e] = True
            off += sz
        return mask
    last = out[-1]
    if torch.is_tensor(last) and not last.is_floating_point():
        idx = last.detach().cpu().numpy().reshape(n_tokens, -1)
        mask = np.zeros((n_tokens, n_experts), dtype=bool)
        np.put_along_axis(mask, idx, True, axis=1)
        return mask
    return None


class MoECapture:
    """Level 1 (§4.1): full router logit vector per (token, layer)."""

    def __init__(self, model, cfg_hf, verify=False):
        self.n_experts = cfg_hf.num_experts if hasattr(cfg_hf, "num_experts") \
            else cfg_hf.num_local_experts
        self.top_k = cfg_hf.num_experts_per_tok
        self.layers_name, layers = find_layers(model)
        self.n_layers = len(layers)
        self.verify = verify
        self.enabled = True
        self.module_names = []
        self.routers = []
        self.handles = []
        self.buf, self.sel, self.inp, self.truth = {}, {}, {}, {}
        for li, layer in enumerate(layers):
            name, router = find_router(layer, self.n_experts)
            self.module_names.append(f"{self.layers_name}.{li}.{name}")
            self.routers.append(router)
            self.handles.append(router.register_forward_hook(self._make(li)))
            if verify and "." in name:
                # the gating module wrapping the linear reports which experts
                # actually ran, which is the strongest available cross-check
                parent = layer.get_submodule(name.rsplit(".", 1)[0])
                self.handles.append(
                    parent.register_forward_hook(self._make_truth(li)))

    def _make(self, li):
        def hook(mod, inp, out):
            if not self.enabled:
                return
            # transformers>=5 routers return (logits, top_k_scores, top_k_index);
            # element 0 is the FULL logit vector, before any top-k selection.
            # Older nn.Linear routers return the logits directly.
            tup = isinstance(out, (tuple, list))
            logits = out[0] if tup else out
            self.buf[li] = logits.detach().float().cpu()
            if self.verify:
                if tup and len(out) >= 3:
                    self.sel[li] = out[-1].detach().cpu()   # experts actually used
                self.inp[li] = inp[0].detach().float().cpu()
        return hook

    def _make_truth(self, li):
        def hook(mod, inp, out):
            if not self.enabled or not self.verify:
                return
            n = self.buf[li].shape[0] if li in self.buf else None
            if n:
                m = truth_from_router_output(out, n, self.n_experts)
                if m is not None:
                    self.truth[li] = m
        return hook

    def reset(self):
        self.buf, self.sel, self.inp, self.truth = {}, {}, {}, {}

    def collect(self, n_tokens):
        """-> topk_ids [T,L,k] uint8, topk_probs [T,L,k] f16, gap [T,L] f16"""
        L, k = self.n_layers, self.top_k
        ids = np.zeros((n_tokens, L, k), dtype=np.uint8)
        prob = np.zeros((n_tokens, L, k), dtype=np.float16)
        gap = np.zeros((n_tokens, L), dtype=np.float16)
        for li in range(L):
            lg = self.buf[li].reshape(-1, self.n_experts)
            assert lg.shape[0] == n_tokens, (lg.shape, n_tokens)
            p = torch.softmax(lg, dim=-1)
            srt, idx = lg.sort(dim=-1, descending=True)
            ids[:, li, :] = idx[:, :k].numpy().astype(np.uint8)
            prob[:, li, :] = torch.gather(p, 1, idx[:, :k]).numpy().astype(np.float16)
            gap[:, li] = (srt[:, k - 1] - srt[:, k]).numpy().astype(np.float16)  # §7
        return ids, prob, gap

    def remove(self):
        for h in self.handles:
            h.remove()


class DenseCapture:
    """§4.3: bands of |silu(gate)*up| taken as the input of down_proj.

    With band_size == 1 the unit is the channel itself (task 2, A.1).  Channel
    sets are far too large to keep as id lists -- 61k tokens x 28 layers x
    ~2760 active channels is 4.7e9 entries -- so they are stored as packed
    bitmaps, 6144 bits = 768 bytes per (token, layer).
    """

    def __init__(self, model, cfg_hf, band_size=None):
        self.layers_name, layers = find_layers(model)
        self.n_layers = len(layers)
        self.d_ffn = cfg_hf.intermediate_size
        if band_size is None:
            band_size = self.d_ffn // DENSE_TARGET_BANDS
        assert self.d_ffn % band_size == 0, (self.d_ffn, band_size)
        self.s = band_size
        self.n_bands = self.d_ffn // band_size
        self.bitmap = (band_size == 1)
        self.enabled = True
        self.module_names = []
        self.handles = []
        self.out_ids = [[] for _ in range(self.n_layers)]
        self.out_cnt = [[] for _ in range(self.n_layers)]
        self.out_chan = [[] for _ in range(self.n_layers)]
        for li, layer in enumerate(layers):
            name, dp = find_down_proj(layer)
            self.module_names.append(f"{self.layers_name}.{li}.{name}")
            self.handles.append(dp.register_forward_pre_hook(self._make(li)))

    def _make(self, li):
        def hook(mod, args):
            if not self.enabled:
                return
            act = args[0]                       # [B, T, d_ffn]
            a = act.detach().float().reshape(-1, self.d_ffn).abs()
            if self.bitmap:                     # channel units, task 2 A.1
                packs, cnts = [], []
                for i in range(0, a.shape[0], 256):
                    m, c = min_cover_mask(a[i:i + 256], MASS_FRACTION)
                    packs.append(np.packbits(m, axis=1))
                    cnts.append(c)
                self.out_ids[li].append(np.concatenate(packs))
                cnt = np.concatenate(cnts)
                self.out_cnt[li].append(cnt.astype(np.int16))
                # packing loses the ordering, so the dominant channel -- which
                # the well-conditioned entropy metric needs -- is kept apart
                self.out_chan[li].append(
                    a.argmax(-1).cpu().numpy().astype(np.int32))
                return
            bm = a.reshape(a.shape[0], self.n_bands, self.s).sum(-1)
            ids, cnt = min_cover(bm, MASS_FRACTION)
            self.out_ids[li].append(ids.astype(np.int16))
            self.out_cnt[li].append(cnt.astype(np.int16))
            # Diagnostic: the same criterion at channel granularity (s = 1).
            # Chunked over tokens: torch.sort also materialises an int64 index
            # tensor, so a [1300, 6144] sort would put ~130 MB of transients on
            # a card that already holds the model.
            ch = []
            for i in range(0, a.shape[0], 256):
                _, c2 = min_cover(a[i:i + 256], MASS_FRACTION, ids_needed=False)
                ch.append(c2)
            self.out_chan[li].append(np.concatenate(ch).astype(np.int32))
        return hook

    def reset(self):
        self.out_ids = [[] for _ in range(self.n_layers)]
        self.out_cnt = [[] for _ in range(self.n_layers)]
        self.out_chan = [[] for _ in range(self.n_layers)]

    def collect(self, n_tokens):
        ids, cnt, chan = [], [], []
        for li in range(self.n_layers):
            ids.append(np.concatenate(self.out_ids[li]))
            c = np.concatenate(self.out_cnt[li])
            assert len(c) == n_tokens, (len(c), n_tokens)
            cnt.append(c)
            chan.append(np.concatenate(self.out_chan[li]))
        return ids, np.stack(cnt, 1), np.stack(chan, 1)

    def remove(self):
        for h in self.handles:
            h.remove()


def min_cover(mass, frac, ids_needed=True):
    """Minimal set of units covering `frac` of the row mass (§3.2)."""
    srt, idx = mass.sort(dim=-1, descending=True)
    cum = srt.cumsum(-1)
    thr = frac * srt.sum(-1, keepdim=True)
    k = (cum < thr).sum(-1) + 1                      # [T]
    k = torch.clamp(k, max=mass.shape[-1])
    kn = k.cpu().numpy()
    if not ids_needed:
        return None, kn
    idxn = idx.cpu().numpy()
    flat = np.concatenate([idxn[t, :kn[t]] for t in range(len(kn))])
    return flat, kn


def min_cover_mask(mass, frac):
    """Same criterion as min_cover, returned as a boolean [T, n] mask.

    Used for channel units, where the id list is far too large to keep but the
    mask packs to one bit per channel.
    """
    srt, idx = mass.sort(dim=-1, descending=True)
    cum = srt.cumsum(-1)
    thr = frac * srt.sum(-1, keepdim=True)
    k = torch.clamp((cum < thr).sum(-1) + 1, max=mass.shape[-1])
    rank = torch.arange(mass.shape[-1], device=mass.device)[None, :]
    sorted_mask = rank < k[:, None]
    active = torch.zeros_like(sorted_mask)
    active.scatter_(1, idx, sorted_mask)
    return active.cpu().numpy(), k.cpu().numpy()


# ------------------------------------------------------------ generation ----

def turn_prompt(text, first):
    """Plain-text conversation format, identical for every model (see corpus)."""
    return ("" if first else "\n\n") + f"User: {text}\nAssistant:"


@torch.no_grad()
def generate_batch(tok, model, items, device):
    """Phase 1 for single-turn requests, several at a time.

    Decoding one sequence at a time on a small model is dominated by per-step
    overhead rather than by the GPU, so part A (510 independent one-turn
    requests) is generated in batches.  Left padding plus the attention mask
    keeps each sequence's positions right; verify_batching() checks that the
    batched result is token-identical to generating alone.
    """
    prompts = [turn_prompt(it["turns"][0], True) for it in items]
    enc = tok(prompts, return_tensors="pt", padding=True,
              padding_side="left").to(device)
    n_in = enc["input_ids"].shape[1]
    eos = tok.eos_token_id
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos
    out = model.generate(**enc, max_new_tokens=items[0]["max_new_tokens"],
                         do_sample=False, pad_token_id=pad)
    results = []
    for i, item in enumerate(items):
        new = out[i, n_in:].tolist()
        if eos is not None and eos in new:                # drop trailing padding
            new = new[:new.index(eos) + 1]
        prompt_ids = tok(prompts[i], add_special_tokens=True)["input_ids"]
        seq = prompt_ids + new
        gen = np.array([False] * len(prompt_ids) + [True] * len(new), dtype=bool)
        turn = np.zeros(len(seq), dtype=np.int16)
        results.append((torch.tensor(seq, dtype=torch.long), gen, turn))
    return results


@torch.no_grad()
def verify_batching(tok, model, items, device, log=print):
    """Batched generation must be token-identical to one-at-a-time."""
    batched = generate_batch(tok, model, items, device)
    ok = True
    for i, item in enumerate(items):
        alone = generate_sequence(tok, model, item, device)
        same = alone[0].tolist() == batched[i][0].tolist()
        ok &= same
        if not same:
            log(f"  request {item['request_id']}: batched != alone "
                f"({len(alone[0])} vs {len(batched[i][0])} tokens)")
    log(f"  batching check on {len(items)} requests: "
        f"{'identical' if ok else 'MISMATCH'}")
    return ok


@torch.no_grad()
def generate_sequence(tok, model, item, device):
    """Phase 1. Returns token ids and a per-position mask of generated tokens.

    The sequence is accumulated as token ids rather than re-tokenised from
    rendered text, so the generated spans are exact by construction:
    re-tokenising a prefix can merge across the prompt/answer boundary and
    shift the mask by a token per turn.
    """
    seq = []
    gen, turn_id = [], []
    for ti, text in enumerate(item["turns"]):
        p = tok(turn_prompt(text, ti == 0),
                add_special_tokens=(ti == 0))["input_ids"]
        seq += p
        gen += [False] * len(p)
        turn_id += [ti] * len(p)
        enc = torch.tensor([seq], dtype=torch.long, device=device)
        out = model.generate(
            input_ids=enc,
            attention_mask=torch.ones_like(enc),
            max_new_tokens=item["max_new_tokens"],
            do_sample=False,
            pad_token_id=tok.pad_token_id if tok.pad_token_id is not None
            else tok.eos_token_id,
        )
        new = out[0, len(seq):].tolist()
        seq += new
        gen += [True] * len(new)
        turn_id += [ti] * len(new)
    return (torch.tensor(seq, dtype=torch.long),
            np.array(gen, dtype=bool),
            np.array(turn_id, dtype=np.int16))


# ------------------------------------------------------------------ run -----

def setup_log(tag):
    """Log to a file and to the console, flushed, with timestamps.

    Progress lines go through tqdm.write so the bar is not broken up.
    """
    (ROOT / "logs").mkdir(exist_ok=True)
    lg = logging.getLogger(tag)
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    fh = logging.FileHandler(ROOT / "logs" / f"{tag}.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
    lg.addHandler(fh)

    class Bar(logging.Handler):
        def emit(self, r):
            tqdm.write(self.format(r))
    sh = Bar()
    sh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
    lg.addHandler(sh)
    return lg


class Acc:
    """Per-request results, flushed to shards so a long run can resume."""

    def __init__(self, kind, n_layers):
        self.kind, self.n_layers = kind, n_layers
        self.clear()

    def clear(self):
        self.rid, self.pos, self.turn, self.tok = [], [], [], []
        self.moe = ([], [], [])
        self.dense_ids = [[] for _ in range(self.n_layers)]
        self.dense_cnt, self.dense_chan = [], []
        self.seqs = {}
        self.n_req = 0

    def to_blob(self):
        b = {"request_id": np.concatenate(self.rid),
             "token_pos": np.concatenate(self.pos),
             "turn": np.concatenate(self.turn),
             "token_id": np.concatenate(self.tok)}
        if self.kind == "moe":
            b["topk_ids"] = np.concatenate(self.moe[0])
            b["topk_prob"] = np.concatenate(self.moe[1])
            b["gap"] = np.concatenate(self.moe[2])
        else:
            b["band_cnt"] = np.concatenate(self.dense_cnt)
            b["chan_cnt"] = np.concatenate(self.dense_chan)
            for li in range(self.n_layers):
                b[f"band_ids_{li}"] = np.concatenate(self.dense_ids[li])
        return b

    @property
    def n_tok(self):
        return sum(len(x) for x in self.pos)


def merge_shards(paths, kind, n_layers):
    parts = [np.load(p, allow_pickle=False) for p in paths]
    keys = ["request_id", "token_pos", "turn", "token_id"]
    keys += (["topk_ids", "topk_prob", "gap"] if kind == "moe"
             else ["band_cnt", "chan_cnt"] + [f"band_ids_{i}"
                                              for i in range(n_layers)])
    out = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    for p in parts:
        p.close()
    return out


@torch.no_grad()
def run(tag, corpus, out_path, limit=None, replay=None, band_size=None,
        batch=8, checkpoint=50, resume=True):
    lg = setup_log(tag)
    out_path = Path(out_path)
    parts = out_path.with_suffix("").with_name(out_path.stem + ".parts")
    parts.mkdir(parents=True, exist_ok=True)

    tok, model, cfg = load(tag)
    device = "cuda" if cfg["device"] == "cuda" else "cpu"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    hf = model.config
    cap = (MoECapture(model, hf) if cfg["kind"] == "moe"
           else DenseCapture(model, hf, band_size=band_size))
    lg.info(f"{cfg['repo']}  quant={cfg['quant']}  dev={cfg['device']}  "
            f"kind={cfg['kind']}")
    lg.info(f"hooked {len(cap.module_names)} modules, e.g. {cap.module_names[0]}")
    if cfg["kind"] == "dense":
        lg.info(f"d_ffn={cap.d_ffn} band_size={cap.s} n_bands={cap.n_bands}")
    else:
        lg.info(f"experts={cap.n_experts} top_k={cap.top_k} layers={cap.n_layers}")

    items = corpus if limit is None else corpus[:limit]

    done_shards = sorted(parts.glob("part_*.npz")) if resume else []
    done_ids = set()
    for p in done_shards:
        with np.load(p, allow_pickle=False) as z:
            done_ids.update(np.unique(z["request_id"]).tolist())
    if done_shards:
        lg.info(f"resuming: {len(done_shards)} shard(s) on disk, "
                f"{len(done_ids)} requests already traced")
    seq_file = out_path.with_suffix(".seq.json")
    all_seqs = {}
    seq_part = parts / "seqs.json"
    if seq_part.exists() and resume:
        all_seqs = json.loads(seq_part.read_text(encoding="utf-8"))

    pending = [it for it in items if it["request_id"] not in done_ids]
    if not pending:
        lg.info("nothing left to trace")
    acc = Acc(cfg["kind"], cap.n_layers)
    shard_no = len(done_shards)

    def trace_one(item, ids, gen, turn_id):
        cap.reset()
        model(input_ids=ids.unsqueeze(0).to(device))
        T = len(ids)
        g = np.nonzero(gen)[0]
        if cfg["kind"] == "moe":
            a, b, c = cap.collect(T)
            acc.moe[0].append(a[gen]); acc.moe[1].append(b[gen])
            acc.moe[2].append(c[gen])
        else:
            ids_l, cnt, chan = cap.collect(T)
            for li in range(cap.n_layers):
                if cap.bitmap:                  # one row of bits per token
                    acc.dense_ids[li].append(ids_l[li][gen])
                    continue
                off = np.concatenate([[0], np.cumsum(cnt[:, li])])
                starts, lens = off[g], cnt[g, li].astype(np.int64)
                base = np.repeat(
                    starts - np.concatenate([[0], np.cumsum(lens)[:-1]]), lens)
                acc.dense_ids[li].append(ids_l[li][base + np.arange(lens.sum())])
            acc.dense_cnt.append(cnt[gen]); acc.dense_chan.append(chan[gen])
        acc.rid.append(np.full(len(g), item["request_id"], dtype=np.int32))
        acc.pos.append(g.astype(np.int32))
        acc.turn.append(turn_id[g])
        acc.tok.append(ids.numpy()[g].astype(np.int32))
        acc.n_req += 1

    def flush():
        nonlocal shard_no
        if acc.n_req == 0:
            return
        p = parts / f"part_{shard_no:04d}.npz"
        np.savez_compressed(p, **acc.to_blob())
        all_seqs.update(acc.seqs)
        if all_seqs:
            seq_part.write_text(json.dumps(all_seqs), encoding="utf-8")
        lg.info(f"checkpoint -> {p.name}  ({acc.n_req} req, {acc.n_tok} tok, "
                f"{p.stat().st_size/1e6:.1f} MB)")
        shard_no += 1
        acc.clear()

    if replay is None and pending and batch > 1:
        probe_items = [it for it in pending if len(it["turns"]) == 1][:2]
        if probe_items:
            cap.enabled = False
            verify_batching(tok, model, probe_items, device, log=lg.info)
            cap.enabled = True

    t0 = time.time()
    n_tok_total = 0
    bar = tqdm(total=len(pending), unit="req", dynamic_ncols=True,
               desc=f"{tag} trace")
    i = 0
    try:
        while i < len(pending):
            if replay is not None:
                chunk = [pending[i]]
                rid = str(chunk[0]["request_id"])
                if rid not in replay:
                    i += 1
                    bar.update(1)
                    continue
                r = replay[rid]
                res = [(torch.tensor(r["ids"], dtype=torch.long),
                        np.array(r["gen"], dtype=bool),
                        np.array(r["turn"], dtype=np.int16))]
            elif len(pending[i]["turns"]) == 1:
                chunk = []
                while (i + len(chunk) < len(pending) and len(chunk) < batch
                       and len(pending[i + len(chunk)]["turns"]) == 1):
                    chunk.append(pending[i + len(chunk)])
                cap.enabled = False
                res = generate_batch(tok, model, chunk, device)
                cap.enabled = True
            else:
                chunk = [pending[i]]
                cap.enabled = False
                res = [generate_sequence(tok, model, chunk[0], device)]
                cap.enabled = True

            for item, (ids, gen, turn_id) in zip(chunk, res):
                if replay is None:
                    acc.seqs[item["request_id"]] = {
                        "ids": ids.tolist(), "gen": gen.tolist(),
                        "turn": turn_id.tolist()}
                trace_one(item, ids, gen, turn_id)
                n_tok_total += int(gen.sum())

            i += len(chunk)
            bar.update(len(chunk))
            bar.set_postfix_str(f"{n_tok_total} tok, "
                                f"{n_tok_total/max(time.time()-t0,1e-9):.1f} tok/s")
            if acc.n_req >= checkpoint:
                flush()
    finally:
        bar.close()
        flush()
        cap.remove()

    shards = sorted(parts.glob("part_*.npz"))
    blob = merge_shards(shards, cfg["kind"], cap.n_layers)
    blob["model_tag"] = np.array([tag])
    blob["repo"] = np.array([cfg["repo"]])
    blob["quant"] = np.array([cfg["quant"]])
    blob["kind"] = np.array([cfg["kind"]])
    blob["modules"] = np.array(cap.module_names)
    blob["n_layers"] = np.array([cap.n_layers])
    blob["is_gen"] = np.ones(len(blob["request_id"]), dtype=bool)
    if cfg["kind"] == "moe":
        blob["n_experts"] = np.array([cap.n_experts])
        blob["top_k"] = np.array([cap.top_k])
    else:
        blob["n_bands"] = np.array([cap.n_bands])
        blob["band_s"] = np.array([cap.s])
        blob["d_ffn"] = np.array([cap.d_ffn])
        blob["unit"] = np.array(["channel" if cap.bitmap else "band"])
    np.savez_compressed(out_path, **blob)
    lg.info(f"saved {out_path.name}  ({out_path.stat().st_size/1e6:.1f} MB, "
            f"{len(blob['request_id'])} tokens from {len(shards)} shard(s))")
    if all_seqs:
        seq_file.write_text(json.dumps(all_seqs), encoding="utf-8")
        lg.info(f"saved {seq_file.name}  ({len(all_seqs)} sequences)")
    del model
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------- introspect ------

@torch.no_grad()
def introspect(tag):
    introspect_facts = {}
    tok, model, cfg = load(tag)
    hf = model.config
    device = "cuda" if cfg["device"] == "cuda" else "cpu"
    print("=" * 78)
    print(f"MODEL {tag}: {cfg['repo']}   quant={cfg['quant']}   device={cfg['device']}")
    print(f"config: {type(hf).__name__}")
    for f in ("num_hidden_layers", "hidden_size", "intermediate_size",
              "num_experts", "num_local_experts", "num_experts_per_tok",
              "moe_intermediate_size", "norm_topk_prob"):
        if hasattr(hf, f):
            print(f"  {f} = {getattr(hf, f)}")
    lname, layers = find_layers(model)
    print(f"decoder layers found at: {lname}  (n={len(layers)})")
    print("--- layer 0 module tree (depth<=2) ---")
    for n, m in layers[0].named_modules():
        if n and n.count(".") <= 2:
            extra = ""
            if hasattr(m, "in_features"):
                extra = f"  in={m.in_features} out={m.out_features}"
            print(f"    {n:34s} {type(m).__name__}{extra}")

    text = "User: Explain what a race condition is.\nAssistant: A race condition"
    ids = tok(text, return_tensors="pt")["input_ids"].to(device)
    T = ids.shape[1]
    print(f"--- probe forward, {T} tokens ---")

    if cfg["kind"] == "moe":
        cap = MoECapture(model, hf, verify=True)
        print(f"router modules: {cap.module_names[0]} ... "
              f"({len(cap.module_names)}), class "
              f"{type(cap.routers[0]).__name__}")
        cap.reset()
        model(input_ids=ids)
        raw = cap.buf[0]
        print(f"hook capture layer0 shape = {tuple(raw.shape)} "
              f"(expected [{T}, {cap.n_experts}])")
        print(f"  -> FULL logit vector, not top-k: last dim == num_experts = "
              f"{raw.shape[-1] == cap.n_experts}")
        print(f"  distinct values in row 0 = {len(torch.unique(raw[0]))} of "
              f"{cap.n_experts} (a post-top-k tensor would have ties at -inf/0)")
        print(f"  softmax sums to 1: "
              f"{[round(float(v),6) for v in torch.softmax(raw,-1).sum(-1)[:3]]}")

        # Check 1: recompute the logits from the router's own input and weight.
        # The model computes this in its own dtype on the GPU while the
        # recomputation is fp32 on the CPU, so the residual is the numerical
        # noise floor of the router, not a hook error.  It is reported because
        # §7 measures a gap that can be of the same order.
        w = cap.routers[0].weight.detach().float().cpu()
        ref = torch.nn.functional.linear(
            cap.inp[0].reshape(-1, w.shape[1]), w)
        d = (ref - raw).abs().max().item()
        scale = raw.max().item() - raw.min().item()
        rel = d / max(scale, 1e-9)
        print(f"  CHECK 1 recompute W_gate @ x : max|diff| = {d:.3e}, "
              f"logit range {scale:.3f} -> relative {rel:.2%} "
              f"(hook correct: {rel < 0.01})")
        print(f"    numerical noise floor of the router = {d:.3e}; §7 gaps "
              f"below this are not resolvable in "
              f"{next(model.parameters()).dtype}")
        introspect_facts["router_noise_floor"] = d
        introspect_facts["logit_range"] = scale

        # Check 2: our top-k must be the experts the model actually ran.
        a, b, c = cap.collect(T)
        ours = np.zeros((T, cap.n_experts), dtype=bool)
        np.put_along_axis(ours, a[:, 0, :].astype(np.int64), True, axis=1)
        if 0 in cap.truth:
            same = (ours == cap.truth[0])
            print(f"  CHECK 2 vs experts the model actually ran: "
                  f"{100*same.all(1).mean():.2f} % of tokens identical")
        elif 0 in cap.sel:
            truth = cap.sel[0].reshape(T, -1).numpy()
            same = (np.sort(truth, 1) == np.sort(a[:, 0, :].astype(truth.dtype), 1))
            print(f"  CHECK 2 vs router_indices actually used by the model: "
                  f"{100*same.mean():.2f} % of selections identical")
        else:
            print("  CHECK 2 unavailable: router does not expose selected experts")
        print(f"  top_k={cap.top_k}  ids[tok0,layer0]={a[0,0].tolist()}")
        print(f"  probs[tok0,layer0]={[round(float(x),4) for x in b[0,0]]}")
        print(f"  gap[tok0,:4]={[round(float(x),4) for x in c[0,:4]]}")
        cap.remove()
    else:
        cap = DenseCapture(model, hf)
        print(f"ffn modules: {cap.module_names[0]} ... ({len(cap.module_names)})")
        print(f"d_ffn={cap.d_ffn}  band_size s={cap.s}  n_bands={cap.n_bands}")
        cap.reset()
        model(input_ids=ids)
        cnt = np.concatenate(cap.out_cnt[0])
        chan = np.concatenate(cap.out_chan[0])
        print(f"  captured rows layer0 = {len(cnt)} (expected {T})")
        print(f"  active bands per token (layer0): {cnt.tolist()}  of {cap.n_bands}")
        print(f"  active channels per token (layer0): {chan.tolist()} of {cap.d_ffn}")
        cap.remove()
        # Independent recomputation: the input of down_proj must equal
        # silu(gate_proj(h)) * up_proj(h) where h is the input of mlp.
        lay = layers[0]
        h = {}
        hk = lay.mlp.down_proj.register_forward_pre_hook(
            lambda m, a: h.__setitem__("x", a[0].detach().float()))
        hi = lay.mlp.register_forward_pre_hook(
            lambda m, a: h.__setitem__("h", a[0].detach()))
        model(input_ids=ids)
        hk.remove(); hi.remove()
        ref = (torch.nn.functional.silu(lay.mlp.gate_proj(h["h"]))
               * lay.mlp.up_proj(h["h"])).float()
        d = (ref - h["x"]).abs().max().item()
        scale = h["x"].abs().max().item()
        print(f"  down_proj input shape = {tuple(h['x'].shape)}  "
              f"(last dim == d_ffn: {h['x'].shape[-1] == cap.d_ffn})")
        print(f"  CROSS-CHECK vs silu(gate_proj(h))*up_proj(h): "
              f"max|diff|={d:.3e} (scale {scale:.2f}) -> identical: {d/scale < 1e-3}")
        print("  -> band mass summed over the d_ffn axis (last), per token row")
    print("=" * 78)
    if introspect_facts:
        (ROOT / "out").mkdir(exist_ok=True)
        introspect_facts["dtype"] = str(next(model.parameters()).dtype)
        (ROOT / "out" / f"{tag}_introspect.json").write_text(
            json.dumps(introspect_facts, indent=1), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--introspect", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--replay", type=str, default=None)
    ap.add_argument("--band", type=int, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--batch", type=int, default=8,
                    help="single-turn requests generated together (1 disables)")
    ap.add_argument("--checkpoint", type=int, default=50,
                    help="write a resumable shard every N requests")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore shards already on disk and start over")
    ap.add_argument("--stride", type=int, default=1,
                    help="keep every Nth request WITHIN each corpus part, so "
                         "halving the corpus keeps all 17 domains and the A/B "
                         "proportion (task 3, §4.2)")
    a = ap.parse_args()
    if a.introspect:
        introspect(a.tag)
    else:
        corpus = json.loads((ROOT / "corpus.json").read_text(encoding="utf-8"))
        if a.stride > 1:
            seen = {}
            kept = []
            for it in corpus:
                i = seen.get(it["part"], 0)
                seen[it["part"]] = i + 1
                if i % a.stride == 0:
                    kept.append(it)
            n_a = sum(1 for c in kept if c["part"] == "A")
            n_b = sum(1 for c in kept if c["part"] == "B")
            print(f"stride {a.stride}: {len(kept)} of {len(corpus)} requests "
                  f"(part A {n_a}, part B {n_b}), "
                  f"~{n_a*80 + n_b*1000} tokens", flush=True)
            corpus = kept
        rep = json.loads(Path(a.replay).read_text(encoding="utf-8")) if a.replay else None
        out = a.out or str(ROOT / "traces" / f"{a.tag}.npz")
        run(a.tag, corpus, out, limit=a.limit, replay=rep, band_size=a.band,
            batch=a.batch, checkpoint=a.checkpoint, resume=not a.no_resume)
