"""Cache simulation (§5) and trajectory metrics (§6, §7).

Fully offline over a collected trace.  Every policy operates on the same
per-token list of required units, so the only thing that differs between
policies is the eviction rule.

Unit = (layer, expert) for MoE level 1, (layer, band) for dense.
All units of one model have the same parameter size, therefore
"budget as a fraction of total FFN/MoE mass" == "fraction of the unit count".
"""

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
M_GRID = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50]
# EPOCH_BAL is not in the task spec.  It is added because a global demand
# ranking starves whole layers whenever the budget is below one token's working
# set, which confounds "the policy is bad" with "there is no structure".
POLICIES = ("STATIC", "LRU", "LFU", "EPOCH", "EPOCH_BAL")
EPOCH_LEN = 15
LFU_TAU = 100.0
POPCNT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1).astype(np.int32)

# GPT-OSS-120B target: 36 layers x 128 experts = 4608 units, 144 active/token.
# M = 5 % of mass -> 230 resident experts -> resident/active = 1.60.
TARGET_CACHE_OVER_ACTIVE = 230 / 144


class Trace:
    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.path = Path(path)
        self.tag = str(z["model_tag"][0])
        self.repo = str(z["repo"][0])
        self.quant = str(z["quant"][0])
        self.kind = str(z["kind"][0])
        self.L = int(z["n_layers"][0])
        self.request_id = z["request_id"]
        self.turn = z["turn"]
        self.n_tok = T = len(self.request_id)

        if self.kind == "moe":
            self.E = int(z["n_experts"][0])
            self.top_k = int(z["top_k"][0])
            self.per_layer = self.E
            self.U = self.L * self.E
            ids = z["topk_ids"].astype(np.int32)                    # [T, L, k]
            off = (np.arange(self.L, dtype=np.int32) * self.E)[None, :, None]
            self.flat = (ids + off).reshape(-1)
            per_tok = self.L * self.top_k
            self.offsets = np.arange(T + 1, dtype=np.int64) * per_tok
            self.sig_src = ids
            self.gap = z["gap"]
        else:
            self.n_bands = int(z["n_bands"][0])
            self.band_s = int(z["band_s"][0])
            self.d_ffn = int(z["d_ffn"][0])
            self.unit = str(z["unit"][0]) if "unit" in z.files else "band"
            cnt = z["band_cnt"].astype(np.int64)                    # [T, L]
            self.band_cnt = cnt
            self.chan_cnt = z["chan_cnt"]
            if self.unit in ("channel", "cluster"):
                # one packed bit per unit; an id list would be 4.7e9 entries
                self.per_layer = self.n_bands
                self.U = self.L * self.n_bands
                self._bitrows = [z[f"band_ids_{li}"] for li in range(self.L)]
                self._bits = np.concatenate(self._bitrows, axis=1)
                self.top1 = z["chan_cnt"].astype(np.int64)          # dominant ch
                self.counts = cnt.sum(1).astype(np.int32)
                self.offsets = np.concatenate(
                    [[0], np.cumsum(self.counts)]).astype(np.int64)
                self.flat = None
            else:
                self.per_layer = self.n_bands
                self.U = self.L * self.n_bands
                self._layer_ids = [z[f"band_ids_{li}"].astype(np.int32)
                                   for li in range(self.L)]
                self._layer_offs = [np.concatenate([[0], np.cumsum(cnt[:, li])])
                                    for li in range(self.L)]
                self.flat, self.offsets = self._interleave(cnt)

        if getattr(self, "unit", "band") not in ("channel", "cluster"):
            self.counts = np.diff(self.offsets).astype(np.int32)
        self.mean_active = float(self.counts.mean())
        self.layer_of = (np.arange(self.U, dtype=np.int32) // self.per_layer)
        if not hasattr(self, "_bits"):
            self._bits = None
        self._freq = None

    def _interleave(self, cnt):
        """Reorder per-layer band lists into one per-token, layer-ordered array."""
        T, L = cnt.shape
        tot_tok = cnt.sum(1)
        offsets = np.concatenate([[0], np.cumsum(tot_tok)]).astype(np.int64)
        within = np.concatenate([np.zeros((T, 1), np.int64),
                                 np.cumsum(cnt, axis=1)[:, :-1]], axis=1)
        flat = np.empty(int(offsets[-1]), dtype=np.int32)
        for li in range(L):
            c = cnt[:, li]
            src = self._layer_ids[li]
            src_start = np.repeat(self._layer_offs[li][:-1], c)
            dst_start = np.repeat(offsets[:-1] + within[:, li], c)
            local = np.arange(len(src), dtype=np.int64) - src_start
            flat[dst_start + local] = src + li * self.n_bands
        return flat, offsets

    def token_units(self, t):
        if self.flat is None:                 # channel units, unpacked on demand
            return np.nonzero(np.unpackbits(self._bits[t]))[0].astype(np.int32)
        return self.flat[self.offsets[t]:self.offsets[t + 1]]

    def freq(self):
        """How many tokens use each unit."""
        if self._freq is not None:
            return self._freq
        if self.flat is not None:
            self._freq = np.bincount(self.flat, minlength=self.U)
        else:
            f = np.zeros(self.U, dtype=np.int64)
            for i in range(0, self.n_tok, 1024):   # chunked: unpacking the whole
                f += np.unpackbits(self._bits[i:i + 1024],  # trace would be 10 GB
                                   axis=1).sum(0, dtype=np.int64)
            self._freq = f
        return self._freq

    def bitsets(self):
        """Packed per-token active-set bitmaps, for cheap overlap (§6.1)."""
        if self._bits is None:
            nbytes = (self.U + 7) // 8
            b = np.zeros((self.n_tok, self.U), dtype=np.uint8)
            rows = np.repeat(np.arange(self.n_tok), self.counts)
            b[rows, self.flat] = 1
            self._bits = np.packbits(b, axis=1)
            del b
        return self._bits

    def describe(self):
        d = dict(tag=self.tag, repo=self.repo, quant=self.quant, kind=self.kind,
                 layers=self.L, units_total=self.U, tokens=self.n_tok,
                 mean_active_per_token=round(self.mean_active, 2),
                 requests=int(len(np.unique(self.request_id))))
        if self.kind == "moe":
            d.update(experts=self.E, top_k=self.top_k)
        else:
            d.update(n_bands=self.n_bands, band_s=self.band_s, d_ffn=self.d_ffn,
                     mean_active_bands=round(float(self.band_cnt.mean()), 2),
                     mean_active_channels=round(float(self.chan_cnt.mean()), 1))
        return d


# ------------------------------------------------------------ policies ------

def simulate(tr, capacity, policy, per_layer=False):
    U, T = tr.U, tr.n_tok
    resident = np.zeros(U, dtype=bool)
    freq = tr.freq()

    if policy == "STATIC":
        # filled once from whole-trace frequency and never changed (§5.2)
        resident[np.argsort(-freq, kind="stable")[:capacity]] = True

    last_use = np.full(U, -1, dtype=np.int64)
    score = np.zeros(U, dtype=np.float64)
    score_t = np.zeros(U, dtype=np.int64)
    epoch_demand = np.zeros(U, dtype=np.int64)

    hits = np.zeros(T, dtype=np.int32)
    need = tr.counts
    churn = []
    runs = RunTracker()
    lh = np.zeros(tr.L) if per_layer else None
    ln = np.zeros(tr.L) if per_layer else None
    n_res = int(resident.sum())

    for t in range(T):
        units = tr.token_units(t)
        res = resident[units]
        hits[t] = int(res.sum())
        runs.add(res)
        if per_layer:
            lay = tr.layer_of[units]
            lh += np.bincount(lay[res], minlength=tr.L)
            ln += np.bincount(lay, minlength=tr.L)

        if policy == "STATIC":
            continue

        if policy == "LRU":
            last_use[units[res]] = t
            # already unique: a layer contributes a set, and layers are offset
            # into disjoint id ranges (checked in selftest)
            miss = units[~res]
            if len(miss):
                # A token needs `need[t]` units at once.  When the cache is
                # smaller than that, only the most recently accessed ones can
                # survive -- the cache thrashes, which is the honest outcome.
                ins = miss[-min(len(miss), capacity):]
                n_ev = max(0, n_res + len(ins) - capacity)
                if n_ev:
                    cand = np.nonzero(resident)[0]
                    n_ev = min(n_ev, len(cand))
                    victims = cand[np.argsort(last_use[cand], kind="stable")[:n_ev]]
                    resident[victims] = False
                resident[ins] = True
                last_use[ins] = t
                n_res = int(resident.sum())

        elif policy == "LFU":
            score[units] = score[units] * np.exp(-(t - score_t[units]) / LFU_TAU) + 1.0
            score_t[units] = t
            miss = units[~res]
            if len(miss):
                ins = miss[-min(len(miss), capacity):]
                n_ev = max(0, n_res + len(ins) - capacity)
                if n_ev:
                    cand = np.nonzero(resident)[0]
                    n_ev = min(n_ev, len(cand))
                    cs = score[cand] * np.exp(-(t - score_t[cand]) / LFU_TAU)
                    victims = cand[np.argsort(cs, kind="stable")[:n_ev]]
                    resident[victims] = False
                resident[ins] = True
                n_res = int(resident.sum())

        elif policy in ("EPOCH", "EPOCH_BAL"):
            epoch_demand[units] += 1
            if (t + 1) % EPOCH_LEN == 0:
                want = np.zeros(U, dtype=bool)
                if policy == "EPOCH":
                    order = np.argsort(-epoch_demand, kind="stable")[:capacity]
                    want[order[epoch_demand[order] > 0]] = True
                else:
                    # equal share of the budget to every layer, then top demand
                    # within the layer
                    per = capacity // tr.L
                    extra = capacity - per * tr.L
                    dm = epoch_demand.reshape(tr.L, tr.per_layer)
                    for li in range(tr.L):
                        take = per + (1 if li < extra else 0)
                        o = np.argsort(-dm[li], kind="stable")[:take]
                        o = o[dm[li][o] > 0]
                        want[li * tr.per_layer + o] = True
                short = capacity - int(want.sum())
                if short > 0:                      # keep previous content as filler
                    keep = np.nonzero(resident & ~want)[0][:short]
                    want[keep] = True
                churn.append(int((want & ~resident).sum()) / max(capacity, 1))
                resident = want
                n_res = int(resident.sum())
                epoch_demand[:] = 0
        else:
            raise ValueError(policy)

    assert int(resident.sum()) <= capacity, "cache overflowed its budget"
    runs.finish()
    n_warm, n_steady = int(T * 0.10), int(T * 0.50)
    out = dict(
        q=float(hits.sum() / need.sum()),
        q_warm=float(hits[:n_warm].sum() / max(need[:n_warm].sum(), 1)),
        q_steady=float(hits[-n_steady:].sum() / max(need[-n_steady:].sum(), 1)),
        churn=float(np.mean(churn)) if churn else None,
        capacity_units=int(capacity),
        # a cache smaller than one token's working set cannot reach q = 1
        q_ceiling=float(min(1.0, capacity / tr.mean_active)),
        miss_run_p50=runs.quantile(0.5),
        miss_run_p90=runs.quantile(0.9),
        miss_run_max=int(runs.max),
    )
    if per_layer:
        out["q_by_layer"] = (lh / np.maximum(ln, 1)).tolist()
    return out


def miss_runs(hit_seq):
    m = (~hit_seq).astype(np.int8)
    if m.sum() == 0:
        return np.array([], dtype=np.int64)
    pad = np.zeros(m.size + 2, np.int8)
    pad[1:-1] = m
    d = np.diff(pad)
    return np.nonzero(d == -1)[0] - np.nonzero(d == 1)[0]


class RunTracker:
    """Streaming length histogram of consecutive misses (§5.3).

    The whole hit sequence is `tokens x units_per_token`, which at channel
    granularity is 4.7e9 entries -- it can be neither stored nor turned into a
    list of run lengths.  Runs are accumulated per token, carrying across the
    token boundary, into a fixed histogram; median and p90 come from that.
    """

    def __init__(self, nbins=65536):
        self.hist = np.zeros(nbins + 1, dtype=np.int64)
        self.nbins = nbins
        self.carry = 0
        self.max = 0

    def _push(self, length):
        if length <= 0:
            return
        self.max = max(self.max, length)
        self.hist[min(length, self.nbins)] += 1

    def add(self, hit):
        m = ~hit
        n = m.size
        if n == 0:
            return
        if m.all():
            self.carry += n
            return
        pad = np.zeros(n + 2, np.int8)
        pad[1:-1] = m
        d = np.diff(pad)
        lens = (np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).tolist()
        if m[0]:
            lens[0] += self.carry            # this run continues the carry
        elif self.carry:
            self._push(self.carry)
        self.carry = 0
        if m[-1]:
            self.carry = lens.pop()          # unfinished, carried to next token
        for x in lens:
            self._push(x)

    def finish(self):
        self._push(self.carry)
        self.carry = 0

    def quantile(self, q):
        tot = self.hist.sum()
        if tot == 0:
            return 0.0
        c = np.cumsum(self.hist)
        return float(np.searchsorted(c, q * tot))


# ------------------------------------------------------------- metrics ------

def core_units(tr, threshold=0.9):
    """Units active in at least `threshold` of all tokens.

    A large always-on core makes o_k look high for free: if 84 % of the units
    are active every token, any two tokens overlap by 84 % regardless of the
    mode.  Splitting it out is what makes §6.1 mean anything on a dense model.
    """
    rate = tr.freq() / tr.n_tok
    return rate >= threshold, rate


def persistence(tr, kmax=64, stride=5, exclude=None):
    """§6.1  o_k = |active(t) & active(t+k)| / |active(t)|, median over trace.

    With `exclude` (a unit mask) the overlap is measured on the variable part
    only, which is the part a cache policy can actually win or lose on.
    """
    bits = tr.bitsets()
    if exclude is not None and exclude.any():
        keep = np.packbits((~exclude).astype(np.uint8))
        bits = bits & keep[None, :]
        counts = POPCNT[bits].sum(1)
    else:
        counts = tr.counts
    rid = tr.request_id
    idx = np.arange(0, tr.n_tok, stride)
    o = {}
    for k in range(1, kmax + 1):
        i = idx[idx + k < tr.n_tok]
        i = i[rid[i] == rid[i + k]]
        i = i[counts[i] > 0]
        if len(i) == 0:
            o[k] = float("nan")
            continue
        inter = POPCNT[bits[i] & bits[i + k]].sum(1)
        o[k] = float(np.median(inter / counts[i]))
    lp = next((k for k in range(1, kmax + 1)
               if not math.isnan(o[k]) and o[k] < 0.5), None)
    return o, lp


def signatures(tr):
    """Per-layer active-set signature id for every token.

    Keyed by a 64-bit hash of the sorted id list rather than by the list
    itself: a dense active set is ~430 bytes, so interning 61k x 28 of them
    would cost most of a gigabyte to no purpose.  Collisions at this scale are
    negligible and would only merge two signatures, never split one.
    """
    sig = np.zeros((tr.n_tok, tr.L), dtype=np.int64)
    sizes = []
    for li in range(tr.L):
        table, col = {}, sig[:, li]
        if tr.kind == "moe":
            arr = np.sort(tr.sig_src[:, li, :], axis=1)
            keys = (hash(a.tobytes()) for a in arr)
        elif getattr(tr, "unit", "band") in ("channel", "cluster"):
            rows = tr._bitrows[li]              # the packed row IS the signature
            keys = (hash(rows[t].tobytes()) for t in range(tr.n_tok))
        else:
            ids, offs = tr._layer_ids[li], tr._layer_offs[li]
            keys = (hash(np.sort(ids[offs[t]:offs[t + 1]]).tobytes())
                    for t in range(tr.n_tok))
        for t, key in enumerate(keys):
            v = table.get(key)
            if v is None:
                v = len(table)
                table[key] = v
            col[t] = v
        sizes.append(len(table))
    return sig, sizes


def dedupe_ordered(a):
    """Unique values of `a`, keeping first-appearance (access) order."""
    if len(a) == 0:
        return a
    _, first = np.unique(a, return_index=True)
    return a[np.sort(first)]


def conditional_entropy(sig):
    """§6.2  H_l = H(sig_{l+1} | sig_l), plug-in estimator, bits.

    WARNING: this is the estimator the task specifies, and it is badly biased
    downwards exactly where it matters.  If every token has its own signature,
    every observed p(b|a) equals 1 and the estimate collapses to 0 -- reading
    as "perfectly deterministic routing" when the truth is "maximally random".
    Kept for reference; `conditional_entropy_cv` is what the report uses.
    """
    T, L = sig.shape
    H = []
    for li in range(L - 1):
        a, b = sig[:, li], sig[:, li + 1]
        pair = Counter(zip(a.tolist(), b.tolist()))
        prev = Counter(a.tolist())
        h = -sum((c / T) * math.log2(c / prev[k[0]]) for k, c in pair.items())
        H.append(h)
    return H


def top1_chain(tr):
    """Dominant unit per (token, layer): the highest-mass band / top router logit.

    Set signatures explode combinatorially -- on dense traces every token gets
    its own signature, so any entropy over them is unestimable.  The dominant
    unit has an alphabet of only `per_layer` symbols, so H(c_{l+1} | c_l) is
    well conditioned and is directly comparable to the 0.2-0.5 bit/layer
    reference in §6.2.
    """
    if tr.kind == "moe":
        return tr.sig_src[:, :, 0].astype(np.int64)      # highest router logit
    if getattr(tr, "unit", "band") in ("channel", "cluster"):
        return tr.top1                                   # stored during capture
    c = np.zeros((tr.n_tok, tr.L), dtype=np.int64)
    for li in range(tr.L):
        # band ids were stored in descending-mass order, so element 0 is top-1
        c[:, li] = tr._layer_ids[li][tr._layer_offs[li][:-1]]
    return c


def conditional_entropy_cv(sig, alpha=0.5, split=0.5):
    """§6.2, held-out estimator that does not collapse on unique signatures.

    P(sig_{l+1} | sig_l) is fitted with add-alpha smoothing on the first half
    of the trace and scored as mean -log2 P on the second half.  Contexts never
    seen in training fall back to the marginal.  Symbols never seen in training
    are charged to a single "novel" bucket, so the numbers are LOWER BOUNDS on
    the true conditional entropy; `novel_rate` says how much of the test mass
    landed in that bucket, i.e. how much the bound is understating things.

    Also returns the marginal H(sig_{l+1}) under the same scheme: the gap
    H_marginal - H_conditional is the information the previous layer actually
    carries about the next one, which is the quantity §6.2 is really after.
    """
    T, L = sig.shape
    cut = int(T * split)
    cond, marg, novel = [], [], []
    for li in range(L - 1):
        a_tr, b_tr = sig[:cut, li].tolist(), sig[:cut, li + 1].tolist()
        a_te, b_te = sig[cut:, li].tolist(), sig[cut:, li + 1].tolist()
        joint, ctx = defaultdict(Counter), Counter()
        marg_c = Counter(b_tr)
        for x, y in zip(a_tr, b_tr):
            joint[x][y] += 1
            ctx[x] += 1
        K = len(marg_c) + 1                      # +1 novel bucket
        n_tr = len(b_tr)
        hc = hm = 0.0
        nv = 0
        for x, y in zip(a_te, b_te):
            seen = y in marg_c
            nv += not seen
            pm = (marg_c[y] + alpha) / (n_tr + alpha * K)
            hm -= math.log2(pm)
            if x in ctx:
                pc = (joint[x][y] + alpha) / (ctx[x] + alpha * K)
            else:
                pc = pm
            hc -= math.log2(pc)
        n_te = max(len(b_te), 1)
        cond.append(hc / n_te)
        marg.append(hm / n_te)
        novel.append(nv / n_te)
    return dict(H_cond=cond, H_marg=marg, novel_rate=novel,
                info_gain=[m - c for m, c in zip(marg, cond)])


def continuation_concentration(sig, min_visits=5):
    """§6.4 median p1, p1+p2, p1+p2+p3 over nodes."""
    T, L = sig.shape
    p1, p2, p3 = [], [], []
    for li in range(L - 1):
        succ = defaultdict(Counter)
        a, b = sig[:, li].tolist(), sig[:, li + 1].tolist()
        for x, y in zip(a, b):
            succ[x][y] += 1
        for c in succ.values():
            tot = sum(c.values())
            if tot < min_visits:
                continue
            top = [v / tot for _, v in c.most_common(3)]
            p1.append(top[0])
            p2.append(sum(top[:2]))
            p3.append(sum(top[:3]))
    return dict(p1=float(np.median(p1)), p12=float(np.median(p2)),
                p123=float(np.median(p3)), nodes=len(p1))


def saturation(tr, sig, points=60):
    """§6.3 unique units / transitions / full paths vs requests processed."""
    rid = tr.request_id
    ends = np.concatenate([np.nonzero(np.diff(rid))[0] + 1, [tr.n_tok]])
    marks = set(np.linspace(0, len(ends) - 1, points).astype(int).tolist())
    seen_u, seen_tr, seen_path = set(), set(), set()
    seen_bits = np.zeros(tr._bits.shape[1], np.uint8) if tr.flat is None else None
    curve, start = [], 0
    for i, end in enumerate(ends):
        if tr.flat is None:                     # OR the rows, never expand ids
            seen_bits |= np.bitwise_or.reduce(tr._bits[start:end], axis=0)
        else:
            seen_u.update(tr.flat[tr.offsets[start]:tr.offsets[end]].tolist())
        block = sig[start:end]
        for li in range(tr.L - 1):
            seen_tr.update(zip([li] * len(block), block[:, li].tolist(),
                               block[:, li + 1].tolist()))
        seen_path.update(r.tobytes() for r in block)
        start = end
        if i in marks:
            n_u = int(POPCNT[seen_bits].sum()) if seen_bits is not None \
                else len(seen_u)
            curve.append([i + 1, n_u, len(seen_tr), len(seen_path)])
    return curve


def gap_quantiles(tr):
    """§7 router cut gap, per layer."""
    return {li: dict(q05=float(np.percentile(tr.gap[:, li].astype(np.float32), 5)),
                     q25=float(np.percentile(tr.gap[:, li].astype(np.float32), 25)),
                     q50=float(np.percentile(tr.gap[:, li].astype(np.float32), 50)),
                     mean=float(tr.gap[:, li].astype(np.float32).mean()))
            for li in range(tr.L)}


# ---------------------------------------------------------------- main ------

def analyse(path, out_json):
    tr = Trace(path)
    res = {"trace": tr.describe()}
    print(json.dumps(res["trace"], indent=1))

    table = {}
    for M in M_GRID:
        cap = max(1, round(tr.U * M / 100))
        table[M] = {}
        for pol in POLICIES:
            r = simulate(tr, cap, pol, per_layer=(M == 5 and pol == "EPOCH"))
            if "q_by_layer" in r:
                res["q_by_layer_M5_EPOCH"] = r.pop("q_by_layer")
            table[M][pol] = r
            ch = "-" if r["churn"] is None else f"{r['churn']:.3f}"
            print(f"  M={M:2d}% cap={cap:5d} {pol:6s} q={r['q']:.4f} "
                  f"steady={r['q_steady']:.4f} warm={r['q_warm']:.4f} churn={ch}")
    res["q_table"] = table

    want_cap = round(TARGET_CACHE_OVER_ACTIVE * tr.mean_active)
    iso_cap = max(1, min(want_cap, tr.U))
    r = simulate(tr, iso_cap, "EPOCH")
    res["iso_ratio_point"] = dict(cache_over_active=round(TARGET_CACHE_OVER_ACTIVE, 3),
                                  M_percent=round(100 * iso_cap / tr.U, 2),
                                  clamped=bool(want_cap > tr.U),
                                  wanted_M_percent=round(100 * want_cap / tr.U, 2),
                                  **r)
    if want_cap > tr.U:
        print(f"  NOTE: matching the target cache/activation ratio would need "
              f"M = {100*want_cap/tr.U:.0f} % of the units, i.e. more than the "
              f"model has. This unit definition is too coarse to be sparse.")
    print(f"  iso-ratio (cache/active = 1.60 as in the 120B target): "
          f"M={res['iso_ratio_point']['M_percent']}%  "
          f"q_steady={r['q_steady']:.4f}")

    core, rate = core_units(tr)
    n_core = int(core.sum())
    core_share = float(rate[core].sum() / tr.mean_active) if n_core else 0.0
    res["core"] = dict(threshold=0.9, n_core=n_core,
                       core_of_units=round(n_core / tr.U, 4),
                       core_of_working_set=round(core_share, 4),
                       variable_per_token=round(tr.mean_active * (1 - core_share), 1))
    print(f"  always-on core: {n_core} units ({100*n_core/tr.U:.1f} % of the model), "
          f"{100*core_share:.1f} % of the working set; "
          f"{res['core']['variable_per_token']} units per token are variable")

    ok, lp = persistence(tr)
    res["persistence"] = {"o_k": ok, "L_persist": lp}
    print(f"  L_persist={lp}  o_1={ok[1]:.3f} o_8={ok[8]:.3f} o_32={ok[32]:.3f}")
    if n_core:
        okv, lpv = persistence(tr, exclude=core)
        res["persistence_variable"] = {"o_k": okv, "L_persist": lpv}
        print(f"  excluding the core: L_persist={lpv}  o_1={okv[1]:.3f} "
              f"o_8={okv[8]:.3f} o_32={okv[32]:.3f}")

    sig, sizes = signatures(tr)
    res["signature_counts"] = sizes
    res["H_layer_plugin"] = conditional_entropy(sig)
    cv = conditional_entropy_cv(sig)
    res["H_layer_cv"] = cv
    res["H_layer"] = cv["H_cond"]
    res["N_eff_log2"] = float(sum(cv["H_cond"]))
    res["N_eff_log2_plugin"] = float(sum(res["H_layer_plugin"]))
    print(f"  distinct signatures/layer: min={min(sizes)} max={max(sizes)} "
          f"(trace has {tr.n_tok} tokens)")
    print(f"  held-out  sum H(l+1|l) = {sum(cv['H_cond']):.2f} bits "
          f"-> N_eff = 2^{sum(cv['H_cond']):.2f}")
    print(f"  held-out  sum H(l+1)   = {sum(cv['H_marg']):.2f} bits, "
          f"info gain = {sum(cv['info_gain']):.2f} bits, "
          f"novel-symbol rate = {np.mean(cv['novel_rate']):.3f}")
    print(f"  plug-in (task §6.2, biased) sum = {sum(res['H_layer_plugin']):.2f} bits")
    c1 = top1_chain(tr)
    cv1 = conditional_entropy_cv(c1)
    res["top1_chain"] = dict(
        H_cond=cv1["H_cond"], H_marg=cv1["H_marg"],
        info_gain=cv1["info_gain"], novel_rate=cv1["novel_rate"],
        sum_H_cond=float(sum(cv1["H_cond"])),
        sum_info_gain=float(sum(cv1["info_gain"])),
        alphabet=int(tr.per_layer))
    res["top1_continuation"] = continuation_concentration(c1)
    print(f"  dominant-unit chain (alphabet {tr.per_layer}): "
          f"sum H(c_l+1|c_l) = {sum(cv1['H_cond']):.2f} bits "
          f"({sum(cv1['H_cond'])/max(len(cv1['H_cond']),1):.3f} bit/layer), "
          f"info gain = {sum(cv1['info_gain']):.2f} bits")
    print(f"  dominant-unit continuation: p1={res['top1_continuation']['p1']:.3f} "
          f"p1+p2={res['top1_continuation']['p12']:.3f} "
          f"p1+p2+p3={res['top1_continuation']['p123']:.3f}")

    res["continuation"] = continuation_concentration(sig)
    res["saturation"] = saturation(tr, sig)
    if tr.kind == "moe":
        res["gap"] = gap_quantiles(tr)
    else:
        res["channel_vs_band"] = dict(
            mean_active_bands=float(tr.band_cnt.mean()),
            mean_active_channels=float(tr.chan_cnt.mean()),
            band_frac=float(tr.band_cnt.mean() / tr.n_bands),
            channel_frac=float(tr.chan_cnt.mean() / tr.d_ffn))

    Path(out_json).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"  wrote {out_json}")
    return res


if __name__ == "__main__":
    p = sys.argv[1]
    o = sys.argv[2] if len(sys.argv) > 2 else str(
        ROOT / "out" / (Path(p).stem + ".analysis.json"))
    analyse(p, o)
