"""Build the SWE1 report from a run file. Numbers only, no interpretation."""
import json, glob, os, sys, collections

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from kan import score as S  # noqa

ARMS = ("trunc", "solo", "kan")
LBL = {"trunc": "TRUNC", "solo": "SOLO", "kan": "KAN"}


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "swe1"
    fs = sorted(glob.glob(os.path.join(HERE, "runs", f"{tag}_*.json")))
    if not fs:
        sys.exit(f"no runs matching {tag}_*.json")
    path = fs[-1]
    data = json.load(open(path, encoding="utf-8"))
    n = len(data)
    full = [r for r in data if len(r["gold_in_surface"]) == len(r["gold_files"])
            and r["gold_files"]]
    print(f"# SWE1 — {os.path.basename(path)}   n={n}   "
          f"(full gold coverage: {len(full)})\n")

    # ------------------------------------------------------------- regime
    ratios = [r["window_ratio"] for r in data]
    print("## Regime\n")
    print(f"- surface / window: min {min(ratios):.2f}x, median "
          f"{sorted(ratios)[len(ratios)//2]:.2f}x, max {max(ratios):.2f}x")
    print(f"- instances in target regime (>1x): {sum(1 for x in ratios if x > 1)} of {n}")
    print(f"- slices: {sum(r['n_slices'] for r in data)} total, "
          f"{min(r['n_slices'] for r in data)}-{max(r['n_slices'] for r in data)} per instance")
    print(f"- files split by a cut: {sum(r['files_split'] for r in data)}; "
          f"symbols straddling a cut: {sum(r['symbols_straddling'] for r in data)}")
    cov = collections.Counter(
        "full" if len(r["gold_in_surface"]) == len(r["gold_files"]) else
        ("none" if not r["gold_in_surface"] else "partial") for r in data)
    print(f"- gold coverage by the surface: {dict(cov)}")

    # ------------------------------------------------------------ per arm
    def agg(rows, arm, fn):
        vals = [fn(r[arm]) for r in rows if arm in r]
        return sum(vals) / max(1, len(vals))

    def block(rows, title):
        print(f"\n## {title}  (n={len(rows)})\n")
        print("| metric | TRUNC | SOLO | KAN |")
        print("|---|---:|---:|---:|")
        rowspec = [
            ("file_recall", lambda a: a["score"]["file_recall"], "{:.3f}"),
            ("file_precision", lambda a: a["score"]["file_precision"], "{:.3f}"),
            ("hunk_line_recall", lambda a: a["score"]["hunk_line_recall"], "{:.3f}"),
            ("normcontrol passed", lambda a: 1.0 if a["normcontrol"] else 0.0, "{:.3f}"),
            ("non-empty patch", lambda a: 0.0 if a["score"]["pred_empty"] else 1.0, "{:.3f}"),
            ("edits raw", lambda a: a["raw_edits"], "{:.2f}"),
            ("edits kept", lambda a: a["kept"], "{:.2f}"),
            ("VDP survival", lambda a: (a["kept"] / a["raw_edits"]) if a["raw_edits"] else 0.0, "{:.3f}"),
            ("MDP unresolved", lambda a: a["mdp_unresolved"], "{:.2f}"),
            ("slices read", lambda a: a["slices_read"], "{:.1f}"),
            ("chars read", lambda a: a["read_chars"], "{:.0f}"),
            ("calls", lambda a: a["totals"]["calls"], "{:.1f}"),
            ("prompt tokens", lambda a: a["totals"]["prompt_tokens"], "{:.0f}"),
            ("completion tokens", lambda a: a["totals"]["completion_tokens"], "{:.0f}"),
            ("declined (NO EDITS)", lambda a: a["declined"], "{:.1f}"),
            ("truncated answers", lambda a: a["truncated"], "{:.2f}"),
            ("wall minutes", lambda a: a["wall_seconds"] / 60, "{:.1f}"),
        ]
        for name, fn, fmt in rowspec:
            vals = [agg(rows, a, fn) for a in ARMS]
            print(f"| {name} | " + " | ".join(fmt.format(v) for v in vals) + " |")

    block(data, "All instances")
    if full and len(full) != n:
        block(full, "Subset with full gold coverage")

    # -------------------------------------------------- pre-registered
    print("\n## Pre-registered thresholds\n")
    fr = {a: agg(data, a, lambda x: x["score"]["file_recall"]) for a in ARMS}
    hr = {a: agg(data, a, lambda x: x["score"]["hunk_line_recall"]) for a in ARMS}
    normk = agg(data, "kan", lambda a: 1.0 if a["normcontrol"] else 0.0)
    vdpk = agg(data, "kan", lambda a: (a["kept"] / a["raw_edits"]) if a["raw_edits"] else 0.0)
    mdpk = agg(data, "kan", lambda a: a["mdp_unresolved"])
    mdps = agg(data, "solo", lambda a: a["mdp_unresolved"])

    def verdict(v, pred, fals):
        if fals(v):
            return "FALSIFIED"
        return "CONFIRMED" if pred(v) else "not confirmed"

    rows = [
        ("SWE1-decomp", "file_recall(SOLO) - file_recall(TRUNC)", fr["solo"] - fr["trunc"],
         ">= +0.15", "<= 0", lambda v: v >= 0.15, lambda v: v <= 0),
        ("SWE1-kan", "file_recall(KAN) - file_recall(SOLO)", fr["kan"] - fr["solo"],
         "> 0", "<= 0", lambda v: v > 0, lambda v: v <= 0),
        ("SWE1-hunk", "hunk_recall(KAN) - hunk_recall(SOLO)", hr["kan"] - hr["solo"],
         "> 0", "<= 0", lambda v: v > 0, lambda v: v <= 0),
        ("SWE1-norm", "normcontrol rate, KAN", normk,
         ">= 0.70", "< 0.50", lambda v: v >= 0.70, lambda v: v < 0.50),
        ("SWE1-vdp", "VDP survival, KAN", vdpk,
         ">= 0.60", "< 0.35", lambda v: v >= 0.60, lambda v: v < 0.35),
        ("SWE1-mdp", "MDP unresolved, KAN - SOLO", mdpk - mdps,
         "<= 0", "> 0", lambda v: v <= 0, lambda v: v > 0),
        ("SWE1-floor", "file_recall TRUNC", fr["trunc"],
         "-", "< 0.15 -> out of range", lambda v: True, lambda v: v < 0.15),
    ]
    print("| code | quantity | measured | prediction | falsification | verdict |")
    print("|---|---|---:|---|---|---|")
    for code, what, val, pred, fals, pf, ff in rows:
        print(f"| **{code}** | {what} | **{val:+.3f}** | {pred} | {fals} | "
              f"{verdict(val, pf, ff)} |")

    # --------------------------------------------- paired, not just means
    # At n=20 a difference of means sits inside its own noise. The paired count
    # is where the resolution actually is -- the M1 lesson: put the metric where
    # the comparison is, not on the end-to-end average.
    print("\n## Paired comparison (per instance, same surface, same slices)\n")
    print("| comparison | metric | wins | losses | ties | sign-test p |")
    print("|---|---|---:|---:|---:|---:|")
    from math import comb

    def sign_p(w, l):
        k, m = min(w, l), w + l
        if m == 0:
            return 1.0
        return min(1.0, 2 * sum(comb(m, i) for i in range(k + 1)) / 2 ** m)

    for a, b, label in (("kan", "solo", "KAN vs SOLO"), ("solo", "trunc", "SOLO vs TRUNC")):
        for key, mname in (("file_recall", "file_recall"),
                           ("hunk_line_recall", "hunk_recall")):
            w = sum(1 for r in data if r[a]["score"][key] > r[b]["score"][key])
            l = sum(1 for r in data if r[a]["score"][key] < r[b]["score"][key])
            t = len(data) - w - l
            print(f"| {label} | {mname} | {w} | {l} | {t} | {sign_p(w, l):.3f} |")

    # ----------------------------------------------------- rejections
    print("\n## VDP rejections\n")
    per = {a: collections.Counter() for a in ARMS}
    for r in data:
        for a in ARMS:
            for d in r[a].get("dropped", []):
                per[a][d["reason"]] += 1
    reasons = sorted({k for a in ARMS for k in per[a]},
                     key=lambda k: -sum(per[a][k] for a in ARMS))
    print("| reason | TRUNC | SOLO | KAN |")
    print("|---|---:|---:|---:|")
    for k in reasons:
        print(f"| `{k}` | " + " | ".join(str(per[a][k]) for a in ARMS) + " |")

    # ------------------------------------------------------ per instance
    print("\n## Per instance\n")
    print("| repo | ratio | slices | gold | in surf | "
          "TRUNC f/h | SOLO f/h | KAN f/h |")
    print("|---|---:|---:|---:|---:|---|---|---|")
    for r in data:
        c = []
        for a in ARMS:
            s = r[a]["score"]
            c.append(f"{s['file_recall']:.2f}/{s['hunk_line_recall']:.2f}")
        print(f"| {r['repo'].split('/')[-1][:12]} | {r['window_ratio']:.2f}x | "
              f"{r['n_slices']} | {len(r['gold_files'])} | {len(r['gold_in_surface'])} | "
              + " | ".join(c) + " |")

    tot = sum(r["instance_seconds"] for r in data) / 3600
    print(f"\n**{tot:.2f} h for {n} instances = {tot*60/n:.1f} min/instance (three arms)**")


if __name__ == "__main__":
    main()
