"""Build the pilot report from a run file. Numbers only, no interpretation."""
import json, glob, os, sys, collections

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")


def load(tag="pilot3"):
    fs = sorted(glob.glob(os.path.join(RUNS, f"{tag}_*.json")))
    if not fs:
        sys.exit(f"no runs matching {tag}_*.json")
    return fs[-1], json.load(open(fs[-1], encoding="utf-8"))


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "pilot3"
    path, data = load(tag)
    n = len(data)
    print(f"# Pilot run — {os.path.basename(path)}   n={n}\n")

    print("## Per instance\n")
    print("| repo | arm | min | calls | in tok | out tok | raw | kept | norm | file_recall | hunk_recall |")
    print("|---|---|---:|---:|---:|---:|---:|---:|:-:|---:|---:|")
    for r in data:
        for arm in ("solo", "kan"):
            a = r[arm]
            s = a["score"]
            print(f"| {r['repo'].split('/')[-1]} | {arm.upper()} | {a['wall_seconds']/60:.1f} | "
                  f"{a['totals']['calls']} | {a['totals']['prompt_tokens']} | "
                  f"{a['totals']['completion_tokens']} | {a.get('raw_edits',0)} | {a.get('kept',0)} | "
                  f"{'yes' if a['normcontrol'] else 'no'} | {s['file_recall']:.2f} | "
                  f"{s['hunk_line_recall']:.2f} |")

    print("\n## Aggregate\n")
    print("| metric | SOLO | KAN | KAN - SOLO |")
    print("|---|---:|---:|---:|")

    def avg(arm, fn):
        return sum(fn(r[arm]) for r in data) / max(1, n)

    rows = [
        ("normcontrol passed", lambda a: 1.0 if a["normcontrol"] else 0.0),
        ("file_recall", lambda a: a["score"]["file_recall"]),
        ("file_precision", lambda a: a["score"]["file_precision"]),
        ("hunk_line_recall", lambda a: a["score"]["hunk_line_recall"]),
        ("edits kept / raw", lambda a: (a.get("kept", 0) / a["raw_edits"]) if a.get("raw_edits") else 0.0),
        ("wall minutes", lambda a: a["wall_seconds"] / 60),
        ("prompt tokens", lambda a: a["totals"]["prompt_tokens"]),
        ("completion tokens", lambda a: a["totals"]["completion_tokens"]),
        ("context chars read", lambda a: a.get("context_chars", 0)),
    ]
    for name, fn in rows:
        s, k = avg("solo", fn), avg("kan", fn)
        fmt = "{:.0f}" if abs(s) > 100 or abs(k) > 100 else "{:.3f}"
        print(f"| {name} | {fmt.format(s)} | {fmt.format(k)} | {fmt.format(k - s)} |")

    print("\n## VDP rejections\n")
    reasons = collections.Counter()
    per_arm = {"solo": collections.Counter(), "kan": collections.Counter()}
    for r in data:
        for arm in ("solo", "kan"):
            for d in r[arm].get("dropped", []):
                reasons[d["reason"]] += 1
                per_arm[arm][d["reason"]] += 1
    print("| reason | SOLO | KAN | total |")
    print("|---|---:|---:|---:|")
    for reason, tot in reasons.most_common():
        print(f"| `{reason}` | {per_arm['solo'][reason]} | {per_arm['kan'][reason]} | {tot} |")

    trunc = sum(1 for r in data for arm in ("solo", "kan")
                for c in r[arm]["calls"] if c["finish"] == "length")
    calls = sum(len(r[arm]["calls"]) for r in data for arm in ("solo", "kan"))
    print(f"\ntruncated answers: {trunc} of {calls} calls")

    print("\n## Retrieval\n")
    print("| repo | indexed | pool | gold code files | in pool |")
    print("|---|---:|---:|---|---|")
    for r in data:
        from kan import score as S
        gold_code = [g for g in r["gold_files"] if not S.is_doc(g)]
        inpool = [g for g in gold_code if g in r["pool"]]
        print(f"| {r['repo'].split('/')[-1]} | {r['indexed_files']} | {len(r['pool'])} | "
              f"{len(gold_code)} | {len(inpool)} |")

    tot_min = sum(r["instance_seconds"] for r in data) / 60
    print(f"\n**total {tot_min:.1f} min for {n} instances = "
          f"{tot_min/max(1,n):.1f} min/instance (both arms)**")


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    main()
