"""SWE1 driver: TRUNC / SOLO_SERIAL / KAN in the target regime.

Target regime means the surface does not fit the window. That is checked and
recorded per instance, and an instance whose surface DOES fit is reported as
out-of-regime rather than quietly averaged in -- the whole point of the rewrite.
"""
import os, sys, json, time, argparse, traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kan import llm, retrieve, pipeline, score, slices as SL, arms, seams as SE  # noqa

HERE = os.path.dirname(os.path.abspath(__file__))
REPOS = r"H:\swebench_pro_repos"
OUTDIR = os.path.join(HERE, "runs")


def build_surface(root, language, query, pool_files):
    ranked = retrieve.build_pool(root, language, query, pool_files=pool_files)[0]
    files = [(rel, retrieve.read_file_raw(root, rel)) for _, rel in ranked]
    files = [(rel, t) for rel, t in files if t]
    return SL.Surface(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", default=os.path.join(HERE, "data", "swe1.json"))
    ap.add_argument("--pool", type=int, default=8)
    ap.add_argument("--slice-chars", type=int, default=30000,
                    help="~8.3k tokens; leaves room in a 16384 window for output")
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--window-tokens", type=int, default=16384)
    ap.add_argument("--tag", default="swe1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--arms", default="trunc,solo,kan")
    ap.add_argument("--deadline-hours", type=float, default=0,
                    help="stop cleanly BETWEEN instances once this much wall "
                         "clock has passed; 0 = no limit")
    args = ap.parse_args()

    if not llm.health():
        print("llama-server is not answering on", llm.HOST)
        return 2

    rows = json.load(open(args.instances, encoding="utf-8"))
    if args.limit:
        rows = rows[:args.limit]
    os.makedirs(OUTDIR, exist_ok=True)
    outpath = os.path.join(OUTDIR, f"{args.tag}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    want = set(args.arms.split(","))

    results, t_all = [], time.time()
    deadline = t_all + args.deadline_hours * 3600 if args.deadline_hours else None
    stopped_early = False
    for i, inst in enumerate(rows, 1):
        # Stop BETWEEN instances only. A half-finished instance would carry one
        # arm and not the others, and a per-instance comparison built on that is
        # worse than a missing row.
        if deadline and time.time() >= deadline:
            print(f"\n=== deadline reached after {len(results)} instances, stopping cleanly")
            stopped_early = True
            break
        iid = inst["instance_id"]
        root = os.path.join(REPOS, iid)
        lang = inst.get("repo_language", "python")
        el = (time.time() - t_all) / 3600
        print(f"\n=== [{i}/{len(rows)}] {inst['repo']}  {iid[:46]}   (+{el:.1f}h elapsed)")
        if not os.path.isdir(os.path.join(root, ".git")):
            print("  no checkout, skipping")
            continue

        pipeline.reset(root)
        t0 = time.time()
        query = inst["problem_statement"] + "\n" + inst["requirements"] + "\n" + inst["interface"]
        surface = build_surface(root, lang, query, args.pool)
        sl = SL.cut(surface, args.slice_chars)
        window_chars = args.window_tokens * 3.6
        ratio = surface.total / window_chars
        split = SL.seam_index(sl)
        crossing = SL.split_across(sl, lang)
        gold = [g for g in score.files_of(inst["gold_patch"]) if not score.is_doc(g)]
        in_surface = [g for g in gold if g in surface.rel_list()]

        print(f"  surface {surface.total} chars over {len(surface.files)} files "
              f"= {ratio:.2f}x window -> {len(sl)} slices")
        print(f"  files split by a cut: {len(split)}   symbols straddling a cut: {len(crossing)}")
        print(f"  gold code files {len(gold)}, in surface {len(in_surface)}")
        if ratio <= 1.0:
            print("  *** OUT OF REGIME: surface fits one window, decomposition buys nothing")

        rec = dict(instance_id=iid, repo=inst["repo"], gold_files=gold,
                   gold_in_surface=in_surface, surface_chars=surface.total,
                   surface_files=surface.rel_list(), window_ratio=round(ratio, 2),
                   n_slices=len(sl), files_split=len(split),
                   symbols_straddling=len(crossing), in_regime=ratio > 1.0,
                   n_requirements=len(inst["req_bullets"]),
                   setup_seconds=round(time.time() - t0, 1))

        blocks = pipeline.split_requirements(inst["req_bullets"], args.blocks)
        plan = [("trunc", arms.run_trunc, {}),
                ("solo", arms.run_solo_serial, {}),
                ("kan", arms.run_kan, dict(n_blocks=len(blocks), blocks_of_reqs=blocks))]

        for arm, fn, kw in plan:
            if arm not in want:
                continue
            llm.reset()
            pipeline.reset(root)
            t1 = time.time()
            try:
                r = fn(inst, root, surface, sl, args.max_tokens, None, **kw)
            except Exception as ex:
                traceback.print_exc()
                r = dict(kept=[], dropped=[], calls=list(llm.CALL_LOG), answers=[],
                         raw=0, read_chars=0, slices_read=0, declined=0,
                         truncated=0, error=str(ex))
            applied, conflicts = arms.E.apply_edits(r["kept"], root)
            patch, ok = pipeline.cut_patch(root)
            md = SE.mdp(r["kept"], surface, lang)
            sc = score.score(patch, inst["gold_patch"])
            out = dict(arm=arm, raw_edits=r["raw"], kept=len(r["kept"]),
                       dropped=r["dropped"], applied=len(applied),
                       conflicts=len(conflicts), patch=patch, normcontrol=ok,
                       calls=r["calls"], raw_answers=r["answers"],
                       read_chars=r["read_chars"], slices_read=r["slices_read"],
                       declined=r["declined"], truncated=r["truncated"],
                       mdp_unresolved=len(md["unresolved"]),
                       mdp_cross=len(md["cross_discipline"]), mdp=md,
                       blocks=r.get("blocks"),
                       files_touched=sorted({e["file"] for e in r["kept"]}),
                       wall_seconds=round(time.time() - t1, 1),
                       totals=llm.totals(), score=sc)
            rec[arm] = out
            print(f"  {arm.upper():5s} {out['wall_seconds']/60:5.1f} min  "
                  f"calls={out['totals']['calls']:2d} in={out['totals']['prompt_tokens']:6d} "
                  f"out={out['totals']['completion_tokens']:5d}  read={out['read_chars']:6d}c  "
                  f"raw={out['raw_edits']:2d} kept={out['kept']:2d} decl={out['declined']} "
                  f"trunc={out['truncated']} norm={'y' if ok else 'n'}  "
                  f"file={sc['file_recall']:.2f} hunk={sc['hunk_line_recall']:.2f} "
                  f"mdp={out['mdp_unresolved']}")

        rec["instance_seconds"] = round(time.time() - t0, 1)
        results.append(rec)
        json.dump(results, open(outpath, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    total = time.time() - t_all
    print(f"\n=== TOTAL {total/3600:.2f} h for {len(results)} of {len(rows)} instances "
          f"({total/60/max(1,len(results)):.1f} min/instance)"
          + ("  [stopped at deadline]" if stopped_early else ""))
    print("wrote", outpath)
    return 0


if __name__ == "__main__":
    sys.exit(main())
