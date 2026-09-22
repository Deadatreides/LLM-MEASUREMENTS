"""Pilot driver: N instances x {KAN, SOLO}, full accounting of wall time.

The pilot exists to replace an estimate with a measurement. Every call's token
counts and seconds are recorded so the scale decision is made on numbers.
"""
import os, sys, json, time, argparse, traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kan import llm, retrieve, pipeline, score  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPOS = r"H:\swebench_pro_repos"
DEFAULT_IN = os.path.join(HERE, "data", "pilot3.json")
OUTDIR = os.path.join(HERE, "runs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", default=DEFAULT_IN)
    ap.add_argument("--pool", type=int, default=8, help="files in the retrieval pool")
    ap.add_argument("--ctx-chars", type=int, default=34000,
                    help="code context budget, shared identically by both arms")
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if not llm.health():
        print("llama-server is not answering on", llm.HOST)
        return 2

    rows = json.load(open(args.instances, encoding="utf-8"))
    if args.limit:
        rows = rows[:args.limit]
    os.makedirs(OUTDIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    outpath = os.path.join(OUTDIR, f"{args.tag}_{stamp}.json")

    results, t_all = [], time.time()
    for i, inst in enumerate(rows, 1):
        iid = inst["instance_id"]
        root = os.path.join(REPOS, iid)
        lang = "python"
        print(f"\n=== [{i}/{len(rows)}] {inst['repo']}  {iid[:48]}")
        if not os.path.isdir(os.path.join(root, ".git")):
            print("  no checkout, skipping")
            continue

        pipeline.reset(root)
        t0 = time.time()
        query = inst["problem_statement"] + "\n" + inst["requirements"] + "\n" + inst["interface"]
        pool, n_files = retrieve.build_pool(root, lang, query, pool_files=args.pool)
        t_ret = time.time() - t0
        gold = score.files_of(inst["gold_patch"])
        pool_files = [rel for _, rel in pool]
        print(f"  retrieval: {n_files} files indexed in {t_ret:.1f}s -> pool {len(pool)}")
        print(f"    pool: {pool_files}")
        print(f"    gold: {gold}")
        print(f"    gold in pool: {sorted(set(gold) & set(pool_files))}")

        rec = dict(instance_id=iid, repo=inst["repo"], gold_files=gold,
                   pool=pool_files, retrieval_seconds=round(t_ret, 1),
                   indexed_files=n_files,
                   gold_in_pool=sorted(set(gold) & set(pool_files)),
                   n_requirements=len(inst["req_bullets"]))

        for arm, fn in (("solo", pipeline.run_solo), ("kan", pipeline.run_kan)):
            llm.reset()
            t1 = time.time()
            try:
                # Same per-call ceiling for both arms, set high enough that
                # neither hits it. Giving SOLO the pooled budget looked fair on
                # paper but clipped every long Kan block at 1400 while SOLO,
                # writing once, never came near its 4200 -- that measures the
                # ceiling, not the scheme. Actual consumption is reported.
                kw = dict(char_budget=args.ctx_chars, max_tokens=args.max_tokens)
                if arm == "kan":
                    kw["n_blocks"] = args.blocks
                out = fn(inst, root, pool, lang, **kw)
            except Exception as ex:
                traceback.print_exc()
                out = dict(arm=arm, error=str(ex), patch="", normcontrol=False,
                           calls=list(llm.CALL_LOG))
            wall = time.time() - t1
            sc = score.score(out.get("patch", ""), inst["gold_patch"])
            out["wall_seconds"] = round(wall, 1)
            out["totals"] = llm.totals()
            out["score"] = sc
            rec[arm] = out
            print(f"  {arm.upper():4s} {wall/60:5.1f} min  calls={out['totals']['calls']} "
                  f"in={out['totals']['prompt_tokens']} out={out['totals']['completion_tokens']}  "
                  f"raw={out.get('raw_edits','-')} kept={out.get('kept','-')} "
                  f"norm={out.get('normcontrol')}  "
                  f"file_recall={sc['file_recall']:.2f} hunk_recall={sc['hunk_line_recall']:.2f}")

        rec["instance_seconds"] = round(time.time() - t0, 1)
        results.append(rec)
        json.dump(results, open(outpath, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    total = time.time() - t_all
    print(f"\n=== TOTAL {total/60:.1f} min for {len(results)} instances "
          f"({total/60/max(1,len(results)):.1f} min/instance)")
    print("wrote", outpath)
    return 0


if __name__ == "__main__":
    sys.exit(main())
