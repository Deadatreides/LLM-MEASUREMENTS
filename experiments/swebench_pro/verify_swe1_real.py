"""Second pass: re-check the polygon on the REAL instance set, not synthetics.

Different assertions from verify_swe1.py on purpose. Two things a synthetic
fixture cannot show: whether the target regime actually holds on these repos,
and whether anything about the gold patch leaks into a prompt.
"""
import os, sys, json, re

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from kan import retrieve, slices as SL, seams as SE, edits as E, arms, pipeline, score  # noqa
from run_swe1 import build_surface  # noqa

REPOS = r"H:\swebench_pro_repos"
WINDOW_TOK = 16384
SLICE_CHARS = 30000
POOL = 6
OK, BAD = [], []


def check(name, cond, detail=""):
    (OK if cond else BAD).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    return cond


def main():
    path = os.path.join(HERE, "data", "swe1.json")
    rows = json.load(open(path, encoding="utf-8"))
    print(f"instance set: {len(rows)}  (pool={POOL}, slice={SLICE_CHARS})\n")

    tot_slices = 0
    for inst in rows:
        root = os.path.join(REPOS, inst["instance_id"])
        lang = inst.get("repo_language", "python")
        print(f"--- {inst['repo']}  {inst['instance_id'][:40]}")
        if not os.path.isdir(os.path.join(root, ".git")):
            check("checkout present", False)
            continue
        pipeline.reset(root)
        q = inst["problem_statement"] + "\n" + inst["requirements"] + "\n" + inst["interface"]
        surf = build_surface(root, lang, q, POOL)
        sl = SL.cut(surf, SLICE_CHARS)
        tot_slices += len(sl)
        ratio = surf.total / (WINDOW_TOK * 3.6)

        check("IN TARGET REGIME (surface exceeds one window)", ratio > 1.0,
              f"{ratio:.2f}x, {surf.total} chars, {len(sl)} slices")

        # slices reproduce the surface byte for byte
        joined = {rel: "" for rel in surf.rel_list()}
        for s in sl:
            for rel, a, b in s.spans:
                joined[rel] += surf.text[rel][a:b]
        check("slices reproduce the surface byte for byte",
              all(joined[r] == surf.text[r] for r in surf.rel_list()))

        # surface text == what is on disk, byte for byte (offsets must be real)
        disk_ok = all(retrieve.read_file_raw(root, r) == surf.text[r]
                      for r in surf.rel_list())
        check("surface text equals the bytes on disk", disk_ok)

        split = SL.seam_index(sl)
        straddle = SL.split_across(sl, lang)
        check("cuts actually split files", len(split) >= 1,
              f"{len(split)} of {len(surf.files)} files")
        check("symbols straddle a cut (MDP has something to inspect)",
              len(straddle) >= 1, f"{len(straddle)} symbols")

        gold = [g for g in score.files_of(inst["gold_patch"]) if not score.is_doc(g)]
        inpool = [g for g in gold if g in surf.rel_list()]
        print(f"       gold code files {len(gold)}, in surface {len(inpool)}"
              + ("" if len(inpool) == len(gold) else "   <-- reachable ceiling < 1.0"))

        # --- leakage: nothing from the gold patch may reach a prompt
        blocks = pipeline.split_requirements(inst["req_bullets"], 3)
        dealt = arms.deal_slices(sl, len(blocks))
        prompts = []
        for bi, (reqs, mine) in enumerate(zip(blocks, dealt), 1):
            for s in mine:
                # must match run_kan exactly, or the test checks a prompt the
                # run never sends
                prompts.append(arms.build_prompt(inst["problem_statement"], reqs,
                                                 s.render(), s.idx, len(sl), "",
                                                 discipline=f"block {bi} of {len(blocks)}"))
        blob = "\n".join(prompts)
        # Only lines the gold patch actually INTRODUCES can leak. A line that
        # already exists somewhere at base_commit is ordinary repo text, and
        # flagging it made the first version of this check fire on `except
        # Exception:` and on docstring lines the patch merely moves.
        base = "\n".join(surf.text[r] for r in surf.rel_list())
        gold_added = [ln[1:].strip() for ln in inst["gold_patch"].split("\n")
                      if ln.startswith("+") and not ln.startswith("+++")
                      and len(ln.strip()) > 25]
        novel = [ln for ln in gold_added if ln not in base]
        leaked = [ln for ln in novel if ln in blob]
        check("no line introduced by the gold patch appears in any prompt",
              not leaked,
              f"{len(novel)} novel lines checked" + (f", {len(leaked)} LEAKED" if leaked else ""))

        # Provenance, not substring whack-a-mole. Every line of every prompt must
        # come from an allowed source: the fixed template, the dataset's own
        # input fields, or verbatim repo text at base_commit. A previous version
        # asked "is a gold path ever named" and fired on the dataset itself --
        # `requirements` and `interface` legitimately name the file to change,
        # and one repo file mentioned another in a TODO comment.
        allowed = "\n".join([arms.SYSTEM, inst["problem_statement"],
                             inst["requirements"], inst["interface"], base])
        # template lines, taken as they are actually rendered (with the block
        # number and slice number already substituted)
        TEMPLATE = re.compile(
            r"^(#\s*(Issue|Requirements|Interface)"
            r"|#\s*Your discipline \(block \d+ of \d+\)"
            r"|#\s*Slice \d+ of \d+"
            r"|#\s*What you noted in earlier slices"
            r"|You are responsible for ONLY these requirements\..*"
            r"|Produce ### EDIT blocks for this slice, or NO EDITS\."
            r"|---\s*(FILE|END)\b.*"
            r"|@@ lines \d+\.\. @@"
            r"|slice \d+: .*)$")
        stray = [ln for ln in {l.strip() for l in blob.split("\n") if len(l.strip()) > 12}
                 if ln not in allowed and not TEMPLATE.match(ln)]
        check("every prompt line traces to template, dataset fields, or repo text",
              not stray, f"{len(stray)} untraceable" + (f": {stray[:2]}" if stray else ""))
        check("fail_to_pass tests are not named in any prompt",
              str(inst["fail_to_pass"])[:40] not in blob)

        # --- every call sees exactly one slice, and the union is all slices
        seen = [s.idx for grp in dealt for s in grp]
        check("KAN: union of dealt slices == all slices, no repeats",
              sorted(seen) == list(range(len(sl))), f"{len(seen)} dealt")
        check("KAN and SOLO read identical byte counts",
              sum(s.chars() for grp in dealt for s in grp) == sum(s.chars() for s in sl))

        # --- VDP slice restriction bites on real text
        if len(sl) >= 2:
            far = sl[-1]
            rel, a, b = far.spans[0]
            lines = surf.text[rel][a:b].split("\n")
            start = next((i for i, l in enumerate(lines) if l.strip()), 0)
            anchor = "\n".join(lines[start:start + 3])
            if anchor.strip():
                kept, dropped = SE.vdp([dict(file=rel, search=anchor, replace=anchor + "\n#x")],
                                       root, surf, [sl[0]])
                check("anchor from the last slice is rejected when only the first was shown",
                      not kept and dropped and dropped[0]["reason"] == "outside_slice",
                      dropped[0]["reason"] if dropped else "kept!")
                kept, _ = SE.vdp([dict(file=rel, search=anchor, replace=anchor + "\n#x")],
                                 root, surf, [far])
                check("the same anchor is accepted when its own slice is shown", len(kept) == 1)
        print()

    print("=" * 72)
    print(f"total slices across the set: {tot_slices}  "
          f"(calls per instance = 1 + {tot_slices//max(1,len(rows))} + "
          f"{tot_slices//max(1,len(rows))} on average)")
    print(f"PASSED {len(OK)}   FAILED {len(BAD)}")
    for b in BAD:
        print("  FAILED:", b)
    return 1 if BAD else 0


if __name__ == "__main__":
    sys.exit(main())
